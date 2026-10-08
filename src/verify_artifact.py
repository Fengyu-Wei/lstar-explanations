"""Read-only verification of the supplied reference artifact; standard library only."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def number(value):
    return float(value) if value else math.nan


def bounded_path(root, relative):
    path = (root / relative.replace('\\', '/')).resolve()
    require(path.is_relative_to(root), f'Path outside artifact: {relative}')
    require(path.is_file(), f'Missing reference file: {relative}')
    return path


def verify(root):
    root = root.resolve()
    manifest = load(root / 'reference-sha256.json')
    checksums = manifest['sha256']
    require(bool(checksums), 'Empty reference manifest')
    for relative, expected in checksums.items():
        require(digest(bounded_path(root, relative)) == expected,
                f'Reference checksum mismatch: {relative}')

    datasets = {item['dataset']: item for item in load(root / 'data/manifest.json')}
    require(len(datasets) == 14, 'Expected fourteen tasks')
    splits = load(root / 'data/splits.json')
    require(len(splits) == 140, 'Expected 140 splits')
    for name, split in splits.items():
        dataset, seed = name.rsplit(':', 1)
        require(dataset in datasets and 0 <= int(seed) < 10, f'Unknown split: {name}')
        require(set(split) == {'fit', 'select', 'reference', 'query'}, name)
        pools = [set(split[k]) for k in ['fit', 'select', 'reference', 'query']]
        require(all(len(pool) == len(split[k]) for pool, k in
                    zip(pools, ['fit', 'select', 'reference', 'query'])),
                f'Duplicate split indices: {name}')
        require(all(not a.intersection(b) for a, b in itertools.combinations(pools, 2)),
                f'Overlapping pools: {name}')
        require(all(0 <= i < datasets[dataset]['n'] for pool in pools for i in pool),
                f'Invalid index: {name}')
        require(0 < len(pools[3]) <= 20, f'Invalid query cap: {name}')

    exp = root / 'experiments'
    main = exp / 'heldout-utility/results'
    tree = rows(main / 'queries.csv')
    fits = rows(main / 'fits.csv')
    require(len(fits) == 280 and len(tree) == 33600, 'Incomplete tree benchmark')
    key = ['dataset', 'seed', 'depth', 'query_index']
    methods = {'full_path', 'pure_prefix', 'exact_delete', 'empirical_prefix',
               'empirical_delete', 'certified_prefix', 'constant'}
    paired = defaultdict(list)
    for row in tree:
        paired[tuple(row[k] for k in key)].append(row['method'])
        support = int(row['ref_count'])
        precision = number(row['precision'])
        require(bool(int(row['evaluable'])) == (support > 0), 'Invalid evaluability')
        require(math.isfinite(precision) == (support > 0), 'Undefined precision lost')
        if support > 0:
            require(abs(precision - int(row['ref_success']) / support) < 1e-12,
                    'Precision/count mismatch')
        if row['method'] in {'full_path', 'pure_prefix', 'exact_delete'} and support > 0:
            require(precision == 1, 'Exact rule disagreement')
    require(len(paired) == 4800, 'Expected 4,800 paired queries')
    require(all(len(v) == 7 and set(v) == methods for v in paired.values()),
            'Missing or duplicate paired methods')
    curves = rows(main / 'paired_curves.csv')
    curve_keys = [tuple(r[k] for k in key) + (r['k'],) for r in curves]
    require(len(curves) == len(set(curve_keys)) == 38400, 'Incomplete prefix curves')
    require(set(curve_keys) == {k + (str(i),) for k in paired for i in range(1, 9)},
            'Prefix curves do not match the main queries')
    anchors = rows(main / 'anchors.csv')
    anchor_keys = {tuple(r[k] for k in key) for r in anchors}
    require(len(anchors) == len(anchor_keys) == 105, 'Incomplete Anchors queries')
    require(all(r['status'] == 'ok' for r in anchors), 'Unaccounted Anchors failure')
    require(anchor_keys <= paired.keys(), 'Unmatched Anchors query')

    cross = exp / 'crossmodel-robustness/results'
    require(len(rows(cross / 'fits.csv')) == 205, 'Incomplete cross-model fits')
    complete = exp / 'exact-projection-audit/results'
    projections = rows(complete / 'queries.csv')
    require(len(rows(complete / 'fits.csv')) == 205 and len(projections) == 3500,
            'Incomplete complete-projection audit')
    require(all(number(r['contains_query']) == 1 for r in projections),
            'Complete projection misses query')
    require(all(number(r['precision']) == 1 for r in projections
                if int(r['ref_count']) > 0), 'Complete projection disagreement')
    require(all(not math.isfinite(number(r['precision'])) for r in projections
                if int(r['ref_count']) == 0), 'Missing complete-cell precision lost')
    controls = rows(exp / 'matched-subset-audit/results/queries.csv')
    require(len(controls) == 210, 'Incomplete minimum-rule controls')
    minimum = defaultdict(list)
    for row in controls:
        minimum[tuple(row[k] for k in key)].append(row['method'])
    require(set(minimum) == anchor_keys, 'Minimum controls use different queries')
    require(all(len(v) == 2 and set(v) == {'minimum_exact_subset',
                'minimum_empirical_subset'} for v in minimum.values()),
            'Missing minimum-rule method')

    gpu = exp / 'reviewer-followup-gpu/results'
    for relative, expected in load(gpu / 'input_hashes.json').items():
        require(digest(bounded_path(root, relative)) == expected,
                f'GPU input checksum mismatch: {relative}')
    validation = load(gpu / 'validation.json')
    expected_counts = {'reconstructed_tree_fits': 280, 'tree_queries': 4800,
                       'stored_curve_counts_reproduced': 38400,
                       'gpu_tree_prediction_cpu_disagreements': 0,
                       'same_network_fits': 65, 'same_network_queries': 1100,
                       'neural_support_inclusion_violations': 0,
                       'matched_queries': 105, 'matched_both_supported': 104}
    require(all(validation.get(k) == v for k, v in expected_counts.items()),
            'GPU validation counts differ')
    require(validation.get('original_inputs_unchanged') is True, 'GPU changed inputs')
    require(validation['sibling_identity_max_error'] < 1e-12, 'Sibling identity error')
    errors = [value for k, value in validation.items()
              if k.endswith('bootstrap_max_difference')]
    require(bool(errors) and max(errors) < 3e-15, 'GPU/CPU bootstrap mismatch')
    matched = rows(gpu / 'matched_pairs.csv')
    require(len(matched) == 105 and sum(number(r['both_supported']) for r in matched)
            == 104, 'Matched common-support count mismatch')
    exclusions = rows(gpu / 'matched_leave_one_out.csv')
    require(len({r['excluded_task'] for r in exclusions}) == 7,
            'Missing leave-one-task-out result')

    return {'status': 'passed', 'reference_files_verified': len(checksums),
            'disjoint_splits': len(splits), 'tree_fits': len(fits),
            'paired_tree_queries': len(paired), 'method_query_rows': len(tree),
            'anchors_queries': len(anchors), 'complete_projections': len(projections),
            'gpu_input_hashes_match': True, 'recorded_gpu_cpu_checks_pass': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Artifact root (defaults to this package)')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Artifact verification failed: {error}\n')
