# Faithful Explanations

Reproducibility artifact for **A Unified Logical Framework for Faithful
Explanations: Translation and Projection across Tree, Linear, and Neural Models**,
Fengyu Wei and Di Zhang, manuscript prepared for *Knowledge-Based Systems*.

This package contains the current independent-pool experiments and the post-review
GPU audits, synchronized with the local manuscript on **2026-10-08**. Reference
results are included so they can be inspected without training or downloading data.

## Layout

```text
code/                         current experiments, analysis and verification
data/                        dataset identity, split indices and display names
results/
  heldout-utility/            independent tree-rule benchmark and Anchors
  crossmodel-robustness/      independent tree, linear and ReLU comparisons
  exact-projection-audit/     complete projection validation
  matched-subset-audit/       minimum path-subset controls
  reviewer-followup-gpu/      mass budget, localization and representation audits
requirements.txt             shared CPU dependencies; optional installs below
reference-sha256.json         reference-data and figure checksums
```

Each experiment contains its dated protocol and final `results/` tables. Logs,
Python environments, downloaded datasets, private editing notes, LaTeX caches and
manuscript-rewriting scripts are excluded. The package contains code and reference
results; plotting commands below regenerate figures from those results.

## Code and results

| Directory | Role |
| --- | --- |
| `code/` | Code that generates and analyzes the current experimental results |
| `results/` | Current protocols and final results used by the KBS manuscript |

The current experiments use separate fit, calibration, reference and query pools,
fit-only preprocessing, compression controls and independent audits.

To reproduce the current paper, use `code/` and `results/`. The current scripts
read their required results from `results/`. The GPU follow-up depends on the
four current CPU result families.

## Quick verification: no extra dependencies or GPU

From the repository root:

```sh
python code/verify_artifact.py
```

This standard-library-only command checks reference checksums, four-way split
disjointness, method/query pairing, exact-region agreement, the GPU input hashes
and recorded numerical cross-checks. It reads the archived evidence without
changing it. It does not retrain models or prove formal statements.
`reference-sha256.json` stores file fingerprints for checking that the supplied
reference data and statistical outputs are intact.

## Install the recorded CPU environment

The reference CPU runs used Python **3.14.3**. Create a virtual environment and
install the pinned versions:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -r requirements.txt
```

When rerunning the Anchors comparison, additionally install:

```sh
python -m pip install --no-deps anchor-exp==0.0.2.0 lime==0.2.0.1
```

This matches the tabular-only Anchors setup used in the recorded run. Its
NumPy/SciPy/scikit-learn dependencies come from `requirements.txt`; spaCy
and image-LIME dependencies are unused by these scripts. This installation is not
intended to support every optional feature of the upstream Anchors/LIME packages.

## Reproduce the CPU experiments

Run from the repository root, in a **separate checkout or copy**: the experiment
commands overwrite that copy's result tables, and timing fields vary by machine.

```sh
python code/validate_rules.py
python code/utility_experiment.py
python code/utility_experiment.py --mode anchor
python code/crossmodel_replication.py
python code/exact_projection_audit.py
python code/matched_subset_audit.py
python code/analyze_results.py
python code/analyze_results.py --crossmodel
```

`matched_subset_audit.py` requires the Anchors query identifiers;
`analyze_results.py` requires completed main and Anchors results. Numerical
comparisons should use scientific metrics and the recorded environments, rather
than byte equality of timing-bearing regenerated CSVs. The checksum manifest
identifies the supplied reference snapshot; it is not updated automatically.

To recompute statistics from existing CSVs without fitting models, run only the
two `analyze_results.py` commands in a copy. Their figures are written into the
corresponding study's `results/` directory.

## Reproduce the GPU follow-up

Use a separate environment on an NVIDIA CUDA-compatible GPU:

```sh
python -m venv .venv-gpu
```

Activate this environment as above, then install dependencies **sequentially**:

```sh
python -m pip install -r requirements.txt
python -m pip install numpy==2.5.3 cupy-cuda12x==14.0.1 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
python code/gpu_followup_audit.py
```

The optional GPU installation replaces NumPy 2.5.2 with **2.5.3**, matching the recorded GPU
run. The reference hardware was an RTX 4060 Laptop GPU with CuPy 14.0.1. Actual
device, driver/runtime versions, precision and random seed are recorded in
[runtime.json](results/reviewer-followup-gpu/results/runtime.json).

The GPU command requires the existing results from all four CPU experiment
families; these are already included. CART is reconstructed on CPU with the same
scikit-learn algorithm. CUDA float64 performs region counting and 5,000-resample
hierarchical bootstrap aggregation, while neural controls reuse fixed-fit records.
The command requires a real GPU and rejects CPU fallback. Compiler caches stay in
the ignored `.cache/` directory within this package.

Generate the GPU audit figure from cached tables, with CPU dependencies only:

```sh
python code/plot_followup_audit.py
```

It writes `compression-budget-audit.pdf` and `.png` under
`results/reviewer-followup-gpu/results/`. No script needs an
external manuscript directory or a machine-specific Python path.

## Evidence and manuscript mapping

| Experiment | Main evidence | Manuscript topic |
| --- | --- | --- |
| [heldout-utility](results/heldout-utility/protocol.md) | `queries.csv`, `fits.csv`, `anchors.csv`, `summary_ci.csv`, `paired_differences.csv`, `cases.json` | Independent shortening utility, Anchors, target/count sensitivity and audit cases |
| [crossmodel-robustness](results/crossmodel-robustness/protocol.md) | `queries.csv`, `fits.csv`, `summary_ci.csv`, `convergence.csv`, `converged_summary.csv` | Independent cross-model precision, coverage and evaluability |
| [exact-projection-audit](results/exact-projection-audit/protocol.md) | `queries.csv`, `summary.csv`, `diagnostics.json` | Complete paths, linear half-spaces and frozen ReLU cells |
| [matched-subset-audit](results/matched-subset-audit/protocol.md) | `queries.csv`, `summary.csv` | Minimum exact and empirical path-literal subsets |
| [reviewer-followup-gpu](results/reviewer-followup-gpu/protocol.md) | `prefixes.csv`, `siblings.csv`, `localization.csv`, `budget_controls_ci.csv`, `neural_regions_ci.csv`, `matched_reliability_ci.csv`, `matched_leave_one_out.csv` | Expansion budget, independent sibling localization, region representation and matched reliability uncertainty |

## Data and final protocol

The current benchmark has **14 binary tasks**: iris, wine, breast_cancer, digits-35,
synthetic_d8, synthetic_d20, synthetic_d50, synthetic_lowsep, banknote, blood,
diabetes, ionosphere, sonar and internet-ads. The six OpenML identifiers are
declared in `code/utility_experiment.py`: 1462, 1464, 37, 59, 40 and 40978.
`digits-35` retains digits 3 and 5; other multiclass datasets retain their two
most frequent classes. Raw datasets are loaded from scikit-learn's cache or
downloaded on first use. Download failure raises an error instead of substituting
or silently skipping a task. Raw data are not redistributed in this repository.

For each task, the tree benchmark uses ten seeds and depths 4/8: **280 fits,
4,800 queries and 33,600 method/query records**. Draw the 20% calibration pool
uniformly before inspecting labels, then stratify the remaining pools to obtain
50% fitting, 25% reference and 5% query fractions. Randomly cap queries at 20.
Median imputation and model-specific scaling are fitted only on the fitting pool.
The 140 split records are in `data/splits.json`; `data/manifest.json` records
dataset hashes, sizes and binary label mappings.

Anchors uses seven preselected tasks, three seeds and five randomly selected
queries per task/seed, totaling **105 queries**, with an eight-predicate budget.
Its native perturbation precision is reported separately from agreement on the
shared unperturbed reference pool.

Cross-model and complete-projection audits use **205 fits** (70 trees, 70 linear
models and 65 ReLU networks). Neural Internet-Advertisements is excluded a priori.
All convergence warnings remain in the tables. Complete projections are checked on
3,500 queries; the GPU neural controls reuse the same 65 networks and 1,100 queries.

Protocols were amended after inspecting pilot outputs. Read the
[amendment](results/heldout-utility/protocol-amendment.md) alongside the original
protocol: its uniform calibration and randomized query selection supersede the
original stratified-calibration/first-query wording. Superseded pilot tables are
not included in the current reference results. The GPU protocol explicitly records
prior access to results; it is a post-review analysis, not original preregistration.

## Interpretation and provenance

Precision means agreement with the fitted classifier, not ground-truth accuracy.
Zero-support precision is missing, with evaluability reported separately. Exact
deletion is an established sufficient, subset-minimal path-literal comparator;
bounded exhaustive controls are not scalable unrestricted AXp solvers. Calibration
mass-budget acceptance is not a population certificate. Sibling localization is
conditional on eligible candidates and independent disagreement. Full class
predicates are semantic controls, not concise human-readable rules. Matched
minimum-rule reliability differences are heterogeneous and statistically
inconclusive. Coverage and literal count measure computational utility, without
establishing human understanding or deployment value.

During packaging, reference numerical tables were copied without
recomputation. Code changes remove external workspace dependencies and keep output
paths within the package.
The original recorded GPU input hashes are preserved, including their Windows path
separators and original `experiments/` prefix; the verifier maps that prefix to
`results/` when locating files. Runtime and validation records
describe the original runs, rather than a fresh run on the reader's computer.

Current verification covers 101 reference files. Packaging validation on 2026-10-08
used a standalone copy and the recorded local environments. It passed a deliberately
corrupted-file negative control, the 416-subset exact verifier, cached CPU interval
regeneration, experimental plotting and a real GPU follow-up rerun. Regenerated GPU
summary metrics and intervals matched the reference within 1e-12; the supplied
current reference files remained unchanged. This was a packaging check, not a new
experimental design.

## License and citation

The repository code is released under the [MIT License](LICENSE). Upstream
dependencies and datasets retain their respective terms. Cite the manuscript
title and authors above; no acceptance, DOI or new release tag is claimed here.
