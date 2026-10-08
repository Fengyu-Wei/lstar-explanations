# Historical exploratory artifact

This directory preserves the five scripts, ten CSV tables and two PNG figures
previously stored at the repository root, including existing local revisions.
The current KBS manuscript's independent-pool results are under `../experiments/`.

The archived studies use five-fold cross-validation and training-fold rule
precision. StandardScaler is fitted before the CV split in these scripts.
Consequently, these are exploratory finite-pool measurements and must not be
presented as leakage-free independent evaluation. The OpenML dataset previously
called `climate` is Internet-Advertisements (data ID 40978); its name was corrected
without recomputing archived values.

From this directory, with the CPU dependencies installed:

```sh
cd results
python ../code/faithfulness_cv_experiment.py
python ../code/linear_faithfulness_experiment.py
python ../code/mlp_gatepath_experiment.py
cd ..
python code/unified_figures.py
```

The first three commands overwrite historical tables in the current directory.
Use a separate checkout for regeneration. The figure script reads cached tables
and writes only to this directory's `figures/`; it no longer writes to an external
manuscript directory. No archived numerical values were changed during packaging.
