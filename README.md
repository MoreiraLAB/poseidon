# POSEIDON

*Peptidic Objects SEquence-based Interaction with cellular DOmaiNs*

POSEIDON is a curated cell-penetrating peptide (CPP) database and a machine-learning framework for quantitative prediction of peptide uptake across cell lines.

The published study reports a database of **2,371 entries**, a final modelling benchmark of **1,263 samples**, and a **1,330-feature** representation combining experimental conditions, cargo, sequence descriptors, physicochemical properties, positional sequence encoding, genomic features, and anomalous-position features.

## Scientific reference

Preto AJ, Caniceiro AB, Duarte F, Fernandes H, Ferreira L, Mourão J, Moreira IS.  
**POSEIDON: Peptidic Objects SEquence-based Interaction with cellular DOmaiNs: a new database and predictor.**  
*Journal of Cheminformatics* **16**, 18 (2024).  
DOI: **10.1186/s13321-024-00810-7**

Citation metadata are also provided in `CITATION.cff`.

## Repository scope

This repository contains the **scientific code, data, feature definitions, and supporting artefacts** associated with the published POSEIDON workflow.

It intentionally does **not** contain the production web application, HTML templates, CSS/JavaScript assets, deployment configuration, user uploads, e-mail configuration, or server logs.

## Reproducibility status

The historical scientific pipeline is preserved for provenance, but the repository should not yet be interpreted as a fully portable `clone -> retrain -> reproduce` package.

Known limitations include:

- historical machine-specific paths in `poseidon_variables.py`;
- manually curated intermediate inputs that are not all versioned in the repository;
- final deployed model binaries are not stored in Git history;
- the historical dependency list is documented below, but a fully validated lockfile/conda environment is not yet available;
- some historical comments state “1360 features”, while the published and checked feature schema contains **1,330**.

Detailed notes are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### Repository self-check

A lightweight, dependency-free integrity check validates the published repository contract:

```bash
python scripts/repository_self_check.py
```

It checks, among other things:

- 2,371 rows in `data/CPP_dataset.csv`;
- 1,330 unique feature names in `support/features_header.txt`;
- 42 rows in the deployment cell-line/tissue table;
- the SHA256 identity of the published `support/scaler.pkl`;
- absence of web/deployment code from the scientific repository.

## Historical software stack

The original workflow was developed with the following software versions:

- Python 3.10.8
  - NumPy 1.24.1
  - pandas 1.5.2
  - scikit-learn 1.2.0
  - XGBoost 1.7.3
  - TensorFlow 2.11.0
  - Ray Tune 2.2.0
  - PyArrow 11.0.0
  - tune-sklearn
  - scikit-optimize
  - rpy2
- R 3.6.1
  - Rcpp
  - Peptides
  - reshape2

These versions document the historical environment; they are **not yet claimed as a fully validated portable lockfile**.

## Scientific workflow

The historical scripts are organised as follows:

1. `poseidon_variables.py` — central paths and feature definitions.
2. `poseidon_functions.py` — shared helper functions.
3. `poseidon_process.py` — processing of the curated CPP uptake dataset.
4. `poseidon_main.py` — peptide feature extraction.
5. `poseidon_merge_curated.py` — merge of manually curated sequence/cargo information and sequence anomaly handling.
6. `poseidon_peptides_features.py` — peptide, positional, and genomic feature assembly.
7. `poseidon_process_gdsc.py` — processing of GDSC genomic features.
8. `poseidon_feature_processing.py` — zero-variance filtering, log10 target transformation, train/test split, and scaling.
9. `poseidon_ML_HPT.py` / `poseidon_DL_HPT.py` — hyperparameter tuning.
10. `poseidon_ML_final.py` / `poseidon_DL_final.py` — final model training and evaluation.

Some steps depend on manual curation files and therefore cannot yet be executed end-to-end from a fresh clone without additional provenance work.

## Data and artefacts

Important versioned resources include:

- `data/CPP_dataset.csv` — published CPP database;
- `data/GDSC_features.csv` — processed genomic feature matrix;
- `support/features_header.txt` — final 1,330-feature schema;
- `support/scaler.pkl` — scaler used by the published deployment;
- `support/Peptides_deployment.R` — peptide descriptor definitions;
- `support/cell_line_name_encoder.csv` — mapping between experimental and genomic cell-line names;
- `support/cell_line_tissue_table.csv` — deployment cell-line/tissue mapping.

## Development policy

The historical scripts are part of the provenance of the published model. Refactors that are intended to preserve the published model should be validated against immutable artefacts and regression tests before merge.

Changes that intentionally alter scientific outputs, training data, feature ordering, or fitted models should be developed and versioned as a **new scientific model revision**, rather than silently replacing the published baseline.

## License

GNU General Public License v3.0. See `LICENSE`.
