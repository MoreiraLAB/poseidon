# POSEIDON reproducibility notes

This repository is the **scientific code and data repository** associated with POSEIDON. It intentionally does **not** contain the production web application, templates, static assets, deployment configuration, user uploads, e-mail configuration, or server logs.

## Published reference

Preto AJ, Caniceiro AB, Duarte F, Fernandes H, Ferreira L, Mourão J, Moreira IS. **POSEIDON: Peptidic Objects SEquence-based Interaction with cellular DOmaiNs: a new database and predictor.** *Journal of Cheminformatics* 16, 18 (2024). DOI: 10.1186/s13321-024-00810-7.

## Repository contract

The current published scientific artefacts satisfy the following invariant checks:

- raw POSEIDON database: **2,371 rows** in `data/CPP_dataset.csv`;
- final model feature schema: **1,330 unique features** in `support/features_header.txt`;
- feature-block sizes: 8 experimental + 62 cargo + 18 sequence + 31 peptide physicochemical + 778 positional sequence encoding + 415 genomic + 18 anomalous-position features = **1,330**;
- supported deployment cell-line table: **42 rows**;
- `support/scaler.pkl` SHA256:
  `9df8cdd9f1fd51e46d5e7a911fb35d7c72ef19d2f8b1e5275451d763c162c5f2`.

Run:

```bash
python scripts/repository_self_check.py
```

The check uses only the Python standard library.

## Important distinction: published baseline vs future corrections

The historical scripts are provenance for the published workflow. Some implementation details should not be silently changed in-place if doing so could alter training data, feature order, or fitted models.

Examples that require explicit validation before alteration:

- feature ordering derived through `set(...)` in legacy scripts;
- data curation steps that depended on manually curated inputs;
- model-training behavior and historical package versions.

Changes that alter scientific outputs should be developed as a separately versioned scientific revision and should not be presented as a transparent refactor of the published model.

## Known limitations of the current historical pipeline

The repository is not yet a complete "clone and run the full training pipeline" package. In particular:

1. `poseidon_variables.py` contains historical machine-specific paths for both the project root and R.
2. Some intermediate/manual curation inputs expected by the scripts (for example `data/curated_dataset.csv`) are not versioned here.
3. Final deployed model binaries are not currently versioned in this repository.
4. The original README lists the historical software stack, but there is not yet a tested lockfile/conda environment covering the complete workflow.
5. Several comments state "1360 features"; the checked published feature schema contains **1330**.
6. There is currently no end-to-end CI test for retraining.

These limitations are documented rather than silently corrected so that the published provenance remains inspectable.

## Deployment baseline recovered from the production scientific runtime

For independent validation of the future inference package, the deployed scientific artefacts recovered from the production runtime have the following SHA256 values:

- XGBoost model (`xgb.h5`): `f9bfa449c7654f8a238e87bf155a006df8758f790f96624a4716b7bda0137471`
- Deep Learning model (`DL_model.h5`): `649051293d2072956aa3806d79bfe97caad7d6b91b9137a81036f9c0a077e9f7`
- scaler (`scaler.pkl`): `9df8cdd9f1fd51e46d5e7a911fb35d7c72ef19d2f8b1e5275451d763c162c5f2`

Those model binaries are deliberately **not** added to the Git history in this cleanup branch. A future release can publish immutable model assets with checksums.

## Scope of this cleanup branch

This branch is limited to reproducibility metadata, repository checks, documentation, and packaging preparation. It does not add website code and does not change the trained models or the published scientific outputs.
