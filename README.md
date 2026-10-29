# POSEIDON

[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13321--024--00810--7-blue)](https://doi.org/10.1186/s13321-024-00810-7)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![R](https://img.shields.io/badge/R-3.6.1-blue)

**POSEIDON — Peptidic Objects SEquence-based Interaction with cellular DOmaiNs**

POSEIDON is an open database and machine-learning framework for quantitative prediction of cell-penetrating peptide (CPP) uptake across cell lines.

The published resource contains **2,371 experimentally curated entries**. The final modelling benchmark contains **1,263 samples** represented by **1,330 features** integrating experimental conditions, cargo, peptide sequence, physicochemical descriptors and cell-line genomic information.

## Key features

- curated quantitative CPP uptake database;
- peptide physicochemical and sequence descriptors;
- experimental-condition and cargo features;
- cell-line genomic features;
- classical machine-learning and deep-learning regression models;
- XGBoost and deep-learning models used for quantitative uptake prediction;
- data-processing, hyperparameter-tuning and final-training scripts.

## Installation

POSEIDON is supported here with Python 3.10 and R 3.6.1. The core Python requirements are pinned to the validated POSEIDON runtime used for the published deployment.

### Python

```bash
git clone https://github.com/MoreiraLAB/poseidon.git
cd poseidon

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For hyperparameter tuning and model-training utilities:

```bash
pip install -r requirements-training.txt
```

### R

Install R 3.6.1 and the packages used to calculate peptide descriptors:

```r
install.packages(c("Rcpp", "Peptides", "reshape2"),
                 repos = "https://cran.rstudio.com")
```

If R is installed outside the default system location, set `R_HOME` before running the feature-extraction scripts.

## Repository check

A lightweight integrity check is included:

```bash
python scripts/repository_self_check.py
```

It verifies the published database size, the 1,330-feature schema, the cell-line table and the identity of the published scaler.

## Data

Important files include:

| File | Description |
|---|---|
| `data/CPP_dataset.csv` | Published POSEIDON CPP database |
| `data/GDSC_features.csv` | Processed genomic features used by POSEIDON |
| `support/features_header.txt` | Final 1,330-feature schema |
| `support/scaler.pkl` | StandardScaler used by the published model |
| `support/Peptides_deployment.R` | Peptide physicochemical descriptor definitions |
| `support/cell_line_name_encoder.csv` | Experimental-to-GDSC cell-line mapping |
| `support/cell_line_tissue_table.csv` | Cell-line/tissue mapping |

The raw POSEIDON database contains 2,371 entries. Some training-pipeline stages use manually curated intermediate tables in addition to the files distributed here.

## Scientific workflow

The original scripts follow the published workflow:

1. `poseidon_process.py` — process the CPP uptake dataset.
2. `poseidon_main.py` — calculate peptide descriptors.
3. `poseidon_merge_curated.py` — integrate curated sequence and cargo information.
4. `poseidon_peptides_features.py` — assemble peptide and genomic features.
5. `poseidon_process_gdsc.py` — process GDSC genomic data.
6. `poseidon_feature_processing.py` — log-transform the target, remove zero-variance features, split the dataset and scale features.
7. `poseidon_ML_HPT.py` and `poseidon_DL_HPT.py` — hyperparameter optimisation.
8. `poseidon_ML_final.py` and `poseidon_DL_final.py` — train and evaluate the final models.

The central paths and feature definitions are in `poseidon_variables.py`.

## Feature representation

The final feature vector contains **1,330 features**:

| Feature block | Features |
|---|---:|
| Experimental conditions | 8 |
| Cargo | 62 |
| Sequence / modifications | 18 |
| Peptide physicochemical descriptors | 31 |
| Positional sequence encoding | 778 |
| Genomic features | 415 |
| Anomalous-position features | 18 |
| **Total** | **1,330** |

## Model performance

On the independent test set reported in the publication, POSEIDON achieved:

- Pearson correlation: **0.87**
- Spearman correlation: **0.88**
- R²: **0.76**

## Versioning

The published 2024 scientific baseline is anchored to repository commit:

```text
a5c7eb7af4dda6efc5e6aea95d30701eb5c10862
```

Recommended release tags:

- `published-2024` — immutable snapshot of the published scientific baseline;
- `v1.0.0` — first curated repository release;
- future scientific model changes should increment the semantic version.

## Citation

If you use POSEIDON, please cite:

> Preto AJ, Caniceiro AB, Duarte F, Fernandes H, Ferreira L, Mourão J, Moreira IS. **POSEIDON: Peptidic Objects SEquence-based Interaction with cellular DOmaiNs: a new database and predictor.** *Journal of Cheminformatics* **16**, 18 (2024). https://doi.org/10.1186/s13321-024-00810-7

Citation metadata are also available in `CITATION.cff`.

## License

POSEIDON is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
