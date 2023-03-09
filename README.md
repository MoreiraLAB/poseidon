# POSEIDON
*Peptidic Objects SEquence-based Interaction with cellular DOmaiNs*

## Description

Cell-penetrating peptides (CPPs) are a class of peptides usually characterized by short amino acid sequences (4-40 residues), versatility, modifiability, and effectiveness at delivering coupled therapeutic cargo into cells. CPPs sequences must be specifically designed for each use case to improve their internalization and reduce possible adjacent toxicities. In vivo/in vitro testing of different CPPs can be laborious and expensive, which has led to interest in computational approaches as faster and cheaper methods for CPP design and uptake prediction. However, no CPP-based database that provides quantitative and measurable data is currently available. Consequently, most ML models developed to date focus on classification problems rather than regression problems, as there is a lack of informative quantitative values.
To address these problems, we developed POSEIDON, a novel up-to-date open-access curated database and predictor that provides quantitative uptake value data and contains the most recent data from the literature. POSEIDON summarizes the available experimental quantitative data and physicochemical property data for the 2371 observations, furthermore, it makes available a processed dataset that was thoroughly screened both programatically and manually. Each observation provides information on a CPP, such as peptide name, PubMed ID, cargo, cell line, cellular uptake values, units, experimental conditions (concentration, temperature, and incubation time), methodology, and peptide sequence. The final best ML predictors reach a Pearson correlation of 0.87, Spearman correlation of 0.88 and r2 score of 0.76.

## Installation

POSEIDON usage will require both Python and R packages, it is advisable the usable of an environment to install the following software.

- Python 3.10.8
	- pip install numpy==1.24.1 pandas==1.5.2 scikit-learn==1.2.0 xgboost==1.7.3 tensorflow==2.11.0
	- pip install ray[tune]==2.2.0
	- pip install pyarrow==11.0.0
	- pip install tune-sklearn ray[tune]
	- pip install scikit-optimize rpy2
- R 3.6.1
	- Rscript -e "install.packages(c('Rcpp','Peptides', 'reshape2'), repos='https://cran.rstudio.com')"

## Usage

1. 
	A - poseidon_variables.py - lists most of the important POSEIDON features
	B - poseidon_functions.py - script with functions useful throughout the whole pipeline
2. poseidon_process.py - script to generate the file for feature extraction
3. poseidon_main.py - feature extraction from peptidic sequence
4. poseidon_merge_curated.py - merge the previously curated datasets with the standard sequences and identify position anomalies
5. poseidon_peptides_features.py - add features to the peptide sequences
6. poseidon_process_gdsc.py - process GDSC data to yield usable features
7. poseidon_feature_processing.py - process the features to remove 0 variance, as well as the target to log10, and outliers
8. <br>
	A - poseidon_ML_HPT.py - HyperParameterTuning for Machine Learning parameters, involving non-Deep Learning models<br>
	B - poseidon_DL_HPT.py - HyperParameterTuning for Deep Learning parameters, for forked and regular Deep Learning models
9. <br>
	A - poseidon_ML_final.py  - Run the final prediction models, saving them, by fetching the best parameters from the previously ran Ray tune <br>
	B - poseidon_DL_final.py - Run the final prediction models, saving them, by fetching the best parameters from the previously ran Ray tune

## Webserver

The [Poseidon webserver](http://www.moreiralab.com/resources/poseidon/) is a powerful tool that allows dynamic visualization of the data (**Analysis** section), querying of both the raw and processed database (**Database**) as well as submitting an input peptide with multiple optional specifications (**Home** section). 

## Please cite

*POSEIDON: Peptidic Objects SEquence-based Interaction with cellular DOmaiNs*, A. J. Preto, Ana B. Caniceiro, Francisco Duarte, Hugo Fernandes, Lino Ferreira, Joana Mourão and Irina S. Moreira, (*submitted*), 2023
