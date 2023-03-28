# POSEIDON
*Peptidic Objects SEquence-based Interaction with cellular DOmaiNs*

## Description

Cell-penetrating peptides (CPPs) are short chains of amino acids that have shown remarkable potential to cross the cell membrane and deliver coupled therapeutic cargo into cells. Designing and testing different CPPs to target specific cells or tissues is crucial to ensure high delivery efficiency and reduced toxicity. However, in vivo/in vitro testing of various CPPs can be both time-consuming and costly, which has led to interest in computational methodologies, such as Machine Learning (ML) approaches, as faster and cheaper methods for CPP design and uptake prediction. However, most ML models developed to date focus on classification rather than regression techniques, because of the lack of informative quantitative uptake values. To address these challenges, we developed POSEIDON, an open-access and up-to-date curated database that provides experimental quantitative uptake values for over 2,300 entries and the physicochemical properties of 1,315 peptides. By leveraging this database along with cell line genomic features, we processed a dataset of over 1,200 entries to develop an ML regression CPP uptake predictor. Our results demonstrated that POSEIDON accurately predicted peptide cell line uptake, achieving a Pearson correlation of 0.87, Spearman correlation of 0.88, and r2 score of 0.76, on an independent test set. With its comprehensive data and powerful predictive capabilities, POSEIDON represents a major step forward in CPP research and development that can be easily accessed at https://moreiralab.com/resources/poseidon/.


## Installation

POSEIDON requires both Python and R packages, and it is advisable to use an environment to install the following software:

- Python 3.10.8
	- `pip install numpy==1.24.1 pandas==1.5.2 scikit-learn==1.2.0 xgboost==1.7.3 tensorflow==2.11.0`
	- `pip install ray[tune]==2.2.0`
	- `pip install pyarrow==11.0.0`
	- `pip install tune-sklearn ray[tune]`
	- `pip install scikit-optimize rpy2`
- R 3.6.1
	- `Rscript -e "install.packages(c('Rcpp','Peptides', 'reshape2'), repos='https://cran.rstudio.com')"`

## Usage

1. \
	A - `poseidon_variables.py` - lists most of the important POSEIDON features\
	B - `poseidon_functions.py` - script with functions useful throughout the whole pipeline
2. `poseidon_process.py` - script to generate the file for feature extraction
3. `poseidon_main.py` - feature extraction from peptidic sequence
4. `poseidon_merge_curated.py` - merge the previously curated datasets with the standard sequences and identify position anomalies
5. `poseidon_peptides_features.py` - aadd features to peptide sequences. Please note that, at this point, the samples were manually curated; as such, it is not possible to run the scripts without manually curated files.
6. `poseidon_process_gdsc.py` - process GDSC data to yield usable features
7. `poseidon_feature_processing.py` - process the features to remove 0 variance, as well as the target to log10, and outliers
8. \
	A - `poseidon_ML_HPT.py` - HyperParameterTuning for Machine Learning parameters, involving non-Deep Learning models\
	B - `poseidon_DL_HPT.py` - HyperParameterTuning for Deep Learning parameters, for forked and regular Deep Learning models
9. \
	A - `poseidon_ML_final.py`  - Run the final prediction models, saving them, by fetching the best parameters from the previously ran Ray tune \
	B - `poseidon_DL_final.py` - Run the final prediction models, saving them, by fetching the best parameters from the previously ran Ray tune

## Webserver

The [Poseidon webserver](http://www.moreiralab.com/resources/poseidon/) is a powerful tool that allows dynamic visualization of the data (**Analysis** section), querying of both the raw and processed database (**Database**), and submission of an input peptide with multiple optional specifications  (**Home** section). 


## Please cite

*POSEIDON: Peptidic Objects SEquence-based Interaction with cellular DOmaiNs*, A. J. Preto, Ana B. Caniceiro, Francisco Duarte, Hugo Fernandes, Lino Ferreira, Joana Mourão and Irina S. Moreira, (*submitted*), 2023
