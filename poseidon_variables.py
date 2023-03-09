#!/usr/bin/env python

"""
Generate variables for poseidon
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "POSEIDON"

SYSTEM_SEP = "/"
CSV_SEP = ","
PARAGRAPH_SEP = "\n"
DEFAULT_LOCATION = "C:/Users/marti/Desktop/peptides/apocalypse"
DATA_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "data"
SUPPORT_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "support"
RESULTS_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "results"
MODELS_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "models"

RAW_DATA_FILE = DATA_FOLDER + SYSTEM_SEP + "CPP_SN3_dataset.csv"
PROCESSED_DATA_FILE = DATA_FOLDER + SYSTEM_SEP + "processed_dataset.csv"
LOG10_PROCESSED_DATA_FILE = DATA_FOLDER + SYSTEM_SEP + "log_10_processed_data.csv"
DATASET_FASTA_FILE = DATA_FOLDER + SYSTEM_SEP + "peptides.fasta"
GDSC_GENOMIC_DATA_FILE = DATA_FOLDER + SYSTEM_SEP + "PANCANCER_Genetic_features_Tue_Jan_24_17_09_21_2023.csv"
GDSC_PROCESSED_DATA_FILE = DATA_FOLDER + SYSTEM_SEP + "GDSC_features.csv"
FEATURES_DATASET_FILE = DATA_FOLDER + SYSTEM_SEP + "data_with_features.csv"

TRAIN_FEATURES_FILE = DATA_FOLDER + SYSTEM_SEP + "train_features.csv"
TEST_FEATURES_FILE = DATA_FOLDER + SYSTEM_SEP + "test_features.csv"
TRAIN_TARGET_FILE = DATA_FOLDER + SYSTEM_SEP + "train_target.csv"
TEST_TARGET_FILE = DATA_FOLDER + SYSTEM_SEP + "test_taget.csv"
TRAIN_IDS_FILE = DATA_FOLDER + SYSTEM_SEP + "train_IDs.csv"
TEST_IDS_FILE = DATA_FOLDER + SYSTEM_SEP + "test_IDs.csv"

EXCLUDE_METHODS_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "exclude_methods.txt"
EXCLUDE_UNITS_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "exclude_units.txt"
NORM_CARGOS_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "cargo_list.txt"
CARGO_MATCHER_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "cargo_matcher.csv"
SEQUENCE_ANOMALIES_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "sequence_anomalies.csv"
CELL_LINE_NAME_MATCHER = SUPPORT_FOLDER + SYSTEM_SEP + "cell_line_name_encoder.csv"
VARIANCE_LOG_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "variance_assess.csv"

R_LOCATION = "/Users/marti/anaconda3/envs/Peptides/Lib/R"
R_FEATURES_TEMPLATE = SUPPORT_FOLDER + SYSTEM_SEP + "Peptides_deployment.R"
R_GROUPS = ["tiny","small","aliphatic","aromatic","nonpolar","polar","charged","basic","acidic"]
RANDOM_SEED = 42
AA_LIST = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
AMINO_ACID_DICTIONARY = {"G": 0 , "A": 0, "L": 0 , "M": 0, "F": 0, \
							"W": 0, "K": 0, "Q": 0, "E": 0 , "S": 0, \
							"P": 0, "V": 0, "I": 0, "C": 0, "Y": 0, \
							"H": 0, "R": 0, "N": 0, "D": 0, "T": 0}

DATA_ID_COLUMNS = ["peptide_name", "cell_line", "pubmedid", "peptide_sequence", "standard_sequence", "raw_sequence"]
DATA_TARGET = ["target"]
FEATURES_DICTIONARY = {"experimental": ['conc_uM', 'time_min', 'time_missing', 'temperature_4', 'temperature_23', 'temperature_25', 'temperature_37', 'temperature_42'],\
								"cargo": ["cargo"],
								"sequence": ["sequence"], "anomalous": ["anomalous"], \
								"peptideR": ['boman_index', 'weight', 'AAC_number_tiny', 'AAC_freq_tiny', 'AAC_number_small', 'AAC_freq_small', \
									'AAC_number_aliphatic', 'AAC_freq_aliphatic', 'AAC_number_aromatic', 'AAC_freq_aromatic', 'AAC_number_nonpolar', \
									'AAC_freq_nonpolar', 'AAC_number_polar', 'AAC_freq_polar', 'AAC_number_charged', 'AAC_freq_charged', 'AAC_number_basic', \
									'AAC_freq_basic', 'AAC_number_acidic', 'AAC_freq_acidic', 'charge', 'pI', 'aindex', 'instability', 'hydrophobicity', \
									'hydrophobicity_moment', 'membrane_position_H', 'membrane_position_uH', 'membrane_position_globular', 'membrane_position_surface', \
									'membrane_position_transmembrane'],
								"sequence_encoding": [str(x) for x in list(range(1, 100 + 1))], \
								"genomics": ["gene", "true_match"], \
								"anomalous": ["anomalous"]
								}
PLACEHOLDER_FEATURES_DICTIONARY = {"experimental": "", "cargo": "", "sequence": "", "peptideR": "", "sequence_encoding": "", "genomics": ""}
FEATURES_SIZES_DICTIONARY = {"experimental": 8, "cargo": 62, "sequence": 18, "peptideR": 31, "sequence_encoding": 778, "genomics": 415, "anomalous": 18}
FEATURES_BLOCKS = ["experimental", "cargo", "sequence", "peptideR", "sequence_encoding", "genomics", "anomalous"]						