#!/usr/bin/env python

"""
Generate variables for poseidon
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "POSEIDON"

import os
from pathlib import Path

SYSTEM_SEP = os.sep
CSV_SEP = ","
PARAGRAPH_SEP = "\n"

# Repository root by default; override with POSEIDON_ROOT when needed.
PROJECT_ROOT = Path(
    os.environ.get("POSEIDON_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()

DEFAULT_LOCATION = str(PROJECT_ROOT)
DATA_FOLDER = str(PROJECT_ROOT / "data")
SUPPORT_FOLDER = str(PROJECT_ROOT / "support")
RESULTS_FOLDER = str(PROJECT_ROOT / "results")
MODELS_FOLDER = str(PROJECT_ROOT / "models")

# Output directories are local runtime artefacts and are ignored by Git.
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = str(Path(DATA_FOLDER) / "CPP_SN3_dataset.csv")
PROCESSED_DATA_FILE = str(Path(DATA_FOLDER) / "processed_dataset.csv")
LOG10_PROCESSED_DATA_FILE = str(Path(DATA_FOLDER) / "log_10_processed_data.csv")
DATASET_FASTA_FILE = str(Path(DATA_FOLDER) / "peptides.fasta")
GDSC_GENOMIC_DATA_FILE = str(Path(DATA_FOLDER) / "PANCANCER_Genetic_features_Tue_Jan_24_17_09_21_2023.csv")
GDSC_PROCESSED_DATA_FILE = str(Path(DATA_FOLDER) / "GDSC_features.csv")
FEATURES_DATASET_FILE = str(Path(DATA_FOLDER) / "data_with_features.csv")

TRAIN_FEATURES_FILE = str(Path(DATA_FOLDER) / "train_features.csv")
TEST_FEATURES_FILE = str(Path(DATA_FOLDER) / "test_features.csv")
TRAIN_TARGET_FILE = str(Path(DATA_FOLDER) / "train_target.csv")
TEST_TARGET_FILE = str(Path(DATA_FOLDER) / "test_taget.csv")
TRAIN_IDS_FILE = str(Path(DATA_FOLDER) / "train_IDs.csv")
TEST_IDS_FILE = str(Path(DATA_FOLDER) / "test_IDs.csv")

EXCLUDE_METHODS_FILE = str(Path(SUPPORT_FOLDER) / "exclude_methods.txt")
EXCLUDE_UNITS_FILE = str(Path(SUPPORT_FOLDER) / "exclude_units.txt")
NORM_CARGOS_FILE = str(Path(SUPPORT_FOLDER) / "cargo_list.txt")
CARGO_MATCHER_FILE = str(Path(SUPPORT_FOLDER) / "cargo_matcher.csv")
SEQUENCE_ANOMALIES_FILE = str(Path(SUPPORT_FOLDER) / "sequence_anomalies.csv")
CELL_LINE_NAME_MATCHER = str(Path(SUPPORT_FOLDER) / "cell_line_name_encoder.csv")
VARIANCE_LOG_FILE = str(Path(SUPPORT_FOLDER) / "variance_assess.csv")

# Let rpy2 discover R normally. Set R_HOME explicitly only when required.
R_LOCATION = os.environ.get("R_HOME", "")
R_FEATURES_TEMPLATE = str(Path(SUPPORT_FOLDER) / "Peptides_deployment.R")

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
