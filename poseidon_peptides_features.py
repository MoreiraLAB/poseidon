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
import poseidon_variables
import sys

os.environ['R_HOME'] = poseidon_variables.R_LOCATION

import numpy as np
import pandas as pd
from poseidon_main import peptide

opened_seq_table = pd.read_csv(poseidon_variables.PROCESSED_DATA_FILE, sep = poseidon_variables.CSV_SEP, header = 0)

opened_gdsc_table = pd.read_csv(poseidon_variables.GDSC_GENOMIC_DATA_FILE, sep = poseidon_variables.CSV_SEP, header = 0)
gdsc_unique_cell_lines = list(opened_gdsc_table["Cell Line Name"].unique())

edited_cell_lines_dictionary = {}

peptide_unique_cell_lines = list(opened_seq_table["cell_line"].unique())

"""
with open("cell_line_name_encoder.csv", "w") as ids_file:
	ids_file.write("raw_name,matching_name,true_match\n")
	for current_peptide in peptide_unique_cell_lines:
		true_match = 0
		if current_peptide.replace(" cells", "") in gdsc_unique_cell_lines:
			true_match = 1
		ids_file.write(current_peptide + "," + current_peptide.replace(" cells","") + "," + str(int(true_match)) + "\n")
"""
import difflib
close_matches = difflib.get_close_matches("MCF7.00",gdsc_unique_cell_lines)

for current_cell_name in peptide_unique_cell_lines:
	edited_current_cell_line = current_cell_name.replace(" cells", "")
	edited_cell_lines_dictionary[current_cell_name] = edited_current_cell_line

missing_cell_lines, success_cell_lines = [], []
for raw_cell in edited_cell_lines_dictionary.keys():
	if edited_cell_lines_dictionary[raw_cell] not in gdsc_unique_cell_lines:
		missing_cell_lines.append(edited_cell_lines_dictionary[raw_cell])
	elif edited_cell_lines_dictionary[raw_cell] in gdsc_unique_cell_lines:
		success_cell_lines.append(edited_cell_lines_dictionary[raw_cell])

unique_sequences = list(opened_seq_table["peptide_sequence"].unique())
longest_sequence = 0
with open(poseidon_variables.DATASET_FASTA_FILE, "w") as output_file:
	for index, current_sequence in enumerate(unique_sequences):
		writeable_row = ">" + current_sequence + poseidon_variables.PARAGRAPH_SEP + current_sequence + poseidon_variables.PARAGRAPH_SEP
		if len(current_sequence) > longest_sequence:
			longest_sequence = len(current_sequence)
		output_file.write(writeable_row)

opened_GDSC_features = pd.read_csv(poseidon_variables.GDSC_PROCESSED_DATA_FILE, \
			sep = poseidon_variables.CSV_SEP, header = 0)
opened_cell_line_id_matcher = pd.read_csv(poseidon_variables.CELL_LINE_NAME_MATCHER, sep = ";", header = 0)

unique_dictionary_features, unique_dictionary_features_cell, true_match_dictionary, features_list, header = {}, {}, {}, [], False
cell_header = False
failed = []
for index, row in opened_seq_table.iterrows():
	print("Currently at:", index + 1, "/", opened_seq_table.shape[0])
	current_peptide_sequence = row["peptide_sequence"].replace("X", "A")
	current_cell_line = row["cell_line"]
	if current_cell_line not in unique_dictionary_features_cell:
		usable_cell_line = opened_cell_line_id_matcher.loc[opened_cell_line_id_matcher["raw_name"] == current_cell_line]
		usable_cell_line_name, cell_line_match = usable_cell_line["matching_name"].values[0], usable_cell_line["true_match"].values[0]
		if cell_header == False:
			cell_header = list(opened_GDSC_features.drop(["cell_line"], axis = 1).columns)
		cell_line_features = list(opened_GDSC_features.loc[opened_GDSC_features["cell_line"] == usable_cell_line_name].drop(["cell_line"], axis = 1).values[0])
		unique_dictionary_features_cell[current_cell_line] = cell_line_features
		true_match_dictionary[current_cell_line] = cell_line_match 
	else:
		cell_line_features = unique_dictionary_features_cell[current_cell_line]
		cell_line_match = true_match_dictionary[current_cell_line]

	if current_peptide_sequence not in unique_dictionary_features:
		features_object = peptide(current_peptide_sequence)
		features_object.fetch_R_peptides()
		peptide_encoding = features_object.representation_features(
									max_length = longest_sequence)
		if header == False:
			header = list(features_object.R_features.keys()) + list(peptide_encoding.columns)

		peptide_encoding_features = peptide_encoding.values[0].tolist()
		current_R_features = list(features_object.R_features.values())
		
		unique_dictionary_features[current_peptide_sequence] = current_R_features + peptide_encoding_features
		current_features = current_R_features + peptide_encoding_features

	else:
		current_features = unique_dictionary_features[current_peptide_sequence]
	full_feature_vector = current_features + cell_line_features + [cell_line_match]
	print(current_peptide_sequence, len(full_feature_vector))
	if len(full_feature_vector) != 2767:
		failed.append([index, current_peptide_sequence])
	features_list.append(full_feature_vector)
print(failed)

full_header = header + cell_header + ["true_match"]
features_dataframe = pd.DataFrame(features_list, columns = full_header)
data_with_features = pd.concat([opened_seq_table, features_dataframe], axis = 1)
data_with_features.to_csv(poseidon_variables.FEATURES_DATASET_FILE, index = False)
