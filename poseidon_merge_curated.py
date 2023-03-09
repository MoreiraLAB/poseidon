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
import poseidon_functions
import sys
import pandas as pd
import numpy as np

"""
Open the curated table
"""
curated_table = pd.read_csv(poseidon_variables.DATA_FOLDER + poseidon_variables.SYSTEM_SEP + "curated_dataset.csv", sep = ";", header = 0)

"""
Load the table with the relations between substituents ids, column name in the curated tables, 
and the amino acid to which they should bbe substituted:
- inseq_id: the id in the sequences
- col_name: the identifier in the column, must be added to "sequence_"
- substitute: the amino acid it should be converted to
"""
helper_table = pd.read_csv(poseidon_variables.SEQUENCE_ANOMALIES_FILE, sep = ",", header = 0) #inseq_id, col_name, substitute
unique_substituents = list(helper_table["inseq_id"])

"""
Aggregate cargo protein columns with same meaning 
"""
cargo_matcher = pd.read_csv(poseidon_variables.CARGO_MATCHER_FILE, sep = poseidon_variables.CSV_SEP, header = 0)
unique_cargos = list(cargo_matcher["unique_name"].unique())

for current_cargo in unique_cargos:
	columns_list = ["cargo_" + current_proxy.lower() for current_proxy in list(cargo_matcher.loc[cargo_matcher["unique_name"] == current_cargo]["column_name"])]
	curated_table["cargo_" + current_cargo] = curated_table[columns_list].sum(axis = 1) #Add all columns into single one
	curated_table["cargo_" + current_cargo] = curated_table["cargo_" + current_cargo].astype(bool).astype(int) #Change all non-zero values into
	curated_table = curated_table.drop(columns_list, axis = 1)

"""
Identify the columns that relate to non-standard features.
Isolate the cyclic column, replicate the sequence to keep an unchanged version.
"""
seq_cols_list = [current_column for current_column in curated_table if current_column.startswith('sequence_')]
seq_cols_list.remove("sequence_cyclic") # Cyclic was manually curated and cannot be found otherwise
curated_table[seq_cols_list] = 0 #Turn sequence countable columns into 0
curated_table["raw_sequence"] = curated_table["peptide_sequence"]

length_longest_anomalous_peptide = 0
for index, current_row in curated_table.iterrows():
	if current_row["standard_sequence"] == 1:
		continue

	current_sequence = current_row["peptide_sequence"]
	split_sequence = current_sequence.split("-")
	if split_sequence[-1] == "NH2":
		split_sequence = split_sequence[0:-1]

	standardized_sequence = ""
	for current_block in split_sequence:

		helper_table_row = helper_table.loc[helper_table["inseq_id"] == current_block]
		if current_block not in unique_substituents:
			standardized_sequence += current_block

		elif current_block in unique_substituents:

			standardized_sequence += helper_table_row["substitute"].values[0]
			curated_table.at[index, "sequence_" + helper_table_row["col_name"].values[0]] += 1
	
	curated_table.at[index, "sequence_enantiomer"] = len([current_char for current_char in standardized_sequence if current_char.islower()]) #identify L-aminoacids
	curated_table.at[index, "peptide_sequence"] = standardized_sequence.upper().replace(" ","").replace("X","")
	if (len(standardized_sequence.upper()) > length_longest_anomalous_peptide) and (len(split_sequence) > 1):
		length_longest_anomalous_peptide = len(standardized_sequence.upper())

"""
Feature block to add anomalous position encoding
"""

anomalous_header = ["anomalous_position_" + str(current_anomaly) for current_anomaly in range(1, length_longest_anomalous_peptide + 1)]

anomalies_dataframe = pd.DataFrame(np.zeros(shape = (curated_table.shape[0], len(anomalous_header))), columns = anomalous_header).astype(int)
position_encoded_dataframe = pd.concat([curated_table, anomalies_dataframe], axis = 1)

for index, current_row in position_encoded_dataframe.iterrows():

	raw_sequence = current_row["raw_sequence"]
	split_sequence = raw_sequence.split("-")
	if split_sequence[-1] == "NH2":
		split_sequence = split_sequence[0:-1]
	
	if current_row["standard_sequence"] == 1 or len(split_sequence) <= 1:
		continue

	standardized_sequence = ""
	for current_block in split_sequence:
		if current_block == "miniPEG_linker":
			print(split_sequence)
			print(current_block in unique_substituents)
		helper_table_row = helper_table.loc[helper_table["inseq_id"] == current_block]
		if current_block not in unique_substituents:
			
			standardized_sequence += current_block

		elif current_block in unique_substituents:
			standardized_sequence += helper_table_row["substitute"].values[0]
			current_position = len(standardized_sequence)
			position_encoded_dataframe.at[index, "anomalous_position_" + str(current_position)] = 1

clean_dataframe = position_encoded_dataframe.drop(["pubmedid", "raw_sequence"], axis = 1)
position_encoded_dataframe.to_csv(poseidon_variables.PROCESSED_DATA_FILE, index = False)