#!/usr/bin/env python

"""
Process GDSC to yield proper features 
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "POSEIDON"

import poseidon_variables
import pandas as pd
import sys

raw_gdsc = pd.read_csv(poseidon_variables.GDSC_GENOMIC_DATA_FILE, \
	sep = poseidon_variables.CSV_SEP, header = 0, usecols = ['Cell Line Name', 'Genetic Feature', 'IS Mutated', 'GDSC Desc2'])

cell_line_table = raw_gdsc.drop(["Genetic Feature", "IS Mutated"], axis = 1).drop_duplicates().reset_index(drop = True)
raw_gdsc = raw_gdsc.drop(["GDSC Desc2"], axis = 1)

"""
Open file that matches table IDs with GDSC IDs,
Used to subset GDSC in order to get only the rows to be used
"""
filter_file = pd.read_csv(poseidon_variables.CELL_LINE_NAME_MATCHER, sep = ";", header = 0)
usable_cell_lines = list(filter_file["matching_name"].unique())
subset_gdsc = raw_gdsc.loc[raw_gdsc["Cell Line Name"].isin(usable_cell_lines)]

"""
Identify the genes to be used as features, that means
aggregating all the unique genes for all cell lines
"""
raw_genes = []
for current_cell_line in usable_cell_lines:
	if current_cell_line == "0":
		continue
	raw_genes += list(subset_gdsc.loc[subset_gdsc["Cell Line Name"] == current_cell_line]["Genetic Feature"])
unique_genes = list(set(raw_genes))

"""
Create output GDSC features table using the mutation value column
"""
import numpy as np
output_array = np.zeros(shape = (len(usable_cell_lines), len(unique_genes)))

for index, usable_cell_line in enumerate(usable_cell_lines):

	if usable_cell_line == "0":
		output_array[index] = [0]*735
	subset_features = subset_gdsc.loc[subset_gdsc["Cell Line Name"] == usable_cell_line]
	features_vector = []
	print("Current cell line ", usable_cell_line, ":", index + 1, "/", len(usable_cell_lines))
	for current_gene in unique_genes:
		try:
			features_vector.append(subset_features.loc[subset_features["Genetic Feature"] == current_gene]["IS Mutated"].values[0])
		except:
			features_vector.append(0)
	output_array[index] = features_vector

#43 cell lines, 736 unique genes
output_dataframe = pd.DataFrame(output_array)
output_dataframe.columns = ["gene_" + current_column for current_column in unique_genes]
output_dataframe["cell_line"] = usable_cell_lines
output_dataframe.to_csv(poseidon_variables.GDSC_PROCESSED_DATA_FILE, index = False)

"""
Write a table with the matching tissues, for input selection and data visualization
"""
unique_cell_lines = list(output_dataframe["cell_line"].unique())
usable_index_list = []
for index, tissue_indexed_cell_line in cell_line_table.iterrows():
	current_cell_line = tissue_indexed_cell_line["Cell Line Name"]
	if current_cell_line in usable_cell_lines:
		usable_index_list.append(index)

tissue_indexed_dataframe = cell_line_table.iloc[usable_index_list, :]
tissue_indexed_dataframe.to_csv(poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "cell_line_tissue_table.csv", index = False)