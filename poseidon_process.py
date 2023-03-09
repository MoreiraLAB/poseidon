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

opened_raw_table = pd.read_csv(poseidon_variables.RAW_DATA_FILE, sep = ";", header = 0, encoding = "cp1252")

"""
Remove columns that are useless from the start
"""
droppable_columns = ["cppsite_id","duvida","local","fluorophore_.if_not_present_in_column_4.",\
					"notes","uptake_efficiency","type_of_uptake"]
opened_reduced_table = opened_raw_table.drop(droppable_columns, axis = 1)
no_missing_values_table = opened_reduced_table.dropna(subset = ['cell_uptake_value','peptide_sequence', 'cell_uptake_inc_conc']) #reduced from 2491 to 2067

"""
Exclude rows whose target values were determined with methods not involving fluorescence
"""
exclude_methods_list = poseidon_functions.read_single_column_text(poseidon_variables.EXCLUDE_METHODS_FILE)
only_appropriate_methods_table = no_missing_values_table[~no_missing_values_table["methods"].isin(exclude_methods_list)] #reduced from 2067 to 1765

"""
Exclude rows uptake units are relative and not absolute
"""
exclude_units_list = poseidon_functions.read_single_column_text(poseidon_variables.EXCLUDE_UNITS_FILE)
only_appropriate_units_table = only_appropriate_methods_table[~only_appropriate_methods_table["cell_uptake_units"].isin(exclude_units_list)] #reduced from 1765 to 1563

exclude_index_list, usable_target_values, usable_conc_values, usable_time_values, missing_time_values, peptide_query_vector = [], [], [], [], [], []
max_lenght, longest_peptide = 0, ""
for index, current_row in only_appropriate_units_table.iterrows():

	"""
	Get the strings without spaces
	"""
	uptake_string = current_row["cell_uptake_value"].replace(" ","")
	conc_string = current_row["cell_uptake_inc_conc"].replace(" ","")
	
	"""
	Conditions to exclude rows are  if their concentration is neither uM or nM, 
	they still represent a relative measure (N/P ratio, between, or Charge ratio),
	or if there are still missing values 
	"""
	if ("uM" not in conc_string) and \
			("nM" not in conc_string) or \
			("N/Pratio" in conc_string) or \
			("between" in conc_string) or \
			("Chargeratio" in conc_string) or \
			(conc_string == "na") or \
			("na" in uptake_string):
		exclude_index_list.append(index)
		continue

	"""
	Time missing values must be counted afterwards to not mess with excluded rows indexes
	"""
	try:
		time_string = current_row["cell_uptake_inc_time"].replace(" ","")
		missing_time_values.append(0)
	except:
		time_string = "0min"
		missing_time_values.append(1)

	"""
	All concentration values will be converted to nM
	"""
	if "nM" in conc_string:
		usable_conc_values.append(float(conc_string.split("nM")[0])/1000)

	elif "uM" in conc_string:
		uM_val_string = conc_string.split("uM")[0]
		usable_conc_values.append(float(uM_val_string.split(" ")[-1]))

	"""
	All uptake values will be standardized to floats
	"""
	if "/" in uptake_string:
		split_uptake_string = uptake_string.split("/")
		uptake_string = float(split_uptake_string[0]) / float(split_uptake_string[1])
	elif "±" in uptake_string:
		uptake_string = uptake_string.split("±")[0]
	elif "+-" in uptake_string:
		uptake_string = uptake_string.split("+-")[0]
	elif "*" in uptake_string:
		split_uptake_string = uptake_string.split("*")
		if "^" in split_uptake_string[1]:
			uptake_string = float(split_uptake_string[0]) * float(float(split_uptake_string[1].split("^")[0])**float(split_uptake_string[1].split("^")[1]))
		else:
			uptake_string = float(split_uptake_string[0]) * float(split_uptake_string[1])
	"""
	Time values will be converted to minutes
	"""
	if "h" in time_string:
		usable_time = str(float(time_string.split("h")[0]) * 60) + "min"
	elif "s" in time_string:
		usable_time = str(float(time_string.split("s")[0]) / 60) + "min"
	else:
		usable_time = time_string

	"""
	Check if string has anomalies 
	"""

	standard_aa_count = 0
	for current_char in current_row["peptide_sequence"]:
		if current_char in poseidon_variables.AA_LIST:
			standard_aa_count += 1

	if len(current_row["peptide_sequence"]) > max_lenght:
		max_lenght = len(current_row["peptide_sequence"])
		longest_peptide = current_row["peptide_sequence"]

	if standard_aa_count == len(current_row["peptide_sequence"]):
		peptide_query_vector.append(1)
	else:
		peptide_query_vector.append(0)

	usable_time_values.append(float(usable_time.replace("min","")))
	usable_target_values.append(float(uptake_string))

clean_target_table = only_appropriate_units_table.drop(exclude_index_list, axis = 0) #Reduced from 1563 to 1316
clean_target_table["target"] = usable_target_values
clean_target_table["conc_uM"] = usable_conc_values
clean_target_table["time_min"] = usable_time_values
clean_target_table["time_missing"] = missing_time_values
clean_target_table["standard_sequence"] = peptide_query_vector

print("Max peptide length:", max_lenght, longest_peptide)

minimal_clean_target_table = clean_target_table.drop(["cell_uptake_units","methods","cell_uptake_value", "cell_uptake_inc_conc", "cell_uptake_inc_time"], axis = 1)

encoded_temperature_table = poseidon_functions.encoding_column(minimal_clean_target_table, "cell_uptake_inc_temp", \
												possible_values = ["4", "23", "25", "37", "42"], \
												column_prefix = "temperature_", remove_original = True)

normalised_cargos = poseidon_functions.read_single_column_text(poseidon_variables.NORM_CARGOS_FILE, uniform_font = True)
encoded_cargo_table = poseidon_functions.encoding_column(encoded_temperature_table, "cargo", \
												possible_values = normalised_cargos, \
												column_prefix = "cargo_", remove_original = True, \
												uniform_font = True)

encoded_cargo_table.to_csv("cargo_encoded.csv", index = False)
