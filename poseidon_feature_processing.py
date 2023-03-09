#!/usr/bin/env python

"""
Generate variables for POSEIDON
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
import random
import poseidon_functions

random.seed(poseidon_variables.RANDOM_SEED)
np.random.seed(poseidon_variables.RANDOM_SEED)

def remove_outliers(input_table, target_column, outlier_multiplier = 3, \
		report_outliers = False, outliers_report_file = ""):

	"""
	Identify outliers and remove them.
	Optionally, report the removed samples unto a file
	"""
	target_mean = np.mean(input_table[target_column])
	target_std = np.std(input_table[target_column])
	cut_off = target_std * 3
	lower, upper = target_mean - cut_off, target_mean + cut_off
	started_registering, remove_indexes_list = False, []

	for index, current_row in input_table.iterrows():
		current_target = current_row[target_column]
		if (current_target < lower) or (current_target > upper):
			remove_indexes_list.append(index)
			if started_registering == False:
				outliers_list = [list(current_row.values)]
				started_registering = True
			elif started_registering == True:
				outliers_list.append(list(current_row.values))

	if report_outliers == True:
		outliers_dataframe = pd.DataFrame(outliers_list)
		outliers_dataframe.to_csv(outliers_report_file, index = False)
	
	return opened_features_table.drop(remove_indexes_list)

def convert_to_log10(input_table, target_column):

	"""
	Convert the values on a target column to their log10 counterpart
	-np.inf values were converted back to 0, as it was confirmed that is the actual value
	"""

	log_target = np.log10(input_table[target_column])
	log_target[log_target == -np.inf] = np.nan
	input_table[target_column] = log_target
	output_table = input_table.fillna(0)
	return output_table

def remove_features_variance(input_table, variance_threshold = 0.0, \
		report_var = False, variance_report_file = ""):

	"""
	Remove features according to variance threshold,
	return list with those features
	"""
	variance_list, columns_to_drop = [], [] 
	for current_column in input_table:
		column_variance = input_table[current_column].var()
		if column_variance == 0:
			columns_to_drop.append(current_column)
		variance_list.append([current_column, column_variance])

	if report_var == True:
		variance_dataframe = pd.DataFrame(variance_list, columns = ["Feature", "Variance"])
		variance_dataframe.to_csv(variance_report_file, index = False)

	return columns_to_drop

opened_features_table = pd.read_csv(poseidon_variables.FEATURES_DATASET_FILE, sep = poseidon_variables.CSV_SEP, header = 0)

outlier_free_table = remove_outliers(opened_features_table, poseidon_variables.DATA_TARGET[0], report_outliers = True, \
	outliers_report_file = poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "outliers.csv")

log10_processed_table = convert_to_log10(outlier_free_table, poseidon_variables.DATA_TARGET[0])

"""
Determine columns to drop, either IDs, the target, or low variance
"""
droppable_columns = poseidon_variables.DATA_TARGET + poseidon_variables.DATA_ID_COLUMNS


low_variance_features = remove_features_variance(log10_processed_table.drop(droppable_columns, axis = 1), \
	report_var = True, variance_report_file = poseidon_variables.VARIANCE_LOG_FILE)

log10_processed_table.drop(low_variance_features, axis = 1).to_csv(poseidon_variables.LOG10_PROCESSED_DATA_FILE, index = False)

droppable_columns += low_variance_features

#Data split
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(log10_processed_table, test_size = 0.3, \
			random_state = poseidon_variables.RANDOM_SEED)

train_ids = train_data[poseidon_variables.DATA_ID_COLUMNS]
test_ids = test_data[poseidon_variables.DATA_ID_COLUMNS]

train_target = train_data[poseidon_variables.DATA_TARGET]
test_target = test_data[poseidon_variables.DATA_TARGET]

train_features = train_data.drop(droppable_columns, axis = 1)
test_features = test_data.drop(droppable_columns, axis = 1)

#1360 features left

#Normalise data
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
scaler.fit(train_features)
pickle.dump(scaler, open(poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "scaler.pkl", 'wb'))


header_file_name = poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "features_header.txt"
with open(header_file_name, "w") as header_file:
	for current_column in list(train_features):
		header_file.write(current_column + poseidon_variables.PARAGRAPH_SEP)

normalized_train_features = scaler.transform(train_features)
normalized_test_features = scaler.transform(test_features)

train_ids.to_csv(poseidon_variables.TRAIN_IDS_FILE, sep = poseidon_variables.CSV_SEP, header = 0)
test_ids.to_csv(poseidon_variables.TEST_IDS_FILE, sep = poseidon_variables.CSV_SEP, header = 0)
np.savetxt(poseidon_variables.TRAIN_FEATURES_FILE, normalized_train_features, delimiter = poseidon_variables.CSV_SEP)
np.savetxt(poseidon_variables.TEST_FEATURES_FILE, normalized_test_features, delimiter = poseidon_variables.CSV_SEP)
np.savetxt(poseidon_variables.TRAIN_TARGET_FILE, train_target, delimiter = poseidon_variables.CSV_SEP)
np.savetxt(poseidon_variables.TEST_TARGET_FILE, test_target, delimiter = poseidon_variables.CSV_SEP)
