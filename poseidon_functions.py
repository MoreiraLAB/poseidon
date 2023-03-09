#!/usr/bin/env python

"""
Generate variables for poseidon
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "POSEIDON"

import poseidon_variables

def read_single_column_text(input_file, uniform_font = False):

	"""
	Open a single column txt file and retrieve the input as Python list
	"""
	opened_file = open(input_file, "r").readlines()
	if uniform_font == True:
		return [x.replace(poseidon_variables.PARAGRAPH_SEP, "").lower() for x in opened_file]
	return [x.replace(poseidon_variables.PARAGRAPH_SEP, "") for x in opened_file]

def clean_string(input_string):
	"""
	Remove non utf-8 characters from string
	"""
	try:
		return input_string.encode('utf-8', errors = 'ignore').decode('utf-8')
	except:
		return "nan"

def encoding_column(input_table, target_column, \
								possible_values  = False, \
								column_prefix = "", remove_original = False, \
								uniform_font = False):

	"""
	Take an input table, a target column, unfold into different columns that represent discrete encoding
	of a discrete variable.
	Define a list of possible values if you want an encoding that contains the string,
	if you do not insert the value, all the column will indicate 0
	"""

	if possible_values == False:
		unique_columns_list = [input_table[target_column].unique()]

	else:
		unique_columns_list = possible_values

	column_dictionary = {}

	"""
	Create empty dictionary with vectors for each possible value
	"""

	if possible_values == False:
		for unique_column in unique_columns_list:
			column_dictionary[clean_string(unique_column)] = []

	elif possible_values != False:
		for unique_column in possible_values:
			column_dictionary[unique_column] = []

	"""
	Iterate of the table to fill the dictionary with one-hot encoding
	of the discrete values
	"""
	for index, row in input_table.iterrows():
		current_value = clean_string(row[target_column])
		if uniform_font == True:
			current_value = current_value.lower()
		for current_unique_column in unique_columns_list:
			if possible_values != False:
				if current_unique_column in current_value:
					column_dictionary[current_unique_column].append(1)
				else:
					column_dictionary[current_unique_column].append(0)
			else:
				if current_value == current_unique_column:
					column_dictionary[current_unique_column].append(1)
				else:
					column_dictionary[current_unique_column].append(0)

	"""
	Add the columns to the starting table, if prefix was added, it will show here
	"""
	for appendable_unique_column in unique_columns_list:
		input_table[column_prefix + appendable_unique_column] = column_dictionary[appendable_unique_column]

	"""
	If active, this argument removes the original target column
	Default behavior set to False
	"""
	if remove_original == True:
		input_table = input_table.drop([target_column], axis = 1)
		
	return input_table

def performance_evaluation(input_class, input_predictions, verbose = False, report = False):

	"""
	Performance evaluation for regression methods approaches.
	Pass the file path for "report" if you wish to write the results to a csv
	"""
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
	import math
	from scipy.stats import pearsonr, spearmanr
	import numpy as np
	try:
		list_input_class = list(input_class.iloc[:,0])
	except:
		list_input_class = list(input_class)

	list_input_predictions = list(input_predictions)
	try:
		RMSE = math.sqrt(mean_squared_error(input_class, input_predictions))
	except:
		RMSE = 10000
	try:
		MSE = mean_squared_error(input_class, input_predictions)
	except:
		MSE = 10000
	try:
		Pearson, pval_Pearson = pearsonr([float(x) for x in list_input_class], [float(x) for x in list_input_predictions])
	except:
		Pearson = -1.0
	try:
		r2 = r2_score(input_class, input_predictions)
	except:
		r2 = -1.0
	try:
		MAE = mean_absolute_error(input_class, input_predictions)
	except:
		MAE = 10000
	try:
		Spearman, pval_Spearman = spearmanr(list_input_class, list_input_predictions)
	except:
		Spearman = -1.0

	results_report = {"RMSE": RMSE, "MSE": MSE, "MAE": MAE,"Pearson": Pearson,"Spearman": Spearman,"r2": r2}

	if verbose == True:
		print(results_report)

	if report != False:
		with open(report, "w") as output_file:
			for current_metric in results_report:
				output_file.write(current_metric + poseidon_variables.CSV_SEP + \
					str(results_report[current_metric]) + poseidon_variables.PARAGRAPH_SEP)
	return results_report

def write_hpt_results(input_dictionary, output_location):

	"""
	Report HyperParameter Tuning results
	"""
	import pandas as pd
	output_results_list = []
	for current_parameter in input_dictionary:
		output_results_list.append([current_parameter, str(input_dictionary[current_parameter])])
	output_results_dataframe = pd.DataFrame(output_results_list, columns = ["parameter_name", "best_result"])
	output_results_dataframe.to_csv(output_location, index = False)

def process_parameter_type(input_parameter, input_value):

    """
    Process an input value into possible data types, namely, and in this order:
    - scientific notation
    - float
    - integer
    - True/False
    - None
    - string 
    """
    if ("." in input_value) or ("e-" in input_value):
        try:
            return float(format(float(input_value), 'f'))
        except:
            return float(input_value)
    try:
        return int(input_value)

    except:
        if input_value == "True":
            return True
        elif input_value == "False":
            return False
        elif input_value == "None":
            return None
        else:
           return input_value	