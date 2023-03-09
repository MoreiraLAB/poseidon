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
import numpy as np
import pandas as pd
import random
import poseidon_functions
import math
import pickle

"""
Import non-keras prediction models
"""
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

random.seed(poseidon_variables.RANDOM_SEED)
np.random.seed(poseidon_variables.RANDOM_SEED)

def retrieve_parameters(input_file):

    """
    Open the HyperParameter tuning files and retrieve dictionary with the best performing parameters 
    """
    opened_table = pd.read_csv(input_file, sep = poseidon_variables.CSV_SEP, header = 0)
    output_dictionary = {}
    for index, row in opened_table.iterrows():
        output_dictionary[row["parameter_name"]] = poseidon_functions.process_parameter_type(row["parameter_name"], row["best_result"])
    return output_dictionary

"""
Load train and test data in appropriate formats and fully processed
"""
normalized_train_features = np.genfromtxt(poseidon_variables.TRAIN_FEATURES_FILE, delimiter = poseidon_variables.CSV_SEP)
normalized_test_features = np.genfromtxt(poseidon_variables.TEST_FEATURES_FILE, delimiter = poseidon_variables.CSV_SEP)
train_target = np.genfromtxt(poseidon_variables.TRAIN_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)
test_target = np.genfromtxt(poseidon_variables.TEST_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)

ML_models_dictionary = {
    "linear": LinearRegression(n_jobs = -1), 
    "svm": svm.SVR(),
    "sgd": SGDRegressor(random_state = poseidon_variables.RANDOM_SEED, loss = "squared_error"),
    "knn": KNeighborsRegressor(n_jobs = -1),
    "dt": DecisionTreeRegressor(random_state = poseidon_variables.RANDOM_SEED),
    "rf": RandomForestRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1),
    "etr": ExtraTreesRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1),
    "xgb": xgb.XGBRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1)                        }

for current_method in ML_models_dictionary:
    print("Currently tuning:", current_method)
    predictor = ML_models_dictionary[current_method]
    parameter_path = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + current_method + "_HPT.csv"
    parameters_dictionary = retrieve_parameters(parameter_path)
    
    predictor.set_params(**parameters_dictionary)
    predictor.fit(normalized_train_features, train_target)
    predicted_train = predictor.predict(normalized_train_features)
    predicted_test = predictor.predict(normalized_test_features)

    train_performance = poseidon_functions.performance_evaluation(train_target, predicted_train, \
    verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + current_method + "_train.csv")
    test_performance = poseidon_functions.performance_evaluation(test_target, predicted_test, \
        verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + current_method + "_test.csv")
    model_name = poseidon_variables.MODELS_FOLDER + poseidon_variables.SYSTEM_SEP + current_method + ".h5"
    pickle.dump(predictor, open(model_name, 'wb'))




