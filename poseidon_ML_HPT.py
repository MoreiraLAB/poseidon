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

"""
Import non-keras prediction models
"""
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from tune_sklearn import TuneSearchCV

random.seed(poseidon_variables.RANDOM_SEED)
np.random.seed(poseidon_variables.RANDOM_SEED)


"""
Load train and test data in appropriate formats and fully processed
"""
normalized_train_features = np.genfromtxt(poseidon_variables.TRAIN_FEATURES_FILE, delimiter = poseidon_variables.CSV_SEP)
train_target = np.genfromtxt(poseidon_variables.TRAIN_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)

ML_models_dictionary = {

    "svm": {"model": svm.SVR(), "kernel": ["linear", "poly", "rbf", "sigmoid"], \
                        "C": [0.5, 1.0, 1.5], "gamma": ["scale", "auto"]},
    "sgd": {"model": SGDRegressor(random_state = poseidon_variables.RANDOM_SEED, loss = "squared_error"), \
                        "penalty": ["l2", "l1", "elasticnet"], "alpha": [0.00001, 0.0001, 0.001], \
                        "learning_rate": ["invscaling", "optimal", "constant", "adaptive"]},
    "knn": {"model": KNeighborsRegressor(n_jobs = -1), "n_neighbors": [2, 3, 5, 7], \
                        "p": [1,2], "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]},
    "dt": {"model": DecisionTreeRegressor(random_state = poseidon_variables.RANDOM_SEED), "splitter": ["best", "random"], \
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"], "max_depth": [None, 3, 5, 10, 50, 100], \
                        "min_samples_split": [2, 3, 5, 7, 10], "min_samples_leaf": [2, 3, 5, 7, 10], \
                        "min_weight_fraction_leaf": [0.0, 0.25, 0.50], "max_features": ["auto", "sqrt", "log2", None]},
    "rf": {"model": RandomForestRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1), "n_estimators": [10, 50, 100, 250], \
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"], "max_depth": [3, 5, 10, 50, 100], \
                        "min_samples_split": [2, 3, 5, 7, 10], "min_samples_leaf": [2, 3, 5, 7, 10], \
                        "min_weight_fraction_leaf": [0.0, 0.25, 0.50]},
    "etr": {"model": ExtraTreesRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1), "n_estimators": [10, 50, 100, 250], \
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"], "max_depth": [None, 3, 5, 10, 50, 100], \
                        "min_samples_split": [2, 3, 5, 7, 10], "min_samples_leaf": [2, 3, 5, 7, 10], \
                        "min_weight_fraction_leaf": [0.0, 0.25, 0.50]},
    "xgb": {"model": xgb.XGBRegressor(random_state = poseidon_variables.RANDOM_SEED, n_jobs = -1), "n_estimators": [10, 50, 100, 250], \
                    "max_depth": [None, 3, 5, 10, 50, 100], "max_leaves": [None, 1, 3, 5, 10, 25], "learning_rate": [None, 0.15, 0.3, 0.46, 0.60, 0.76, 0.90], \
                    "booster": [None, "gbtree", "gblinear", "dart"], "alpha": [0, 1, 3, 5], "lambda": [1, 3, 5],"gamma": [0, 1, 3, 5]}                        
}

for current_method in ML_models_dictionary:
    print("Currently tuning:", current_method)
    predictor = ML_models_dictionary[current_method].pop("model", None)
    tune_search = TuneSearchCV(predictor,
       param_distributions = ML_models_dictionary[current_method], \
       max_iters = 10, \
       n_jobs = -1, random_state = poseidon_variables.RANDOM_SEED
    )

    tune_search.fit(normalized_train_features, train_target)
    poseidon_functions.write_hpt_results(tune_search.best_params_, poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + \
        current_method + "_HPT.csv")

