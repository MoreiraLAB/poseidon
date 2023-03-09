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
Import keras variables
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, Input, metrics
from tensorflow import keras

random.seed(poseidon_variables.RANDOM_SEED)
np.random.seed(poseidon_variables.RANDOM_SEED)

normalized_train_features = np.genfromtxt(poseidon_variables.TRAIN_FEATURES_FILE, delimiter = poseidon_variables.CSV_SEP)
normalized_test_features = np.genfromtxt(poseidon_variables.TEST_FEATURES_FILE, delimiter = poseidon_variables.CSV_SEP)
train_target = np.genfromtxt(poseidon_variables.TRAIN_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)
test_target = np.genfromtxt(poseidon_variables.TEST_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)

class forked_neural_network(Model):

    """
    Standard neural network class for iterative deployment
    """
    def __init__(self, hidden_layers_number = 5,  \
                        usable_features = [""], \
                        input_features = 1, \
                        activation_function = "relu", \
                        add_dropout = True, \
                        dropout_rate = 0.5, \
                        hidden_layers_config = {}, \
                        features_sizes = {}):
        super().__init__()
        
        self.usable_features = usable_features
        self.forked_model = {}

        for current_feature_block in usable_features:
            branch_model = Sequential()
            branch_model.add(Input(shape = (features_sizes[current_feature_block])))
            branch_model.add(Dense(hidden_layers_config[current_feature_block + "_layer_size"], input_dim = features_sizes[current_feature_block], \
                                        activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))

            for current_hidden_layer in range(hidden_layers_number - 1):
                if add_dropout == True:
                    branch_model.add(Dropout(dropout_rate))
                branch_model.add(Dense(hidden_layers_config[current_feature_block + "_layer_size"], activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))
            if add_dropout == True:
                branch_model.add(Dropout(dropout_rate))
            branch_model.add(Dense(1, activation = 'linear'))
            self.forked_model[current_feature_block] = branch_model
        
        self.merge_branches = Add()
        self.final_dense = Dense(1, activation = "linear")

    def call(self, inputs):

        branches_list = []
        for index, current_feature_block in enumerate(self.usable_features):
            data = inputs[index]
            data = self.forked_model[current_feature_block](data)
            branches_list.append(data)

        merged = self.merge_branches(branches_list)
        output = self.final_dense(merged)
        return output

def train_forked_DL(config):

	keras_model =  forked_neural_network(hidden_layers_number = config["depth"],  \
	                        dropout_rate = config["dropout"], add_dropout = config["use_dropout"], \
	                        usable_features = poseidon_variables.FEATURES_BLOCKS, \
	                        hidden_layers_config = config, features_sizes = poseidon_variables.FEATURES_SIZES_DICTIONARY)
	optimizer = tf.keras.optimizers.Adam(config["lr"])
	metrics_list = [
	    keras.metrics.MeanSquaredError(name = "MSE"),
	    keras.metrics.RootMeanSquaredError(name = "RMSE")
	]

	keras_model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = metrics_list, experimental_run_tf_function = False)
	keras_model.fit(
	    train_array_list,
	    train_target,
	    epochs = config["epochs"],
	    validation_split = 0.10,
	    verbose = 1)
	return keras_model

class neural_network:

    """
    Standard neural network class for iterative deployment
    """
    def __init__(self, input_architecture,  \
                        input_features, \
                        activation_function = "relu", \
                        add_dropout = True, \
                        dropout_rate = 0.5):
        self.model = Sequential()
        self.model.add(Dense(input_architecture[0], input_dim = input_features, \
                                    activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
        for hidden_layer in input_architecture[1:]:
            if add_dropout == True:
                self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(hidden_layer, activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
            
        self.model.add(Dense(1, activation = 'linear'))

def deploy_DL(config):

    """
    Deploy the ray tune for simple DL
    """
    architecture_list = [normalized_train_features.shape[1]] + [config["layer_size"]]*config["depth"]
    keras_model = neural_network(architecture_list, normalized_train_features.shape[1], \
        add_dropout = config["add_dropout"], dropout_rate = config["dropout_rate"])

    metrics_list = [
        keras.metrics.MeanSquaredError(name = "MSE"),
        keras.metrics.RootMeanSquaredError(name = "RMSE")
    ]
    keras_model.model.compile(
        loss = "mean_squared_error",
        optimizer = Adam(learning_rate = config["lr"]),
        metrics = metrics_list,
    )

    keras_model.model.fit(
        normalized_train_features, train_target,
        epochs = config["epochs"], verbose = 1
    )
    return keras_model.model

opened_header = [x.replace(poseidon_variables.PARAGRAPH_SEP, "") for x in open(poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "features_header.txt").readlines()]
train_features_df = pd.DataFrame(normalized_train_features, columns = opened_header)
test_features_df = pd.DataFrame(normalized_test_features, columns = opened_header)

import copy 
train_features_dicionary = copy.deepcopy(poseidon_variables.PLACEHOLDER_FEATURES_DICTIONARY)
test_features_dicionary = copy.deepcopy(poseidon_variables.PLACEHOLDER_FEATURES_DICTIONARY)
#1360 final features

for current_feature_block in poseidon_variables.FEATURES_DICTIONARY:
	feature_block_list = []
	for feature_start in poseidon_variables.FEATURES_DICTIONARY[current_feature_block]:
		feature_block_list += [entry for entry in opened_header if entry.startswith(feature_start)]
	train_features_dicionary[current_feature_block] = train_features_df[list(set(feature_block_list))]
	test_features_dicionary[current_feature_block] = test_features_df[list(set(feature_block_list))]
	
train_array_list, test_array_list = [], []
for current_feature_block in poseidon_variables.FEATURES_BLOCKS:
	train_array_list.append(train_features_dicionary[current_feature_block].values)
	test_array_list.append(test_features_dicionary[current_feature_block].values)

def fetch_DL_parameters(input_file, target_variable = ""):

    """
    Open the file with the parameters computed by Ray tune
    """
    opened_table = pd.read_csv(input_file, sep = ",", header = 0)
    opened_table = opened_table.sort_values(by = target_variable, ascending = False).iloc[0,:]
    parameters_cell = opened_table[target_variable].split(poseidon_variables.CSV_SEP)
    parameters_cell_1 = "_".join(parameters_cell[0].split("_")[1:])
    all_parameters = [parameters_cell_1] + parameters_cell[1:]
    output_dictionary = {}
    for current_parameter in all_parameters:
        split_parameter = current_parameter.split("=")
        output_dictionary[split_parameter[0]] = poseidon_functions.process_parameter_type(split_parameter[0], split_parameter[1])
    return output_dictionary

"""
Deploy the pipeline for the forked Deep Learning architecture
"""

forked_parameters_dictionary = fetch_DL_parameters(poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "forked_DL_HPT.csv", \
                            target_variable = "experiment_tag")

forked_prediction_model = train_forked_DL(forked_parameters_dictionary)
forked_prediction_model.save(poseidon_variables.MODELS_FOLDER + poseidon_variables.SYSTEM_SEP + "forked_DL_model.h5py")

forked_predicted_train = forked_prediction_model.predict(train_array_list)
forked_predicted_test = forked_prediction_model.predict(test_array_list)

forked_train_performance = poseidon_functions.performance_evaluation(train_target, forked_predicted_train, \
	verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "forked_train.csv")
forked_test_performance = poseidon_functions.performance_evaluation(test_target, forked_predicted_test, \
	verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "forked_test.csv")

"""
Deploy the pipeline for the Deep Learning architecture and parameters
"""
DL_parameters_dictionary = fetch_DL_parameters(poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "DL_HPT.csv", \
                            target_variable = "experiment_tag")

DL_prediction_model = deploy_DL(DL_parameters_dictionary)
DL_prediction_model.save(poseidon_variables.MODELS_FOLDER + poseidon_variables.SYSTEM_SEP + "DL_model.h5")

DL_predicted_train = DL_prediction_model.predict(normalized_train_features)
DL_predicted_test = DL_prediction_model.predict(normalized_test_features)

DL_train_performance = poseidon_functions.performance_evaluation(train_target, DL_predicted_train, \
    verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "DL_train.csv")
DL_test_performance = poseidon_functions.performance_evaluation(test_target, DL_predicted_test, \
    verbose = True, report = poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + "DL_test.csv")
