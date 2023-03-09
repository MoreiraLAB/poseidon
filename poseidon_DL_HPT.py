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
Import packages and variables for HyperParatemeter Tuning Optimization
"""
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray import tune
from ray.air.config import RunConfig


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
train_target = np.genfromtxt(poseidon_variables.TRAIN_TARGET_FILE, delimiter = poseidon_variables.CSV_SEP)

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

	callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10)
	keras_model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = metrics_list, experimental_run_tf_function = False)
	keras_model.fit(
	    train_array_list,
	    train_target,
	    epochs = config["epochs"],
	    validation_split = 0.10,
	    verbose = 1, callbacks = [callback])
	history = keras_model.history
	tune.report(mse = np.mean(history.history['MSE']), rmse = np.mean(history.history["RMSE"]), \
		val_mse = np.mean(history.history["val_MSE"]), val_rmse = np.mean(history.history["val_RMSE"]))

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
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10)
    keras_model.model.fit(
        normalized_train_features, train_target,
        epochs = config["epochs"], verbose = 1, callbacks = [callback], 
        validation_split = 0.10
    )
    history = keras_model.model.history
    tune.report(mse = np.mean(history.history['MSE']), rmse = np.mean(history.history["RMSE"]), \
        val_mse = np.mean(history.history["val_MSE"]), val_rmse = np.mean(history.history["val_RMSE"]))


opened_header = [x.replace(poseidon_variables.PARAGRAPH_SEP, "") for x in open(poseidon_variables.SUPPORT_FOLDER + poseidon_variables.SYSTEM_SEP + "features_header.txt").readlines()]
train_features_df = pd.DataFrame(normalized_train_features, columns = opened_header)

import copy 
train_features_dicionary = copy.deepcopy(poseidon_variables.PLACEHOLDER_FEATURES_DICTIONARY)
#1360 final features

for current_feature_block in poseidon_variables.FEATURES_DICTIONARY:
	feature_block_list = []
	for feature_start in poseidon_variables.FEATURES_DICTIONARY[current_feature_block]:
		feature_block_list += [entry for entry in opened_header if entry.startswith(feature_start)]
	train_features_dicionary[current_feature_block] = train_features_df[list(set(feature_block_list))]
	
forked_parameters_dictionary = {"depth": tune.qrandint(1, 10), \
			"dropout": tune.quniform(0.1, 0.9, q = 0.1), \
			"use_dropout": tune.grid_search([True, False]), \
			"lr": tune.quniform(0.00001, 0.001, q = 0.00001), \
			"experimental_layer_size": tune.qrandint(5, 50), \
			"cargo_layer_size": tune.qrandint(25, 250), \
			"sequence_layer_size": tune.qrandint(5, 200), \
			"peptideR_layer_size": tune.qrandint(10, 300), \
			"sequence_encoding_layer_size": tune.qrandint(100, 1000), \
			"genomics_layer_size": tune.qrandint(100, 750), \
			"anomalous_layer_size": tune.qrandint(5, 50), \
			"epochs": tune.qrandint(100, 1000, 10)
			}

DL_parameters_dictionary = {
            "depth": tune.qrandint(1, 10), \
            "layer_size": tune.qrandint(100, 1500, 100), \
            "add_dropout": tune.grid_search([True, False]), \
            "dropout_rate": tune.quniform(0.1, 0.9, q = 0.1), \
            "epochs": tune.qrandint(100, 1000, 10), \
            "lr": tune.quniform(0.00001, 0.001, q = 0.00001)}


train_array_list = []
for current_feature_block in poseidon_variables.FEATURES_BLOCKS:
	train_array_list.append(train_features_dicionary[current_feature_block].values)

forked_tuner = tune.Tuner(trainable = train_forked_DL, param_space = forked_parameters_dictionary,
                        tune_config = tune.TuneConfig(
                            num_samples = 30,
                            scheduler = ASHAScheduler(metric = "val_mse", mode = "min"),
                            max_concurrent_trials = 1
                        ),
                        run_config = RunConfig(name = "forked_poseidon")
                        ) #picked validation loss as a metric

analysis = forked_tuner.fit()
output_results_list = []
for current_result_index in range(len(analysis)):
    result = analysis[current_result_index]
    
    if not result.error:
        results_dictionary = result.metrics
        del results_dictionary["config"]
        results_dataframe = pd.DataFrame.from_dict(results_dictionary, orient = "index").transpose()
        output_results_list.append(results_dataframe)
    else:
        print(f"Trial failed with error {result.error}.")

output_results_dataframe = pd.concat(output_results_list, axis = 0).to_csv(poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + \
	"forked_DL_HPT.csv", index = False)

"""
Perform simple Deep Learning based approach HyperParameter Tuning 
"""

DL_tuner = tune.Tuner(trainable = deploy_DL, param_space = DL_parameters_dictionary,
                        tune_config = tune.TuneConfig(
                            num_samples = 30,
                            scheduler = ASHAScheduler(metric = "val_mse", mode = "min"),
                            max_concurrent_trials = 1
                        ),
                        run_config = RunConfig(name = "DL_poseidon")
                        ) #picked validation loss as a metric

analysis = DL_tuner.fit()
output_results_list = []
for current_result_index in range(len(analysis)):
    result = analysis[current_result_index]
    
    if not result.error:
        results_dictionary = result.metrics
        del results_dictionary["config"]
        results_dataframe = pd.DataFrame.from_dict(results_dictionary, orient = "index").transpose()
        output_results_list.append(results_dataframe)
    else:
        print(f"Trial failed with error {result.error}.")

output_results_dataframe = pd.concat(output_results_list, axis = 0).to_csv(poseidon_variables.RESULTS_FOLDER + poseidon_variables.SYSTEM_SEP + \
    "DL_HPT.csv", index = False)