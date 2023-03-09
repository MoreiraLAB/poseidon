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

import pandas as pd

class peptide:

	def __init__(self, sequence):
		self.sequence = sequence
		self.length = len(self.sequence)

	def fetch_R_peptides(self, template_loc = poseidon_variables.R_FEATURES_TEMPLATE):

		import rpy2
		from rpy2 import robjects
		import rpy2.robjects.packages as rpackages
		from rpy2.robjects import pandas2ri
		pandas2ri.activate()
		PeptidesR = rpackages.importr('Peptides')
		reshape = rpackages.importr('reshape2')
		peptides_R_features = {}
		with open(template_loc, "r") as input_R_template:
			for row in input_R_template:
				clean_row = row.replace(poseidon_variables.PARAGRAPH_SEP, "")
				split_row = clean_row.split("<-")
				feature_name = split_row[0].replace(" ","")
				command_row = split_row[1].replace("current_sequence", "'" + self.sequence + "'").replace(" ","")
				current_features = robjects.r(command_row)
				converted_features = robjects.conversion.rpy2py(current_features)
				
				if feature_name == "AAC":
					current_feature_table = pd.DataFrame(converted_features, columns = ["number", "freq"])
					current_feature_table["group"] = poseidon_variables.R_GROUPS
					current_feature_table["freq"] = current_feature_table["freq"] / 100 

					for current_R_group in poseidon_variables.R_GROUPS:
						peptides_R_features["AAC_number_" + current_R_group] = current_feature_table.loc[current_feature_table["group"] == current_R_group]["number"].values[0]
						peptides_R_features["AAC_freq_" + current_R_group] = current_feature_table.loc[current_feature_table["group"] == current_R_group]["freq"].values[0]
					continue

				elif feature_name == "membrane_position":
					
					converted_features["globular"] = 0
					converted_features["surface"] = 0
					converted_features["transmembrane"] = 0
					if (converted_features["MembPos"] == "Transmembrane")[0]:
						converted_features["transmembrane"] = 1

					if (converted_features["MembPos"] == "Globular")[0]:
						converted_features["globular"] = 1

					if (converted_features["MembPos"] == "Surface")[0]:
						converted_features["surface"] = 1

					converted_features = converted_features.drop(["Pep","MembPos"], axis = 1).add_prefix("membrane_position_")
					for current_membrane_position_feature in list(converted_features):
						peptides_R_features[current_membrane_position_feature] = converted_features[current_membrane_position_feature].values[0]
					continue

				else:
					peptides_R_features[feature_name] = converted_features[0]
					
		self.R_features = peptides_R_features

	def representation_features(self, max_length = 1):

		"""
		Iterate over the sequences to generate representation features by one-hot encoding the positions
		"""
		import copy 
		amino_acid_dataframe = pd.DataFrame.from_dict(poseidon_variables.AMINO_ACID_DICTIONARY, orient = "index")
		row_features = []
		for current_position in range(1, max_length + 1):
			aa_df = copy.deepcopy(amino_acid_dataframe)
			try:
				current_aa = self.sequence[current_position - 1]
			except:
				current_aa = None
			if current_aa != None:
				aa_df.loc[current_aa] = 1
			aa_df = aa_df.transpose().add_prefix(str(current_position) + "_")
			row_features.append(aa_df)
		row_features_dataframe = pd.concat(row_features, axis = 1)
		return row_features_dataframe

#peptide_object = peptide("AAWWSGAC")
#peptide_object.fetch_R_peptides()
#print(peptide_object.R_features)