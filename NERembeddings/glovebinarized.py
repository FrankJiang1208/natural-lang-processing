#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Assignment 7
# April 26, 2018

import numpy as np

###########################################################
### GloveModel helper functions ###
###########################################################

def separate_words_dims(raw_data):
    """
    Helper function to extract list of words and list of lists of dimensions
    Args:
        raw_data: raw data read in from trained vectors file
    Returns:
        words: list of words in order
        dims: list of lists of dimensions corresponding to the words
    """
    words = []
    dims = []

    for line in raw_data:
        line = line.split()

        words.append(line[0]) # first elem in each list is the word
        dims.append(line[1:len(line)]) # elem 1:n in each list are the dimensions values

    return words, dims, (len(line) - 1) # need num of dimensions of the vector not including the word


def create_dimensions_dict(dimensions_vector):
    """
    Creates a dimensions_dict from a dimensions_vector
    Args:
        dimensions_vect [list of floats]
    Returns:
        dimensions_dict of sequentially numbered keys and values from dimensions_vect
        e.g. {"0": float, "2": float, ...}

    """
    return {
        counter: value
        for counter, value in enumerate(dimensions_vector)
    }

def word_vector_dicts(words, dims):
    """
    Creates {word: dimensions_dict} from a word and a dimensions_vector
    Args:
        words: [list of words as str]
        dims: [list of dimensions_vectors, which are themselves lists]
    Returns:
        {word1: {dimensions_dict1}, word2: {dimensions_dict2}, ...}
    """
    return {
                word: create_dimensions_dict(dimensions_vector)
        for word, dimensions_vector in zip(words, dims)
    }

###########################################################
###########################################################
# GloveModel Class

class GloveModel:

    def __init__(self, filepath):
        """
        """
        self.filepath = filepath
        self.trained_vectors = None
        self.binarized_vectors = None
        self.num_dims = None

        self.load_trained_vectors()
        self.binarize_vectors()
        self.default = self.generate_default()


    def load_trained_vectors(self):
        """
        Loads and converts trained vectors from filepath to dict format
        """
        with open(self.filepath, "r") as f:
            raw_data = [line.strip() for line in f.readlines()]

        words, dims, self.num_dims = separate_words_dims(raw_data)

        self.trained_vectors = word_vector_dicts(words, dims)

    def calculate_dimension_means(self):
        """
        For each dimension in word embeddings, calculate (over all words) mean of positive values and mean of negative values
        """
        dimension_means = dict()

        for counter in range(self.num_dims):
            dimension_values = [vector[counter] for vector in self.trained_vectors]

            pos_mean = np.mean([val for val in dimension_values if val > 0])
            neg_mean = np.mean([val for val in dimension_values if val < 0])

            dimension_means[counter] = {"pos_mean": pos_mean,
                "neg_mean": neg_mean
            }

        return dimension_means

    def binarize_vectors(self):
        """
        Binarize each dimension of the word embedding vector
        """
        dimension_means = self.calculate_dimension_means()

        vectors = self.trained_vectors # each "vector" is a dictionary for a word with keys in 0:49 and values corresponding to embedding dim

        for vector in vectors:
            for key in vector.keys():
                if vector[key] >= dimension_means[key]["pos_mean"]:
                    vector[key] = "U_plus"
                elif vector[key] <= dimension_means[key]["neg_mean"]:
                    vector[key] = "B_minus"
                else:
                    vector[key] = 0

        self.binarized_vectors = vectors

    def generate_default(self):
        """
        Generates a word vector dict with all values = 0, for tokens not in trained_vectors dict
        """
        return {
            feature_num: 0
            for feature_num in range(self.num_dims)
        }

    def get(self, token):
        """
        Gets word vector dict else returns default dict if word is unknown
        """
        return self.trained_vectors.get(token, self.default)

