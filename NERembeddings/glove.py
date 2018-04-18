#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Assignment 7
# April 26, 2018

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
        e.g. {"1": float, "2": float, ...}

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
        self.num_dims = None

        self.load_trained_vectors()
        self.default = self.generate_default()


    def load_trained_vectors(self):
        """
        Loads and converts trained vectors from filepath to dict format
        """
        with open(self.filepath, "r") as f:
            raw_data = [line.strip() for line in f.readlines()]

        words, dims, self.num_dims = separate_words_dims(raw_data)

        self.trained_vectors = word_vector_dicts(words, dims)

    def generate_default(self):
        """
        Generates a word vector dict with all values = 0, for tokens not in trained_vectors dict
        """
        return {
            feature_num: 0
            for feature_num in range(self.num_dims)
        }

    def get(self, token):
        return self.trained_vectors.get(token, self.default)

