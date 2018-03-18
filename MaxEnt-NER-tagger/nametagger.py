#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 6
# March 22, 2018

import argparse
import collections
import itertools
import sys
from geotext import GeoText
from nltk.classify import MaxentClassifier
from nltk.corpus import names
from nltk.corpus import stopwords

### Functions to format raw data ###

def split_raw_data(raw_data):
    """
    Helper function to convert List-of-RawData-Lines to List-of-Dicts
    Args:
        raw_data: Test or training data read into List-of-Strings
    Returns:

    """
    return [
        line if line == "\n"
        else line.split("\t")
        for line in raw_data
    ]

def group_features_by_sentence(raw_data_split):
    """
    Group words into sentences (sentence = [str, ...]) and removes str "\n" dividing them
    Args:
        raw_data_split: List whose elements are features w/ tags
    Returns:
        List of lists, where each list contains words from a sentence.
    """
    sentences_list = []
    for _, g in itertools.groupby(raw_data_split, lambda x: x == "\n"):
        sentences_list.append(list(g)) # Store group iterator as a list

    return [g for g in sentences_list if g not in [["\n"], ["\n", "\n"]] ]


def feature_dict(feature_vector):
    return {
        "token": feature_vector[0],
        "pos": feature_vector[1],
        "chunk": feature_vector[2].strip(),
    }

# def convert_sentence(sentence, is_training):
#     """
#     Converts a sentence (list of lists, inner list = each word and its tags) into correct format for MaxEnt
#     Args:
#         sentence: each sentence is a list of lists. Each element of the outer list is an inner list that corresponds to a word and its tags
#     Returns:
#         If training data: [ [({dict of features for a token}, nametag), ...] ]
#         If test data: [ {dict of features for a token}, {dict of features for a token}, ...]
#     """
#     if is_training:
#         return [
#             (feature_dict(feature_vector), feature_vector[3].strip())
#             for feature_vector in sentence
#         ]
#     else:
#         return [feature_dict(feature_vector) for feature_vector in sentence]

def extract_labels(sentence):
    return [feature_vector[3].strip() for feature_vector in sentence ]


def extract_features_dict(sentence):
    return [feature_dict(feature_vector) for feature_vector in sentence]


def extract_orig_tokens(raw_data_split):
    """
    Takes raw_data_split and returns only the original tokens with "\n" preserved
    """
    return [
            line if line == "\n"
            else line[0]
            for line in raw_data_split
        ]


class FeatureBuilder:
    def __init__(self, filepath, is_training):
        self.filepath = filepath
        self.is_training = is_training

        self.sentences_features_dicts = None
        self.features = None
        self.labels = None
        self.orig_data = None # For output of test/dev data

        self.load()

    def load(self):
        """
        Load training or test data and convert to List-of-Dicts format.
        Args:
            filepath: Path to test or training data
            training: Bool, True if data is training
        Returns:
            Data converted into test/training format for MaxEnt
        """
        with open(self.filepath, "r") as f:
            raw_data = f.readlines()

        split_data = split_raw_data(raw_data)

        self.extract_features_dicts_by_sentence(split_data)
        self.extract_labels(split_data)
        self.extract_orig_data(split_data)


    def extract_features_dicts_by_sentence(self, split_data):
        features_grouped = group_features_by_sentence(split_data)

        self.sentences_features_dicts = [extract_features_dict(sentence) for sentence in features_grouped]

    def extract_labels(self, split_data):
        features_grouped = group_features_by_sentence(split_data)

        if self.is_training:
            labels = [extract_labels(sentence) for sentence in features_grouped]
            # need to flatten out the structure of sentences within the larger list, because features list will also be flattened
            self.labels = list(itertools.chain.from_iterable(labels))


    def extract_orig_data(self, split_data):
        # for orig_data, keep only token in each line
        if not self.is_training:
            self.orig_data = extract_orig_tokens(split_data)

    def format_data_maxent(self):
        return list(zip(self.features, self.labels))


    ###########################################################
    ### Sentence-level features ###
    ###########################################################

    def add_sentence_position(self, sentence):
        """
        Adds sentence position key-value pairs to tokens in a single [sentence]
        Args:
            sentence: [{features_dict},... ] if test data
                    [({features_dict}, nametag), ({features_dict}, name_tag), ...] if training data
            is_training: Bool for training data
        Returns:
            sentence with new key-value for token position in sentence
        """
        for counter, value in enumerate(sentence):
            value["token_position"] = counter

        return sentence

    def add_sentence_boundaries(self, sentence):
        """
        Add key-value pairs for start and end tokens. Value is Bool. Must be run after token_position is created with add_positions_sentence()
        Args:
            sentence
        Returns:
            sentence with new key-values for start and end tokens
        """
        for feature in sentence:
            feature["start_token"] = feature["token_position"] == 0
            feature["end_token"] = feature["token_position"] == len(sentence) - 1

        return sentence


    def add_prior_state(self, sentence, is_training):
        """
        """
        pass


    def add_sentence_features(self):
        """
        Add sentence positions and boundaries to all sentences in data
        Args:

        Returns:
        """
        features_dicts = []
        for sentence in self.sentences_features_dicts:
            sentence = self.add_sentence_position(sentence)
            sentence = self.add_sentence_boundaries(sentence)
            features_dicts.append(sentence)

        self.features = list(itertools.chain.from_iterable(features_dicts))



    ###########################################################
    ### Token-level features ###
    ###########################################################

    ### Operate on a single token's features_dict
    def add_case(self, features_dict):
        features_dict["case"] = "lower" if features_dict["token"] == features_dict["token"].lower() else "upper"

    def add_last_char(self, features_dict):
        features_dict["last_char"] = features_dict["token"][-1]

    def add_stopword(self, features_dict):
        features_dict["nltk_stopword"] = features_dict["token"] in stopwords.words("english")

    def add_nltk_name(self, features_dict):
        features_dict["is_nltk_name"] = features_dict["token"].lower() in (n.lower() for n in names.words())

    def add_geo(self, features_dict):
        features_dict["is_geo_place"] = bool(GeoText(features_dict["token"]).cities or GeoText(features_dict["token"]).countries)

    def token_features(self):
        for features_dict in self.features:
            self.add_case(features_dict)
            self.add_last_char(features_dict)
            self.add_stopword(features_dict)
            self.add_geo(features_dict)


### Functions for output of dev/test data ###

def label_test_data(predicted_classifications, test_fb, output):
    iter_classifications = iter(predicted_classifications)

    for line in test_fb.orig_data:
        if line == "\n":
            print(line.strip(), file = output)
        else:
            print("{}\t{}".format(line, next(iter_classifications)), file = output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training", help = "path to the training data")
    parser.add_argument("test", help = "path to the test data")
    parser.add_argument("n_iterations", help = "num iterations for MaxEnt", type = int)
    parser.add_argument("-o", "--output", help = "file path to write to") # optional
    args = parser.parse_args()

    training_fb = FeatureBuilder(args.training, is_training = True)
    training_fb.add_sentence_features()
    training_fb.token_features()

    test_fb = FeatureBuilder(args.test, is_training = False)
    test_fb.add_sentence_features()
    test_fb.token_features()

    classifier = MaxentClassifier.train(training_fb.format_data_maxent(), max_iter = args.n_iterations)

    predicted_classifications = classifier.classify_many(test_fb.features)

    # counter = collections.Counter(predicted_classifications) # see how many tokens in each class
    # print(counter)

    # print(classifier.show_most_informative_features(10))

    if args.output is not None:
        with open(args.output, "w") as f:
            label_test_data(predicted_classifications, test_fb, f)
    else:
        label_test_data(predicted_classifications, test_fb, sys.stdout) # prints results to stdout if no output file is specified

