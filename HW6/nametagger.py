#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 6
# March 22, 2018

import argparse
import collections
import itertools
from geotext import GeoText
from nltk.classify import MaxentClassifier
from nltk.corpus import names
from nltk.corpus import stopwords

### Functions for formatting raw data ###

def group_words_sentences(features_list):
    """
    Group words into sentences (sentence = [str, ...]) and removes str "\n" dividing them
    Args:
        features_list: List whose elements are features w/ tags
    Returns:
        List of lists, where each list contains words from a sentence.
    """
    sentences_list = []
    for _, g in itertools.groupby(features_list, lambda x: x == "\n"):
        sentences_list.append(list(g)) # Store group iterator as a list

    sentences_list = [g for g in sentences_list if g not in [["\n"], ["\n", "\n"]] ]

    return sentences_list


def feature_dict(feature_vector):
    return {
        "token": feature_vector[0],
        "pos": feature_vector[1],
        "chunk": feature_vector[2].strip(),
    }

def convert_sentence(sentence, is_training):
    """
    Converts a sentence (list of lists, inner list = each word and its tags) into correct format for MaxEnt
    Args:
        sentence: each sentence is a list of lists. Each element of the outer list is an inner list that corresponds to a word and its tags
    Returns:
        If training data: [ [({dict of features for a token}, nametag), ...] ]
        If test data: [ {dict of features for a token}, {dict of features for a token}, ...]
    """
    if is_training:
        return [
            (feature_dict(feature_vector), feature_vector[3].strip())
            for feature_vector in sentence
        ]
    else:
        return [feature_dict(feature_vector) for feature_vector in sentence]

def keep_orig_tokens(features_list):
    """
    Takes features_list and returns only the original tokens with "DOCSTART" and "\n" preserved
    """
    return [
            line if (line == "\n" or "DOCSTART" in line)
            else line[0]
            for line in features_list
        ]


class FeatureBuilder:
    def __init__(self, filepath, is_training):
        self.filepath = filepath
        self.data = None
        self.is_training = is_training
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

        self.convert_data(raw_data, self.is_training)


    def convert_data(self, raw_data, is_training):
        """
        Helper function to convert List-of-RawData-Lines to List-of-Dicts
        Args:
            raw_data: Test or training data read into List-of-Strings
            training: Bool, True if data is training else False.
        Returns:
            Data converted into test/training format for MaxEnt
        """
        features_list = [
            line if (line == "\n" or "DOCSTART" in line) else line.split("\t")
            for line in raw_data
        ]

        # prepare self.data for FeatureBuilder:
        features_grouped = [line for line in features_list if "DOCSTART" not in line]
        features_grouped = group_words_sentences(features_list)
        self.data = [convert_sentence(sentence, self.is_training) for sentence in features_grouped]

        # keep DOCSTART and "\n" in orig_data if test/dev, but keep only tokens from lines
        if not self.is_training:
            self.orig_data = keep_orig_tokens(features_list)


    ###########################################################
    ### Sentence-level features ###
    ###########################################################

    def add_sentence_position(self, sentence, is_training):
        """
        Adds sentence position key-value pairs to tokens in a single [sentence]
        Args:
            sentence: [{features_dict},... ] if test data
                    [({features_dict}, nametag), ({features_dict}, name_tag), ...] if training data
            is_training: Bool for training data
        Returns:
            sentence with new key-value for token position in sentence
        """
        if is_training:
            for counter, value in enumerate(sentence):
                value[0]["token_position"] = counter # value is a tuple ({dict}, nametag)

        else:
            for counter, value in enumerate(sentence):
                value["token_position"] = counter

        return sentence

    def add_sentence_boundaries(self, sentence, is_training):
        """
        Add key-value pairs for start and end tokens. Value is Bool. Must be run after token_position is created with add_positions_sentence()
        Args:
            sentence
        Returns:
            sentence with new key-values for start and end tokens
        """
        if is_training:
            for feature in sentence:
                feature[0]["start_token"] = feature[0]["token_position"] == 0
                feature[0]["end_token"] = feature[0]["token_position"] == len(sentence) - 1

        else:
            for feature in sentence:
                feature["start_token"] = feature["token_position"] == 0
                feature["end_token"] = feature["token_position"] == len(sentence) - 1

        return sentence

    def add_sentence_features(self):
        """
        Add sentence positions and boundaries to all sentences in data
        Args:

        Returns:
        """
        for sentence in self.data:
            sentence = self.add_sentence_position(sentence, is_training = self.is_training)
            sentence = self.add_sentence_boundaries(sentence, is_training = self.is_training)

        return self

    def add_prior_state(self, sentence, is_training):
        """
        """
        pass

    def sentence_features(self):
        """
        Add all features that are dependent on sentence-level structure.
        When finished, flatten List-of-Lists (of sentences) into single list of tokens.
        Args:

        Returns:
        """
        self.add_sentence_features()
        self.data = list(itertools.chain.from_iterable(self.data))

        return self

    ###########################################################
    ### Token-level features ###
    ###########################################################

    ### Operate on a single token's dict of features
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

    def tokens_features(self):
        if self.is_training:
            for (features_dict, _) in self.data:
                self.add_case(features_dict)
                self.add_last_char(features_dict)
                self.add_stopword(features_dict)
                self.add_geo(features_dict)

        else:
            for features_dict in self.data:
                self.add_case(features_dict)
                self.add_last_char(features_dict)
                self.add_stopword(features_dict)
                self.add_geo(features_dict)

        return self


### Functions for output of dev/test data

def label_test_data(predicted_classifications, test_fb, output):
    iter_classifications = iter(predicted_classifications)

    for line in test_fb.orig_data:
        if line == "\n" or "DOCSTART" in line:
            print(line, file = output)
        else:
            print("{}\t{}".format(line, next(iter_classifications)), file = output)

    print(file = output)

    # for (features, label) in zip(test_fb.data, predicted_classifications):
    #     print("{}, {}".format(features, label), file = output)
    # print(file = output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help = "path to the training data")
    parser.add_argument("test", help="path to the test data")
    parser.add_argument("-o", "--output", help = "file path to write to") # optional
    args = parser.parse_args()

    training_fb = FeatureBuilder(args.training, is_training = True)
    training_fb.sentence_features()
    training_fb.tokens_features()

    test_fb = FeatureBuilder(args.test, is_training = False)
    test_fb.sentence_features()
    test_fb.tokens_features()

    classifier = MaxentClassifier.train(training_fb.data, max_iter = 2)

    predicted_classifications = classifier.classify_many(test_fb.data)

    counter = collections.Counter(predicted_classifications)
    print(counter)

    print(classifier.show_most_informative_features(10))

    if args.output is not None:
        with open(args.output, "w") as f:
            label_test_data(predicted_classifications, test_fb, f)
    else:
        label_test_data(predicted_classifications, test_fb, sys.stdout) # prints results to stdout if no output file is specified

