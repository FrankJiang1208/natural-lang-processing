#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 6
# March 22, 2018

import itertools
from geotext import GeoText
from nltk.classify import MaxentClassifier
from nltk.corpus import names
from nltk.corpus import stopwords


class FeatureBuilder(object):
    def __init__(self, filepath, is_training = True):
        self.filepath = filepath
        self.is_training = is_training
        self.data = self.load_data()

    ### Methods for loading data ###

    def load_data(self):
        """
        Load training or test data and convert to List-of-Dicts format.
        Args:
            filepath: Path to test or training data
            training: Bool, True if data is training (incl. token, POS, chunk, and name).
                    False if data is dev/test (token, POS, and chunk only)
        Returns:
            Data converted into test/training format for MaxEnt
        """
        with open(self.filepath, "r") as f:
            raw_data = f.readlines()

        return self.convert_data(raw_data, is_training = self.is_training)

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
                        line if line == "\n" else line.split("\t") for line in raw_data
                        if "DOCSTART" not in line
                        ]

        features_grouped = self.group_words_sentences(features_list)
        return [self.convert_sentence(sentence, is_training = is_training) for sentence in features_grouped]

    def group_words_sentences(self, features_list):
        """
        Group words into sentences (which are lists separated by "\n")
        Args:
            features_list: List whose elements are features w/ tags, with sentences separated by newlines.
        Returns:
            List of lists, where each list contains words from a sentence.
        """
        sentences_list = []
        for _, g in itertools.groupby(features_list, lambda x: x == "\n"):
            sentences_list.append(list(g)) # Store group iterator as a list

        sentences_list = [g for g in sentences_list if g not in [["\n"], ["\n", "\n"]] ]

        return sentences_list


    def convert_sentence(self, sentence, is_training):
        """
        Converts a sentence (list of lists, inner list = each word and its tags) into correct format for MaxEnt
        Args:
            sentence: each sentence is a list of lists. Each element of the outer list is an inner list that corresponds to a word and its tags
        Returns:
            If training data: [ ({dict of features for a token}, nametag), ...]
            If test data: [ {dict of features for a token}, {dict of features for a token}, ...]
        """
        if is_training:
            return ( [
                        (
                            {"token": feature_vector[0],
                            "pos": feature_vector[1],
                            "chunk": feature_vector[2],
                            },
                            feature_vector[3].strip()
                        )
                    for feature_vector in sentence
                        ] )
        else:
            return ( [{"token": feature_vector[0],
                    "pos":feature_vector[1],
                    "chunk": feature_vector[2],
                    }
                for feature_vector in sentence ] )

    ### Methods for building features ###

    # def add_sentence_demarcations(self):
    #     for sentence in self.data:
    #         sentence[-1]["sentence_position"] = "end"
    #         sentence[0]["sentence_position"] = "start"

    def add_case_feat(self):
        for feature in self.data:
            feature["case"] = "lower" if feature["token"] == feature["token"].lower() else "upper"
        return self

    def add_last_char_feat(self):
        for feature in self.data:
            feature["last_char"] = feature["token"][-1]
        return self

    def add_stopword_feat(self):
        for feature in self.data:
            feature["is_nltk_stopword"] = True if feature["token"] in stopwords.words("english") else False
        return self

    def add_nltk_name_feat(self):
        for feature in self.data:
            feature["is_nltk_name"] = True if feature["token"].lower() in [n.lower() for n in names.words()] else False
        return self

    def add_geo_feat(self):
        for feature in self.data:
            feature["is_geo_place"] = True if ( GeoText(feature["token"]).cities or GeoText(feature["token"]).countries ) else False
        return self

training_fb = FeatureBuilder("CONLL_train.pos-chunk-name", is_training = True)
print(training_fb.data)

#print(training_fb.add_last_char_feat().add_stopword_feat().add_geo_feat().add_nltk_name_feat().data)

# test_fb = FeatureBuilder("CONLL_dev.pos-chunk", is_training = False)
# print(test_fb.data)