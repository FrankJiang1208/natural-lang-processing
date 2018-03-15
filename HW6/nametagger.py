#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 6
# March 22, 2018

from nltk.classify import MaxentClassifier
from nltk.corpus import names
from nltk.corpus import stopwords

def load_data(filepath, training = True):
    """
    Load training or test data and convert to List-of-Dicts format.
    Args:
        filepath: Path to test or training data
        training: Bool, True if data is training (incl. token, POS, chunk, and name).
                False if data is dev/test (token, POS, and chunk only)
    Returns:
        Data processed into List-of-Dicts.
    """
    with open(filepath, "r") as f:
        data_raw = f.readlines()

    return convert_data(data_raw, training = training)

def convert_data(raw_data, training = False):
    """
    Helper function to convert List-of-RawData-Lines to List-of-Dicts
    Args:
        raw_data: Test or training data read into List-of-Strings
        training: Bool, True if data is training else False.
    Returns:
        Data processed into List-of-Dicts.
    """
    features_list = [
                    line.split("\t") for line in raw_data
                    if (line != "\n" and "DOCSTART" not in line) # omit first line and empty lines
                    ]

    if training:
        converted_data = [
                        {"token": token,
                        "pos": pos,
                        "chunk": chunk,
                        "name": name.strip(), # remove newline
                        }
                        for (token, pos, chunk, name) in features_list
                        ]
    else:
        converted_data = [
                        {"token": token,
                        "pos": pos,
                        "chunk": chunk,
                        }
                        for (token, pos, chunk) in features_list
                        ]

    return converted_data


class FeatureBuilder(object):
    def __init__(self, data):
        self.data = data

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


training_fb = FeatureBuilder(load_data("CONLL_train.pos-chunk-name", training = True))

print(training_fb.add_last_char_feat().data)

#print(load_data("CONLL_dev.pos-chunk", training = False))