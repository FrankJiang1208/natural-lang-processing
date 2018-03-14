#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 6
# March 22, 2018

from nltk.classify import MaxentClassifier
from nltk.corpus import names

def load_training_data(filepath):
    """
    Load training data and convert to List-of-Dicts format.
    """
    with open(filepath, "r") as f:
        training_raw = f.readlines()

    return convert_raw_to_dicts(training_raw)

def convert_raw_to_dicts(raw_data):
    """
    Helper function to convert List-of-RawData-Lines to List-of-Dicts
    """
    features_list = [
                    line.split("\t") for line in raw_data
                    if (line != "\n" and "DOCSTART" not in line) # omit first line and empty lines
                    ]

    training_data = [
                        {"token": token,
                        "pos": pos,
                        "chunk": chunk,
                        "name": name.strip(), # remove newline
                        }
                    for (token, pos, chunk, name) in features_list
                    ]

    return training_data

print(load_training_data("CONLL_train.pos-chunk-name"))