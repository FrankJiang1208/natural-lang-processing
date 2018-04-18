#!/usr/bin/env python3

# Leslie Huang (LH1036)
# Natural Language Processing Assignment 7
# April 26, 2018

from glove import GloveModel
from featurebuilder import FeatureBuilder

import argparse
import collections
import itertools
from nltk.classify import MaxentClassifier
import sys

##########################################################
### Functions for output of dev/test data ###
##########################################################


def label_test_data(predicted_classifications, test_fb, output):
    """
    Prints or outputs file of tokens from test data, annotated with named entity tags.
    """
    iter_classifications = iter(predicted_classifications)

    for line in test_fb.orig_data:
        if line == "\n":
            print(line.strip(), file = output) # preserve newlines separating sentences in original data
        else:
            print("{}\t{}".format(line, next(iter_classifications)), file = output)


##########################################################
##########################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training", help = "path to the training data")
    parser.add_argument("test", help = "path to the test data")
    parser.add_argument("glove_filepath", help = "path to trained glove word vector file")
    parser.add_argument("n_iterations", help = "num iterations for MaxEnt", type = int)
    parser.add_argument("-o", "--output", help = "file path to write to") # optional
    args = parser.parse_args()

    glove_model = GloveModel(args.glove_filepath)

    training_fb = FeatureBuilder(args.training, is_training = True, glove_model = glove_model)
    training_fb.add_sentence_features()
    training_fb.token_features()

    test_fb = FeatureBuilder(args.test, is_training = False, glove_model = glove_model)
    test_fb.add_sentence_features()
    test_fb.token_features()

    classifier = MaxentClassifier.train(training_fb.format_data_maxent(), max_iter = args.n_iterations)

    predicted_classifications = classifier.classify_many(test_fb.features)


    if args.output is not None:
        with open(args.output, "w") as f:
            label_test_data(predicted_classifications, test_fb, f)
    else:
        label_test_data(predicted_classifications, test_fb, sys.stdout) # prints results to stdout if no output file is specified

