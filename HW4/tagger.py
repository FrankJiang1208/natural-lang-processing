#!/usr/bin/env python3

# Leslie Huang (LH1036)
# HW 4 Viterbi POS Tagger

import argparse
import itertools
import numpy as np
import re
import sys

# Functions for loading training and test data and converting them to transition "matrix" and emissions "matrix" (both are nested dicts)

def load_tagged_pos_file(file_path):
    """
    Args:
        file_path: path to a .pos file of training data
    Returns:
        List where each element is ["word", "pos"] or "\n" separating sentences
    """
    with open(file_path, "r") as f:
        data_raw = f.readlines()

        # convert to format: [ [word, pos], [word, pos], "\n" ], newlines to designate end of sentence
        pos_list = [line if line == "\n" else line.rstrip().split("\t") for line in data_raw]

    return pos_list


def load_test_words_file(file_path):
    """
    Args:
        file_path: path to a .words file of test data
    Returns:
        List where each element is ["word", "pos"] or "\n" separating sentences
    """
    with open(file_path, "r") as f:
        data_raw = f.readlines()

    test_observations = [line if line == "\n" else line.rstrip() for line in data_raw]

    return test_observations


def calculate_word_emission_counts(pos_list):
    """
    Calculate word emission counts from training POS_list
    Args:
        pos_list: List where each element is ["word", "pos"] or "\n" separating sentences
    Returns:
        Dict of dicts. Outer dict keys: "POS", values = dicts. Inner dict keys: "word", inner dict values: emission counts of "word"
    """
    word_emissions = dict()

    # First get the word counts
    for elem in pos_list:
        if elem != "\n":
            word, pos = elem

            word_emissions.setdefault(pos, {}) # add POS key to outer dict if not already included

            # add word key to POS dict, if not already included
            word_emissions[pos].setdefault(word, 0)
            # if word not in word_emissions[pos].keys():
            #     word_emissions[pos][word] = 0 # initialize here; can initialize to 1 for word smoothing

            # POS and word must exist by now. increment the count
            word_emissions[pos][word] += 1

    return word_emissions


def convert_counts_probabilities(count_dict):
    """
    Calculate word emission probabilities from word counts
    Args:
        count_dict: Result of calculate_word_emission_counts;
        Outer dict keys: "POS", values = dicts. Inner dict keys: "word", inner dict values: emission count of "word"
    Returns:
        Dict of dicts. Same as before except innder dict values are Pr("word"|"POS")
    """
    for outer_key in count_dict:
        denom = sum(count_dict.get(outer_key, {}).values()) # get sum of all word emissions for a state

        for inner_key in count_dict[outer_key].keys():
            count_dict[outer_key][inner_key] = count_dict[outer_key][inner_key] / denom

    return count_dict


def group_words_sentences(pos_list):
    """
    Group words into sentences (which are lists separated by "\n") from a pos_list. Thanks itertools!
    Args:
        pos_list: List where each element is ["word", "pos"] or "\n" separating sentences
    Returns:
        List of lists, where each list contains words from a sentence.
    """
    sentences_list = []
    for _, g in itertools.groupby(pos_list, lambda x: x == "\n"):
        sentences_list.append(list(g)) # Store group iterator as a list

    sentences_list = [g for g in sentences_list if g != ["\n"]]

    return sentences_list


def calculate_transition_counts(sentences_list):
    """
    Calculates state transition counts from a list of sentence lists
    Args:
        sentences_list: list of lists generated from group_words_sentences. Elements of each inner list = words from a sentence.
    Returns:
        Dict of dicts. Outer dict keys: "POS" (states), values: dicts.
        Inner dict keys: "POS" state transitioned TO from the outer dict, inner dict value: count of occurrences of this transition
    """
    transitions = dict()

    for sentence in sentences_list:
        pos_sequence = [term[1] for term in sentence] # each term in sentence is ["word", "POS"]

        pos_transitions = list(zip(["START"] + pos_sequence, pos_sequence + ["END"])) # zips a list of tuples of all transition pairs including start and end

        for pair in pos_transitions:
            this_state, next_state = pair

            transitions.setdefault(this_state, dict()) # add key for this_state to outer dict if it doesn't already exist

            transitions[this_state].setdefault(next_state, 0)
            # if next_state not in transitions[this_state].keys():
            #     transitions[this_state][next_state] = 0

            transitions[this_state][next_state] += 1

    return transitions


# Viterbi helper functions

def get_possible_states(transition_pr_dict):
    """
    Get the possible states (rows in the trellis)
    Args:
        transition_pr_dict: Dict of dicts of state transition probabilities from training data
    Returns:
        List of possible states: ["POS1", "POS2", ...]
    """
    possible_from_state = list(transition_pr_dict.keys()) # all the states that are transitioned FROM (keys of outer dict)
    possible_to_state = list(list(v.keys()) for s, v in transition_pr_dict.items() ) # all the states that are transitioned TO (keys of inner dicts)
    possible_to_state = set(item for sublist in possible_to_state for item in sublist) # flatten

    possible_states = list(set(possible_from_state).union(possible_to_state)) # take the set of the union
    possible_states.remove("END") # remove END (END is handled in special termination state)

    return possible_states


def get_vocab(word_emissions_dict):
    """
    Get vocab of unique words from training data word emission probability dict.
    Args:
        word_emissions_dict: Dict of dicts of word emission probabilities from training data
    Returns:
        Set of unique words
    """
    vocab = [list(v.keys()) for s, v in word_emissions_dict.items()] # all the words that can be emitted from any state (keys of inner dicts)
    vocab = set(item for sublist in vocab for item in sublist) # flatten

    return vocab


def calculate_word_emission_probability(this_word, this_possible_state, word_emissions_dict, vocab, possible_states):
    """
    Calculate word emission probability i.e. Pr("word"|this_possible_state)
    Args:
        this_word: "word" as str
        this_possible_state: "POS" as str
        word_emissions_dict: Dict of dicts of word emission probabilities from training data
        vocab: set of unique words that have occurred in the training data
        possible_states: possible POS states
    Returns:
        Probability as float. Uniform probability if word is unknown
    """
    return (
        word_emissions_dict.get(this_possible_state, {}).get(this_word, 0)
        if this_word in vocab else
        handle_unknown_words(this_word,
                                this_possible_state,
                                word_emissions_dict,
                                vocab,
                                possible_states) # if the word is unknown
    )

def handle_unknown_words(this_word, this_possible_state, word_emissions_dict, vocab, possible_states):
    """
    Returns Pr(word|state) of unknown words not in the training data. Handles numbers, acronyms, and words in title case.
    Args:
        this_word: "word" as str
        this_possible_state: "POS" as str
        word_emissions_dict: Dict of dicts of word emission probabilities from training data
        vocab: set of unique words that have occurred in the training data
        possible_states: possible POS states
    Returns:
        Probability as float.
    """
    if not bool(re.match("[a-zA-z]", this_word)):
        return ( sum(prob for prob in word_emissions_dict[this_possible_state].values() ) / len(word_emissions_dict[this_possible_state].values())
                if this_possible_state == "CD" # unknown numbers most likely emitted from CD
                else 0) # numbers aren't emitted from any other class

    elif this_word.upper() == this_word or this_word.title() == this_word: # allcaps = acronyms, title case = proper nouns emitted from NNP


        return ( sum(prob for prob in word_emissions_dict[this_possible_state].values() ) / len(word_emissions_dict[this_possible_state].values())
                if this_possible_state == "NNP"
                    else 0 )

    else:
        return (1 / len(possible_states))


def calculate_state_transition_pr(prior_state, this_state, transition_pr_dict):
    """
    Calculate state transition probability i.e. Pr(this_state|prior_state)
    Args:
        prior_state, this_state: "POS" as str
        transition_pr_dict: Dict of dicts of state transition probabilities from training data
    Returns:
        Pr(this_state|prior_state) as float. Returns zero if transition has never occurred in training data
    """
    return transition_pr_dict.get(prior_state, {}).get(this_state, 0)


def compute_termination_pr(state, transition_pr_dict):
    """
    Calculate termination probabilities i.e. Pr(END|state)
    Args:
        state: "POS" as str
        transition_pr_dict: Dict of dicts of state transition probabilities from training data
    Returns:
        Float of Pr(END|state), is zero if transition has never occurred in training data
    """
    return transition_pr_dict.get(state, {}).get("END", 0) # get state["END"], i.e. Pr(END|state) else is zero


def backtrace_best_path(sentence, trellis, backtracer):
    """
    Traces most likely POS for a sentence
    Args:
        sentence: sentence as list of "word" elements
        trellis: Viterbi "matrix" (is a list of dicts, each dict corresponds to a column)
        backtracer: Viterbi backtrace "matrix" (is a list of dicts, each dict corresponds to a column)
    Returns:
        List of highest probability POS for the sentence ["POS1", "POS2", ...]
    """
    tags = []
    num_obs = len(sentence)

    # Starting with the best final state: for the last word in the sentence and last dict (column) in the Viterbi matrix,
    # retrieve key corresponding to the highest value in the column
    best_state = max(trellis[-1].items(), key=lambda item: item[1])[0]

    for col in reversed(range(1, num_obs)):
        tags.append(best_state)
        best_state = backtracer[col][best_state] # From the backtrace matrix, get the prior state most likely to lead to best_state

    tags.append(best_state) # the for loop does not append the final best_state corresponding to the first word in the sentence

    return list(reversed(tags))


# The main function!
def viterbi(sentence, transition_pr_dict, word_emissions_dict, possible_states, vocab):
    """
    Constructs Viterbi "matrix" and backtrace "matrix" for one sentence.
    Args:
        sentence: sentence as list of "word" elements
        transition_pr_dict: Dict of dicts of state transition probabilities from training data
        word_emissions_dict: Dict of dicts of word emission probabilities from training data
        vocab: set of unique words that have occurred in the training data
        possible_states: possible POS states
    Returns:
        List of most likely POS tags for a sentence.
    """
    trellis = [] # Viterbi matrix: Will be list of dicts
    # each dict corresponds to a "column" of the Viterbi matrix i.e. one of the words in the sentence

    backtracer = [] # Backtrace matrix: Will be list of dicts
    # Each dict corresponds to a "column" of the backtrace matrix

    for col, this_word in enumerate(sentence):
        probabilities = {} # Dict that goes into trellis
        backtrace_states = {} # Dict that goes into backtracer.

        for this_possible_state in possible_states:
            if col == 0: # populate the initial state that transitioned from START

                transition_pr = calculate_state_transition_pr("START",
                                                                this_possible_state,
                                                                transition_pr_dict
                                                                ) # Pr(state|START) for this_possible_state

                emission_pr = calculate_word_emission_probability(this_word,
                                                                    this_possible_state,
                                                                    word_emissions_dict,
                                                                    vocab,
                                                                    possible_states
                                                                    ) # Pr(emission|state)

                probabilities[this_possible_state] = transition_pr * emission_pr # don't need to do backtrace for the first word

            else:

                prior_pr_for_each_path = trellis[-1] # Dict of Pr(prior_state) for all prior_states is the previous element in the trellis list

                # Construct dict of transition probabilities to this state
                # { key: possible_prior_state, value: Pr(this_possible_state|previous_state), ... }
                pr_all_transitions_to_this_state = {
                    possible_prior_state: calculate_state_transition_pr(possible_prior_state,
                                                                        this_possible_state,
                                                                        transition_pr_dict
                                                                        )
                    for possible_prior_state in possible_states
                }

                # in order to select the highest probability path into this_possible_state,
                # Construct dict { key: possible_prior_state, value: Pr(this_state|possible_prior_state) * Pr(prior_state, ... }
                path_probabilities = {
                    state: prior_pr_for_each_path[state] * pr_all_transitions_to_this_state[state]
                    for state in possible_states
                }

                # get the max path to this_possible_state and the prior state on that path
                best_previous_state, max_path_probability = max(path_probabilities.items(), key = lambda item: item[1])

                backtrace_states[this_possible_state] = best_previous_state # fill in the backtrace "column"

                emission_pr = calculate_word_emission_probability(this_word,
                                                                    this_possible_state,
                                                                    word_emissions_dict,
                                                                    vocab,
                                                                    possible_states
                                                                    ) # get Pr(emission|this_possible_state)

                probabilities[this_possible_state] = max_path_probability * emission_pr # calculate Pr for the trellis "cell"

        trellis.append(probabilities) # add this "column" to the "matrix" (add this dict to the list)
        backtracer.append(backtrace_states)

    # termination state
    end_probabilties = {} # final "column" in the matrix: Dict of end probabilities

    for this_possible_state in possible_states:
        prior_pr = trellis[-1][this_possible_state]
        termination_pr = compute_termination_pr(this_possible_state, transition_pr_dict) # Pr(END|this_possible_state)
        end_probabilties[this_possible_state] = termination_pr * prior_pr

    trellis.append(end_probabilties)

    # use the backtracer
    return backtrace_best_path(sentence, trellis, backtracer)


def viterbi_multi(training_filepath, test_filepath, output):
    """
    Runs Viterbi on test data after using training data
    Args:
        training_filepath: path to a .pos file of training data
        test_filepath: path to a .words file of test data
        output: name of file for output
        All args should come in from argparse
    Returns:
        Prints word with their predicted tags to stdout
    """
    # load training data and generate transition probability dictionary and emission probability dictionary
    training_data = load_tagged_pos_file(training_filepath)
    word_emissions_dict = convert_counts_probabilities(calculate_word_emission_counts(training_data))
    transition_pr_dict = convert_counts_probabilities(calculate_transition_counts(group_words_sentences(training_data)))

    possible_states = get_possible_states(transition_pr_dict) # unique possible states (excl START and END)
    vocab = get_vocab(word_emissions_dict) # to check for unknown words

    test_observations = group_words_sentences(load_test_words_file(test_filepath))  # load test data as list of sentences

    all_sentences_tags = (
        viterbi(sentence, transition_pr_dict, word_emissions_dict, possible_states, vocab)
        for sentence in test_observations
    )

    for sentence, tags in zip(test_observations, all_sentences_tags):
        for word, tag in zip(sentence, tags):
            print("{}\t{}".format(word, tag), file = output)
        print(file = output)



######################################################
######################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help = "path to the training data")
    parser.add_argument("test", help="path to the test data")
    parser.add_argument("-o", "--output", help = "file path to write to") # optional
    args = parser.parse_args()

    if args.output is not None:
        with open(args.output, 'w') as f:
            viterbi_multi(args.training, args.test, f)
    else:
        viterbi_multi(args.training, args.test, sys.stdout) # prints results to stdout if no output file is specified
