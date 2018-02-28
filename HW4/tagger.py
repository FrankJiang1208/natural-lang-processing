#! /usr/bin/local/python3

# Leslie Huang (LH1036)
# HW 4 Viterbi POS Tagger

import argparse
import itertools
import numpy as np
import sys

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

            # add POS key to outer dict if not already included
            if pos not in word_emissions.keys():
                word_emissions[pos] = {}

            # add word key to POS dict, if not already included
            if word not in word_emissions[pos].keys():
                word_emissions[pos][word] = 0 # initialize here; can initialize to 1 for word smoothing

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
        denom = sum(count_dict[outer_key][inner_key] for inner_key in count_dict[outer_key].keys() ) # get sum of all word emissions for a state

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

            if this_state not in transitions.keys():
                transitions[this_state] = dict()

            if next_state not in transitions[this_state].keys():
                transitions[this_state][next_state] = 0

            transitions[this_state][next_state] += 1

    return(transitions)


######################################################
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
        1 / len(possible_states) # if the word is unknown
    )


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


# def compute_termination_probabilities(possible_states, transition_pr_dict):
#     """
#     Calculate termination probabilities i.e. Pr(END|prior_state) for prior_state in possible_states
#     Args:
#         possible_states: list of possible POS states
#         transition_pr_dict: Dict of dicts of state transition probabilities from training data
#     Returns:
#         List of probabilities: [Pr(END|prior_state), ...] where each list element corresponds to a possible state
#         Pr(END|prior_state) =  zero if transition has never occurred in training data
#     """
#     return [transition_pr_dict.get(state, {}).get("END", 0) for state in possible_states]


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
    # @param sentence A sentence as a list of words
    # @param trellis Viterbi matrix
    # @param backtracer backtrace array
    # returns a list of predicted POS tags for that sentence
    tags = []
    num_obs = len(sentence)

    best_state = max(trellis[-1].items(), key=lambda item: item[1])[0]

    for col in reversed(range(1, num_obs)):
        tags.append(best_state)
        best_state = backtracer[col][best_state]

    tags.append(best_state)

    return list(reversed(tags))


######################################################
# Viterbi algorithm
def viterbi(sentence, training_transitions, training_emissions, possible_states, vocab):
    trellis = []
    backtracer = []

    for col, this_word in enumerate(sentence):
        probabilities = {}
        backtrace_states = {}

        for this_possible_state in possible_states:
            # populate the initial state that transitioned from START
            if col == 0:
                transition_pr = calculate_state_transition_pr("START", this_possible_state, training_transitions) # Pr(state|START) for this_possible_state

                emission_pr = calculate_word_emission_probability(this_word, this_possible_state, training_emissions, vocab, possible_states) # Pr(emission|state)

                probabilities[this_possible_state] = transition_pr * emission_pr # fill in the cell: initial state following START
            # populate cols 2 through N-1
            else:
                # We will calculate Pr(prior_state) * Pr(this_possible_state|prior_state) for each possible prior_state,
                # in order to select the highest probability path into this_possible_state

                prior_pr_for_each_path = trellis[-1] # Pr(prior_state) is the previous column in Viterbi matrix

                # get Pr(this_possible_state|previous_state) for all possible previous states
                pr_all_transitions_to_this_state = {
                    possible_prior_state: calculate_state_transition_pr(possible_prior_state, this_possible_state, training_transitions)
                    for possible_prior_state in possible_states
                }

                # Pr(this_state|prior_state)*Pr(prior_state) for each possible prior_state
                path_probabilities = {
                    state: prior_pr_for_each_path[state] * pr_all_transitions_to_this_state[state]
                    for state in possible_states
                }

                best_previous_state, max_path_probability = max(path_probabilities.items(), key=lambda item: item[1]) # find the max

                backtrace_states[this_possible_state] = best_previous_state # fill in the backtrace column

                emission_pr = calculate_word_emission_probability(this_word, this_possible_state, training_emissions, vocab, possible_states) # get Pr(emission|this_possible_state)

                probabilities[this_possible_state] = max_path_probability * emission_pr # fill in the trellis cell

        trellis.append(probabilities)
        backtracer.append(backtrace_states)

    end_probabilties = {}
    for this_possible_state in possible_states: # fill in the cell at trellis[row, col]
        prior_pr = trellis[-1][this_possible_state]
        termination_pr = compute_termination_pr(this_possible_state, training_transitions) # Pr(END|this_possible_state)
        end_probabilties[this_possible_state] = termination_pr * prior_pr # no emission in END state

    trellis.append(end_probabilties)

    # now use the backtracer
    return backtrace_best_path(sentence, trellis, backtracer)

def viterbi_multi(training_filepath, test_filepath, output):
    # load training data and generate transition probability dictionary and emission probability dictionary
    training_data = load_tagged_pos_file(training_filepath)
    training_emissions = convert_counts_probabilities(calculate_word_emission_counts(training_data))
    training_transitions = convert_counts_probabilities(calculate_transition_counts(group_words_sentences(training_data)))

    possible_states = get_possible_states(training_transitions) # unique possible states (excl START and END)
    vocab = get_vocab(training_emissions) # to check for unknown words

    test_observations = group_words_sentences(load_test_words_file(test_filepath))  # load test data as list of sentences (sentence = list of words)

    all_sentences_tags = (
        viterbi(sentence, training_transitions, training_emissions, possible_states, vocab)
        for sentence in test_observations
    )

    for sentence, tags in zip(test_observations, all_sentences_tags):
        for word, tag in zip(sentence, tags):
            print('{}\t{}'.format(word, tag), file=output)
        print(file=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', help='path to the training data')
    parser.add_argument('test', help='path to the test data')
    parser.add_argument('-o', '--output', help='file path to write to')
    args = parser.parse_args()

    if args.output is not None:
        with open(args.output, 'w') as f:
            viterbi_multi(args.training, args.test, f)
    else:
        viterbi_multi(args.training, args.test, sys.stdout)
