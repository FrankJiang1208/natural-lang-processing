#! /usr/bin/local/python3

# Leslie Huang (LH1036)
# HW 4 Tagger

import itertools

def load_tagged_pos_file(file_path):
    # read in a .pos file of training data
    # returns POS list where each element is [word, pos]
    with open(file_path, "r") as f:
        data_raw = f.readlines()

        # convert to format: [ [word, pos], [word, pos], "\n" ], newlines to designate end of sentence
        pos_list = [line if line == "\n" else line.rstrip().split("\t") for line in data_raw]
    
    return(pos_list)


def load_test_words_file(file_path):
    # Load a .words file of test data
    # returns list of words
    with open(file_path, "r") as f:
        data_raw = f.readlines()

    test_observations = [line if line == "\n" else line.rstrip() for line in data_raw]

    return(test_observations)


def calculate_word_emission_counts(pos_list):
    # calculate word emission counts from POS_list
    word_emissions = dict()

    # First get the word counts
    for elem in pos_list:
        if elem != "\n":
            word = elem[0]
            pos = elem[1]

            # add POS key to outer dict if not already included
            if pos not in word_emissions.keys():
                word_emissions[pos] = {}
            
            # add word key to POS dict, if not already included
            if word not in word_emissions[pos].keys():
                word_emissions[pos][word] = 0 # initialize here; can initialize to 1 for word smoothing
            
            # POS and word must exist by now, so we can increment the count
            word_emissions[pos][word] += 1
    
    return(word_emissions)


def convert_counts_probabilities(count_dict):
    # convert dict of word counts to dict of probabilities
    # @param count_dict Dictionary in format {Category: {possibility1: n, possibility2: m}, Category2: {etc}}
    for outer_key in count_dict:
        denom = sum(count_dict[outer_key][inner_key] for inner_key in count_dict[outer_key].keys() ) # sum of all counts over inner keys for each outer key

        for inner_key in count_dict[outer_key].keys():
            count_dict[outer_key][inner_key] = count_dict[outer_key][inner_key] / denom

    return(count_dict)


def group_words_sentences(pos_list):
    # Takes a list of POS [ [word, pos], [word, pos]...] or a list of test words [word, word,...] and converts it into list of sentences [ [[sentence1_word, pos], [sentence1_word, pos], ...], [[sentence2_word, pos], ...], ... ]
    # thanks itertools

    sentences_list = []
    for k, g in itertools.groupby(pos_list, lambda x: x == "\n"):
        sentences_list.append(list(g))      # Store group iterator as a list

    sentences_list = [g for g in sentences_list if g != ["\n"]]

    return(sentences_list)


def calculate_transition_counts(sentences_data):
    # calculates state transition counts from list of sentences from group_pos_data_sentences
    transitions = dict()

    for sentence in sentences_data:
        pos_sequence = [term[1] for term in sentence]

        pos_transitions = list(zip(["START"] + pos_sequence, pos_sequence + ["END"])) # zips a list of tuples of all transition pairs including start and end

        for pair in pos_transitions:
            this_state = pair[0]
            next_state = pair[1]

            if this_state not in transitions.keys():
                transitions[this_state] = dict()
            
            if next_state not in transitions[this_state].keys():
                transitions[this_state][next_state] = 0
            
            transitions[this_state][next_state] += 1

    return(transitions)


######################################################
# Viterbi algorithm
def viterbi(training_filepath, test_filepath):

    # load training data and calculate transition and emission probabilities
    training_data = load_tagged_pos_file(training_filepath)

    training_emissions = convert_counts_probabilities(calculate_word_emission_counts(training_data)
                                                        ) 

    training_transitions = convert_counts_probabilities(calculate_transition_counts(
                                                                                    group_words_sentences(training_data)
                                                                                    )
                                                                                    ) 

    # load test data as list of sentences (sentence = list of words)
    test_observations = group_words_sentences(load_test_words_file(test_filepath))

    # tag each sentence one at a time
    all_sentences_tags = []

    for sentence in test_observation:
        this_sentence_tags = []

        prior_probability = 1 # set this to 1 for initial start state

        last_state = "START"

        for word in sentence:

            possible_state_transitions = training_transitions[last_state] # dict of all the states that last_state can transition to (keys), and their probabilities (values)

            possible_state_transitions = { (state, prob * prior_probability) for (state, prob) in possible_state_transitions.items() } # multiply by prior_probability
            
            #deal with unknown words later

            for possible_state in possible_state_transitions.keys():
                if training_emissions[possible_state][word]: # if this word has ever been emitted from this state
                    possible_state_transitions[possible_state] = possible_state_transitions[possible_state] * training_emissions[possible_state][word] # multiply transition probability by word emission probability
                
                else: # if this word has never been emitted from this state
                    possible_state_transitions[possible_state] = 0 # it's zero

                # now we have multipled (STATE | PRIOR STATE) * (WORD | STATE) for all possible states

            # Then locate the state that emitted the word with the highest probability
            max_state_and_word = { trans_state: emission_prob for trans_state, emission_prob in possible_state_transitions.items() if emission_prob == max(possible_state_transitions.values()) }




# Run it
training_filepath = "/Users/lesliehuang/nlp-vc/HW4/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"

test_filepath = "/Users/lesliehuang/nlp-vc/HW4/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"

viterbi(training_filepath, test_filepath)