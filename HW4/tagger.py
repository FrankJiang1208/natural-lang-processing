#! /usr/bin/local/python3

# Leslie Huang (LH1036)
# HW 4 Tagger

import itertools
import numpy as np

np.set_printoptions(threshold=np.inf)

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
    # calculate word emission counts from training POS_list
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
# Viterbi helper functions

def get_possible_states(transition_pr_dict):
    # takes a transition Pr dict and returns a list of all possible states

    possible_from_state = list(transition_pr_dict.keys()) # get all the states that are transitioned FROM (keys of outer dict)
    possible_to_state = list(list(v.keys()) for s, v in transition_pr_dict.items() ) # all the states that are transitioned TO (keys of inner dicts)
    possible_to_state = set([item for sublist in possible_to_state for item in sublist]) # flatten this

    possible_states = list(set(possible_from_state).union(possible_to_state)) # take the set of the union

    possible_states.remove("END") # special termination state will handle END

    return possible_states


def get_vocab(word_emissions_dict):
    # Takes a word emissions dict and returns a set of all unique words in the vocabulary
    vocab = list(list(v.keys()) for s, v in word_emissions_dict.items() ) # all the states that are transitioned TO (keys of inner dicts)
    vocab = set([item for sublist in vocab for item in sublist]) # flatten

    return vocab


def calculate_word_emission_probability(this_word, this_possible_state, word_emissions_dict, vocab):
    # calculate Pr(word|this_possible_state) 
    # Handling of unknown words: assigns uniform probability
    # returns value of probability
    if this_word in vocab:
        try:
            emission_pr = word_emissions_dict[this_possible_state][this_word]
        except:
            emission_pr = 0
    else:
        emission_pr = 1 / len(possible_states) # assign uniform probability if the word is unknown

    return emission_pr


def compute_termination_probabilities(possible_states, transition_pr_dict):
    # computes list of Pr(END|possible_prior_state) from all possible_prior_state
        termination_probabilities = []
        for state in possible_states:
            try: 
                termination_probabilities.append(transition_pr_dict[state]["END"])
            except:
                termination_probabilities.append(0)

        return termination_probabilities

def backtrace_best_path(sentence, trellis, backtracer):
    # takes a sentence, Viterbi matrix, and backtrace array and returns a list of predicted POS tags for that sentence
    tags = []

    for col in reversed(range(len(sentence))):
        max_pr = max(trellis[:][col])
        best_state = backtracer[np.argmax(trellis[:][col])][col]

        tags.append(best_state)
        
    return list(reversed(tags))

def generate_pos_predictions(test_filepath, all_sentences_tags):
    # Takes a .words file of test data and a list of lists (each inner list = POS for a sentence)
    # writes new .pos file with predictions
    with open(test_filepath, "r") as f:
        data_raw = f.readlines()

    flattened_tags = [item for sublist in all_sentences_tags for item in sublist] # flatten the list

    tagged_lines = []
    i = 0

    for line in data_raw:
        if line != "\n":
            tagged_line = "{}\t{}\n".format(line.rstrip(), flattened_tags[i])
            i += 1
            tagged_lines.append(tagged_line)

        else:
            tagged_lines.append(line)

    with open("predicted_24.pos", "w") as f:
        f.writelines(tagged_lines)

######################################################
# Viterbi algorithm
def viterbi(training_filepath, test_filepath):

    # load training data and calculate transition and emission probabilities
    training_data = load_tagged_pos_file(training_filepath)
    training_emissions = convert_counts_probabilities(calculate_word_emission_counts(training_data)) 
    training_transitions = convert_counts_probabilities(calculate_transition_counts(group_words_sentences(training_data)))

    possible_states = get_possible_states(training_transitions) # unique possible states (excl START and END)
    vocab = get_vocab(training_emissions) # to check for unknown words

    # load test data as list of sentences (sentence = list of words)
    test_observations = group_words_sentences(load_test_words_file(test_filepath))

    # tag each sentence one at a time
    all_sentences_tags = []

    for sentence in test_observations:
        trellis = np.empty([len(possible_states), len(sentence)]) #setup the Viterbi matrix, cols = words, rows = possible states (minus start and end)
        backtracer = np.empty([len(possible_states), len(sentence)], dtype = "object") # setup the backward pointer

        for col in range(trellis.shape[1]):
            for row in range(trellis.shape[0]):
                
                this_word = sentence[col]
                this_possible_state = possible_states[row]

                # populate the initial state that transitioned from START
                if col == 0: 

                    # get Pr(state|START) for this_possible_state
                    try:
                        transition_pr = training_transitions["START"][this_possible_state]
                    except:
                        transition_pr = 0

                    # get Pr(emission|state)
                    emission_pr = calculate_word_emission_probability(this_word, this_possible_state, training_emissions, vocab)

                    trellis[row][col] = transition_pr * emission_pr # fill in the cell: initial state following START
                
                # populate cols 2-n
                else: 
                    # We will calculate Pr(prior_state) * Pr(this_possible_state|prior_state) for each possible prior_state,
                    # in order to select the path with the maximum likelihood
                    
                    prior_pr_for_each_path = list(trellis[:][col - 1]) # Pr(prior_state) is the previous column in Viterbi matrix

                    pr_transition_to_this_state = []

                    for possible_prior_state in possible_states: # get Pr(this_possible_state|previous_state) for all possible previous states
                        try:
                            transition_to_state_pr = training_transitions[possible_prior_state][this_possible_state]
                        except: 
                            transition_to_state_pr = 0

                        pr_transition_to_this_state.append(transition_to_state_pr)
                    
                    # find the max
                    path_probabilities = [prior * transition for prior, transition in zip(prior_pr_for_each_path, pr_transition_to_this_state) ]
                    max_path_probability = max(path_probabilities)

                    best_previous_state = possible_states[np.argmax(path_probabilities)] # get the previous state that had the highest probability
                    backtracer[row][col - 1] = best_previous_state # fill in the backtrace column

                    # get Pr(emission|this_possible_state)
                    emission_pr = calculate_word_emission_probability(this_word, this_possible_state, training_emissions, vocab)

                    trellis[row][col] = max_path_probability * emission_pr 

        # termination state

        termination_probabilities = compute_termination_probabilities(possible_states, training_transitions) # computes list of Pr(END|possible_prior_state) from all possible_prior_state
    
        final_probabilities = [transition_to_end * prior_pr for transition_to_end, prior_pr in zip(trellis[:][len(sentence) - 1], termination_probabilities) ]
        max_final_pr = max(final_probabilities)
        last_state = possible_states[np.argmax(max_final_pr)]
    
        # for the sentence, find the highest probability path in reverse
        all_sentences_tags.append(backtrace_best_path(sentence, trellis, backtracer))

    # finally generate pos predictions and write to file
    
    generate_pos_predictions(test_filepath, all_sentences_tags)
    
    

######################################################
# Run it
training_filepath = "/Users/lesliehuang/nlp-vc/HW4/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"

test_filepath = "/Users/lesliehuang/nlp-vc/HW4/WSJ_POS_CORPUS_FOR_STUDENTS/test.words"

viterbi(training_filepath, test_filepath)