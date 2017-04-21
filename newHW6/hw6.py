from provided import *
from collections import defaultdict
from math import *

class HMM:
    """
    Simple class to represent a Hidden Markov Model
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

# end of class


def compute_counts(training_data, order) :
    """Takes in training data in the form of a list of pairs (word, part-of-speech tag) and an int "order",
        returns counts of characteristics of the training data for further computing parameters for a HMM of order "order"
        :param training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
        :param order:  the order of the HMM
        :return: a tuple "result" in the order of (num_of_tokens, count_tag_word, count_tag, count_tag12, count_tag123)
        num_of_tokens: number of tokens in training_data
        count_tag_word: a dictionary that contains that contains C(ti,wi), for every unique tag and unique word (keys correspond to tags)
        count_tag: a dictionary that contains C(ti)
        count_tag12: a 2-d dictionary that contains C(ti-1,ti)
        count_tag123: if order = 2, this variable is omitted; if order = 3, this is the fifth element, a dictionary that contains C(ti-2,ti-1,ti)
"""
    # initialize variables
    num_of_tokens = 0
    count_tag_word = defaultdict(lambda: defaultdict(float))
    count_tag = defaultdict(float)
    count_tag12 = defaultdict(lambda: defaultdict(float))
    count_tag123 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    end = len(training_data)
    list_of_sentences = partition_sentences(training_data)

    # fill base cases and inductive cases based on order
    for i in range(end):
        pair = training_data[i]
        if pair:
            num_of_tokens += 1
            word2, tag2 = decompose_a_pair(training_data, i)
            # if tag2 and word2:
            count_tag_word[tag2][word2] += 1
            count_tag[tag2] += 1

            if i >= 1:
                word1, tag1 = decompose_a_pair(training_data, i - 1)
                    # exclude sequence ". word", ". word word" and "word. word"?
                    # if word1 and tag1: # word1 != "." and
                        # add_to_dict_cell(count_tag12, tag1, tag2)
                count_tag12[tag1][tag2] += 1
                if order == 3 and i >= order - 1:
                    word0, tag0 = decompose_a_pair(training_data, i - 2)
                            # if word0 and tag0: # word0 != "." and
            # add_to_dict_cell(count_tag123, tag0, tag1, tag2)
            # if order == 3:
                    count_tag123[tag0][tag1][tag2] += 1

    return num_of_tokens, count_tag_word, count_tag, count_tag12, count_tag123
# end of function


def compute_initial_distribution(training_data, order):
    """Takes in training data in the form of a list of pairs (word, part-of-speech tag) and an int "order",
        returns initial distribution of the training data for a HMM of order "order"
    :param training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
    :param order:  the order of the HMM
    :return: a dictionary of, if order == 2: {tag: initial probability}; if order == 2: {tag1: {tag2: initial probability}}
    """
    # initialize matrix for initial distribution

    count_sentence = 1
    # fill base cases and inductive cases based on order
    end = len(training_data)
    word, tag = decompose_a_pair(training_data, 0)
    if order == 2:
        init_distribution = defaultdict(float)
        # add_to_dict_cell(init_distribution, tag)
        init_distribution[tag] += 1
    elif order == 3:
        init_distribution = defaultdict(lambda: defaultdict(float))
        word1, tag1 = decompose_a_pair(training_data, 1)
        # add_to_dict_cell(init_distribution, tag, tag1)
        init_distribution[tag][tag1] += 1

    else:
        raise Exception("Invalid order!")

    for i in range(order - 1, end):
        # repetitive code, should modularize
        if training_data[i] :
            word, tag = decompose_a_pair(training_data, i)
            # exclude sequence ". word", ". word word" and "word. word"?
            if word == ".":
                next_sentence = i + 1
                next_sentence = find_first_of_next_sentence(training_data, next_sentence, end)
                if next_sentence >= 0:
                    count_sentence += 1
                    word, tag = decompose_a_pair(training_data, next_sentence)
                    # if word and tag:
                    if order == 2:
                            # add_to_dict_cell(init_distribution, tag)
                        init_distribution[tag] += 1
                    elif next_sentence >= order - 1:
                        word1, tag1 = decompose_a_pair(training_data, next_sentence + 1)
                        # if word1 and tag1:
                                # add_to_dict_cell(init_distribution, tag, tag1)
                        init_distribution[tag][tag1] += 1
        else:
            print "training data at line {0} is empty".format(i)
    # print count_sentence
    if order == 3:
        for key1, key2_dict in init_distribution.items():
            for key2, value in key2_dict.items():
                init_distribution[key1][key2] = value/count_sentence
        return init_distribution
    else:
        for key1, value in init_distribution.items():
            init_distribution[key1] = value / count_sentence
        return init_distribution
# end of function


def compute_emission_probabilities(unique_words, unique_tags, W, C):
    """
    :param unique_words: a list of unique words found in the file.
    :param unique_tags: a list of unique POS tags found in the file.
    :param W: count_tag_word from compute_counts()
    :param C: count_tag from compute_counts()
    :return: e_dict, emission matrix as a dictionary whose keys are the tags
    """
    # todo: add value to account for unknown words
    e_dict = defaultdict(lambda: defaultdict(float))
    for tag, word_dict in W.items():
        for word in word_dict:
            e_dict[tag][word] = (float(W[tag][word])/C[tag])

    return e_dict
# end of function


def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    # todo: test
    """
    :param unique_tags: a list of unique POS tags found in the file. not used
    :param num_tokens: number of tokens in training_data
    :param C1: count_tag from compute_counts()
    :param C2: count_tag12 from compute_counts(), a 2-d dictionary that contains C(ti-1,ti)
    :param C3: count_tag123 from computer_counts(), if order = 3, this is a dictionary that contains C(ti-2,ti-1,ti)
    :param order:  the order of the HMM
    :return: lambdas, a list that contains lambda0, lambda1, lambda2, respectively
    """
    lambdas = [0.0, 0.0, 0.0]

    if order == 2:
        for tag_i_1, tag_i_dict in C2.items():
            for tag_i in tag_i_dict:
                if tag_i_dict[tag_i] > 0:
                    if num_tokens == 0:
                        alpha0 = 0.0
                    else:
                        alpha0 = float((C1[tag_i] - 1))/num_tokens

                    if (C1[tag_i_1] - 1) == 0:
                        alpha1 = 0.0
                    else:
                        alpha1 = float((C2[tag_i_1][tag_i] - 1)) / (C1[tag_i_1] - 1)

                    if alpha0 == max(alpha0, alpha1):
                        lambdas[0] += C2[tag_i_1][tag_i]
                    else:
                        lambdas[1] += C2[tag_i_1][tag_i]

    if order == 3:
        for tag_i_2, tag_i_1_dict in C3.items():
            for tag_i_1, tag_i_dict in tag_i_1_dict.items():
                for tag_i in tag_i_dict:
                    # print tag_i_2, tag_i_1, tag_i
                    if tag_i_dict[tag_i] > 0:
                        if num_tokens == 0:
                            alpha0 = 0.0
                        else:
                            alpha0 = float((C1[tag_i] - 1))/num_tokens

                        if (C1[tag_i_1] - 1) == 0:
                            alpha1 = 0.0
                        else:
                            alpha1 = float((C2[tag_i_1][tag_i] - 1)) / (C1[tag_i_1] - 1)

                        if (C2[tag_i_2][tag_i_1] - 1) == 0:
                            alpha2 = 0.0
                        else:
                            alpha2 = float((C3[tag_i_2][tag_i_1][tag_i] - 1)) / (C2[tag_i_2][tag_i_1] - 1)

                        # print "alpha: ", alpha0, alpha1, alpha2
                        if alpha0 == max(alpha1, alpha0, alpha2):
                            lambdas[0] += C3[tag_i_2][tag_i_1][tag_i]
                        elif alpha1 == max(alpha1, alpha0, alpha2):
                            lambdas[1] += C3[tag_i_2][tag_i_1][tag_i]
                        else:
                            lambdas[2] += C3[tag_i_2][tag_i_1][tag_i]
    # print "lambdas: ", lambdas
    # sum_lambdas = reduce((lambda x, y: x + y), lambdas)
    sum_lambdas = lambdas[0] + lambdas[1] + lambdas[2]
    # result = list(map(lambda x: x/sum_lambdas, lambdas))
    result = [lambdas[0]/sum_lambdas, lambdas[1]/sum_lambdas, lambdas[2]/sum_lambdas]
    print result
    return result
# end of function


def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
    """

    :param training_data:
    :param unique_tags:
    :param unique_words:
    :param order:
    :param use_smoothing:
    :return:
    """
    # todo: test
    init_distribution = compute_initial_distribution(training_data, order)
    num_of_tokens, count_tag_word, count_tags, count_tag2, count_tag3 = compute_counts(training_data, order)
    emis_prob = compute_emission_probabilities(unique_words, unique_tags, count_tag_word, count_tags)

    # parameters: result_matrix, order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2=None
    transit_prob = compute_transition_matrix(order, use_smoothing, count_tags, num_of_tokens, count_tag2, count_tag3)

    # parameters: order, initial_distribution, emission_matrix, transition_matrix
    hmm = HMM(order, init_distribution, emis_prob, transit_prob)
    return hmm
# end of function


def bigram_viterbi(hmm, sentence):
    """

    :param hmm:
    :param sentence: a list of words with a period at the end
    :return:
    """
    # check input
    if not sentence:
        return []

    # initialize variables
    emission = hmm.emission_matrix
    init_distribution = hmm.initial_distribution
    transit_matrix = hmm.transition_matrix
    all_states = emission.keys()

    # print "emission ", emission
    # print "initial distribution ", init_distribution
    # print "transit ", transit_matrix
    v_matrix = defaultdict(lambda: defaultdict(float))
    bp_matrix = defaultdict(lambda: defaultdict(str))
    length = len(sentence)

    z_list = [None] * length
    # fill base cases
    for state in all_states:
        word = sentence[0]
        v_matrix[state][0] = log_wrapper(init_distribution[state]) + log_wrapper(emission[state][word])
        # print state, word, v_matrix[state][0]

    # inductive cases
    for i in range(1, length):

        for curr_state in all_states:
            max_prob = float("-inf")
            arg_max = all_states[0]
            # loop through all states for the previous observation to find max(v*transition)
            for prior_state in emission:
                temp_prob = v_matrix[prior_state][i-1] + log_wrapper(transit_matrix[prior_state][curr_state])
                if temp_prob > max_prob:
                    max_prob = temp_prob
                    arg_max = prior_state
                    # print "bp, curr_state", bp_matrix[curr_state][i]
            v_matrix[curr_state][i]= max_prob + log_wrapper(emission[curr_state][sentence[i]])
            bp_matrix[curr_state][i] = arg_max
                # print curr_state, prior_state, max_prob, v_matrix[curr_state][i], "bp: ", bp_matrix[curr_state][i]

    # get the last state for z_list
    max_prob = float("-inf")
    arg_max_last = all_states[0]
    for last_state in emission:
        temp_prob = v_matrix[last_state][length-1]
        if temp_prob > max_prob:
            max_prob = temp_prob
            arg_max_last = last_state

    z_list[length-1] = arg_max_last
        # print "bp[{0}][{1}] is {2}".format(curr_state, i, bp_matrix[curr_state][i])

    # print bp_matrix
    for i in range(length-2, -1, -1):
        z_list[i] = bp_matrix[z_list[i+1]][i+1]

    list_pairs = []
    for i in range(length):
        list_pairs.append((sentence[i], z_list[i]))
    return list_pairs
# end of function


def trigram_viterbi(hmm, sentence):
    # check input
    if not sentence:
        return []

    # initialize variables
    emission = hmm.emission_matrix
    init_distribution = hmm.initial_distribution
    transit_matrix = hmm.transition_matrix
    all_states = emission.keys()

    v_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    bp_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    # v_matrix = {}
    # bp_matrix = {}
    length = len(sentence)

    # base cases for v
    word0 = sentence[0]
    word1 = sentence[1]
    for state0 in all_states:
        for state1 in all_states:
            # print state0, state1
            if init_distribution[state0][state1] != 0 and emission[state0][word0] != 0 and emission[state1][word1] != 0:
                v_matrix[state0][state1][1] = log_wrapper(init_distribution[state0][state1])\
                                          + log_wrapper(emission[state0][word0]) + log_wrapper(emission[state1][word1])
            else:
                v_matrix[state0][state1][1] = float('-inf')
            # print "base case v: ", state0, state1, v_matrix[state0][state1][1]

    for i in range(2, length):
        for curr_state in all_states: # wrong result: curr_state; correct result: prior_state1, curr_state
            for prior_state1 in all_states: # wrong result: prior_state0; correct result: curr_state, prior_state1
                max_prob = float('-inf')
                arg_max0 = prior_state1
                for prior_state0 in all_states: # wrong result: prior_state1; correct result: prior_state0
                    # arg_max1 = prior_state1
                    if transit_matrix[prior_state0][prior_state1][curr_state] != 0:
                        temp = v_matrix[prior_state0][prior_state1][i-1] + \
                           log_wrapper(transit_matrix[prior_state0][prior_state1][curr_state])
                    else:
                        # print "should be -inf:"
                        temp = float('-inf')
                    if temp > max_prob:
                        max_prob = temp
                        arg_max0 = prior_state0
                        # arg_max1 = prior_state1
                        # print "arg_max ", arg_max0
                if emission[curr_state][sentence[i]] != 0:
                    v_matrix[prior_state1][curr_state][i] = max_prob + log_wrapper(emission[curr_state][sentence[i]])
                else:
                    v_matrix[prior_state1][curr_state][i] = float('-inf')
                bp_matrix[prior_state1][curr_state][i] = arg_max0
            # print "inductive v: ", prior_state1, curr_state, v_matrix[prior_state1][curr_state][i]
            # print "inductive bp: ", prior_state1, curr_state, bp_matrix[prior_state1][curr_state][i]

    max_prob = float('-inf')
    arg_max_last = all_states[0]
    arg_max_second_last = all_states[0]
    # print "last_state0, last_state1 ", arg_max_second_last, arg_max_last
    for prior_state1  in v_matrix:
        for last_state in v_matrix:
            if v_matrix[prior_state1][last_state][length - 1] != 0:
                temp = v_matrix[prior_state1][last_state][length - 1]
                # print "in if, temp = ", temp
                if temp > max_prob:
                    max_prob = temp
                    arg_max_last = last_state
                    arg_max_second_last = prior_state1
            else:
                v_matrix[prior_state1][last_state][length - 1] = float('-inf')

                # print "max, last, sec last: ", max_prob, arg_max_last, arg_max_second_last

    # for state0 in v_matrix:
    #     for state1 in v_matrix[state0]:
    #         if v_matrix[state0][state1][length-1] != 0:
    #             print "last column in v_matrix: ", state0, state1, v_matrix[state0][state1][length-1]
    #         else:
    #             print "zero column in v_matrix: ", state0, state1, v_matrix[state0][state1][length-1]

    z_list = [None] * length
    z_list[length - 1] = arg_max_last
    z_list[length - 2] = arg_max_second_last

    for i in range(length-3, -1, -1):
        z_list[i] = bp_matrix[z_list[i+1]][z_list[i+2]][i+2]
        # print "last for loop ", i, bp_matrix[z_list[i+1]][z_list[i+2]][i+2], bp_matrix[z_list[i+1]]

    list_pairs = []
    for i in range(length):
        list_pairs.append((sentence[i], z_list[i]))
    return list_pairs


# def old_trigram_viterbi(hmm, sentence):
#     # check input
#     if not sentence:
#         return []
#
#     # print sentence
#     # initialize variables
#     emission = hmm.emission_matrix
#     init_distribution = hmm.initial_distribution
#     transit_matrix = hmm.transition_matrix
#
#     v_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#     bp_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
#     length = len(sentence)
#
#     z_list = [None] * length
#     # fill base cases, i = 1, initial 2 tags
#     for state0, state1_dict in init_distribution.items():
#         for state1 in state1_dict:
#             word0 = sentence[0]
#             word1 = sentence[1]
#             v_matrix[state0][state1][1] = log_wrapper(init_distribution[state0][state1])\
#                                           + log_wrapper(emission[state0][word0]) + log_wrapper(emission[state1][word1])
#             # print state1, word1, v_matrix[state0][state1][1]
#     # print v_matrix
#     # inductive cases
#     for i in range(2, length):
#         # print "v ", v_matrix
#         for curr_state in emission:
#             # loop through all 2-state combinations for the previous 2 observations to find max(v*transition)
#             # todo: bp seems wrong
#             max_prob = float("-inf")
#             state_0 = None
#             state_1 = None
#             for prior_state_0, prior_state_1_dict in transit_matrix.items():
#                 for prior_state_1 in prior_state_1_dict:
#                     if v_matrix[prior_state_0][prior_state_1][i-1] == 0:
#                         v_matrix[prior_state_0][prior_state_1][i-1] = float("-inf")
#
#                     temp_prob = v_matrix[prior_state_0][prior_state_1][i-1]\
#                                 + log_wrapper(transit_matrix[prior_state_0][prior_state_1][curr_state])
#                     # if temp_prob == 0:
#                         # print "v_matrix[{0}][{1}][{2}] is {3}".\
#                         #     format(prior_state_0, prior_state_1, i-1, v_matrix[prior_state_0][prior_state_1][i-1])
#                     if temp_prob > max_prob:
#                         # print "temp_prob and max_prob: ", temp_prob, max_prob
#                         max_prob = temp_prob
#                         state_0 = prior_state_0
#                         state_1 = prior_state_1
#                         # print prior_state_0, prior_state_1, curr_state, max_prob
#                         # print "bp[{0}][{1}]: {2}".format(curr_state, i, bp_matrix[curr_state][i])
#                         # print "bp, curr_state", bp_matrix[curr_state][i], curr_state
#                     # if prior_state_0 == "MD" and prior_state_1 == "verb":
#                     #     print "MD, verb ", v_matrix[prior_state_0][prior_state_1][i-1]
#             if state_0 and state_1 :
#                 # print "prior_state_0 ", state_0, ", prior_state_1", state_1
#                 v_matrix[state_1][curr_state][i]= max_prob\
#                                          + log_wrapper(emission[curr_state][sentence[i]])
#                 bp_matrix[state_1][curr_state][i] = state_0
#                     # print "bp[prior_state_1][curr_state][i] ", bp_matrix[prior_state_1][curr_state][i]
#                     # print prior_state_0, prior_state_1, curr_state, max_prob, v_matrix[curr_state][i], "bp: ", bp_matrix[curr_state][i]
#
#     # get the last state for z_list
#     # print v_matrix
#     for last_state in emission:
#         max_prob = float("-inf")
#         for prior_state_0, prior_state_1_dict in transit_matrix.items():
#             for prior_state_1 in prior_state_1_dict:
#             # print each_state, last_state
#                 temp_prob = v_matrix[prior_state_1][last_state][length-1]
#                 if temp_prob > max_prob:
#                     max_prob = temp_prob
#                     z_list[length-1] = last_state
#                     z_list[length-2] = prior_state_1
#                     # print "prior_state is {0}, last_state in if is {1}".format(prior_state_1, last_state)
#
#     # print z_list
#     # print bp_matrix
#     # print z_list
#     for i in range(length-3, -1, -1):
#         # print i, i+1, i+2
#         # print z_list[i+1], z_list[i+2], bp_matrix[z_list[i+1]]
#         z_list[i] = bp_matrix[z_list[i+1]][z_list[i+2]][i]
#         # print "bp[{0}][{1}][{2}] is {3}".format(z_list[i+1], z_list[i+2], i, bp_matrix[z_list[i+1]][z_list[i+2]][i])
#     # print len(bp_matrix)
#     # print length
#     return z_list
# # end of function


################################################# Helper functions ##########################################
def partition_sentences(list_of_pairs):

    if list_of_pairs[len(list_of_pairs)-1] != ".":
        list_of_pairs.append(".")

    untagged_sentences = []
    start = 0
    # run tagging
    while start < (len(list_of_pairs)):
        # print start
        end = list_of_pairs[start:].index(".")
        sentence = list_of_pairs[start:start + end]
        sentence.append(".")
        # print sentence
        untagged_sentences.append(sentence)
        start += end+1

    return untagged_sentences


def check_accuracy(hmm, list_of_words, word_tag_in_test, order=2):

    list_of_sentences = partition_sentences(list_of_words)
    corrects = 0
    # check accuracy
    total_tagges_from_hmm = []
    for sentence in list_of_sentences:
        if order == 2:
            total_tagges_from_hmm.extend(bigram_viterbi(hmm, sentence))
        elif order == 3:
            total_tagges_from_hmm.extend(trigram_viterbi(hmm, sentence))
        else:
            raise  Exception("Invalid order: ", order)

    print len(total_tagges_from_hmm), len(word_tag_in_test)
    for i in range(0, len(word_tag_in_test)):
        # print i
        # print tags_from_hmm[i], pairs_from_file[i][1]
        if total_tagges_from_hmm[i] == word_tag_in_test[i]:
            corrects += 1
    print total_tagges_from_hmm
    return corrects * 100.0 / len(word_tag_in_test)


def log_wrapper(number):
    if number == 0:
        return float("-inf")
    else:
        return log(number)

def update_hmm(emission_matrix, sentence, words_in_training, epsilon=0.00001):
    for input_word in sentence:
        if input_word not in words_in_training:
            for state in emission_matrix:
                for old_word in emission_matrix[state]:
                    # print state, word, emission_matrix[state][word], epsilon
                    emission_matrix[state][old_word] += epsilon
                emission_matrix[state][input_word] += epsilon

    for state in emission_matrix:
        total = 0
        # get sum
        # sum_all = reduce((lambda x, y: x + y), emission_matrix[state].values())
        for word in emission_matrix[state]:
            total += emission_matrix[state][word]
        # normalize
        for word in emission_matrix[state]:
            emission_matrix[state][word] = emission_matrix[state][word]/total
        # print "sum: ", reduce((lambda x, y: x + y), emission_matrix[state].values())


def read_untagged_data(filename):
    f = open(str(filename), "r")
    words = []
    sentences = []
    for line in f: # line = paragraph
        if len(line.split(".")) < 1:
            continue
        else:
            words.extend(line.replace("\t", "").strip().split(" "))
    f.close()
    return words


def compute_transition_matrix(order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2):
    """

    :param result_matrix:
    :param order:
    :param smoothing:
    :param count_tags:
    :param num_tokens:
    :param tag_sequence1:
    :param tag_sequence2:
    :return:
    """

    if smoothing:
        if order == 2:
            # this variable can be seen from outside of the out-most the if statement
            lambdas = compute_lambdas(count_tags, num_tokens, count_tags, tag_sequence1, tag_sequence2, order)
        elif order == 3:
            lambdas = compute_lambdas(count_tags, num_tokens, count_tags, tag_sequence1, tag_sequence2, order)
        else:
            raise Exception("Invalid order")
    else:
        lambdas = [0, 0, 0]
        if order == 2:
            lambdas[1] = 1
        else:
            lambdas[2] = 1

    if order == 2:
        result_matrix = defaultdict(lambda: defaultdict(float))

        for tag_i_1, tag_i_dict in tag_sequence1.items():
            for tag_i in tag_i_dict:
                result_matrix[tag_i_1][tag_i] = lambdas[1] * float(tag_sequence1[tag_i_1][tag_i])/count_tags[tag_i_1]
                + lambdas[0] * float(count_tags[tag_i])/num_tokens
        return result_matrix

    elif order == 3:
        result_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for tag_i_2, tag_i_1_dict in tag_sequence2.items():
            for tag_i_1, tag_i_dict in tag_i_1_dict.items():
                for tag_i in tag_i_dict:
                    # print "in transition ", tag_i_2, tag_i_1, tag_i
                    result_matrix[tag_i_2][tag_i_1][tag_i] = \
                        lambdas[2] * float(tag_sequence2[tag_i_2][tag_i_1][tag_i])/tag_sequence1[tag_i_2][tag_i_1]
                    + lambdas[1] * float(tag_sequence1[tag_i_1][tag_i]) / count_tags[tag_i_1]
                    + lambdas[0] * float(count_tags[tag_i])/num_tokens
        return result_matrix
# end of function


def find_first_of_next_sentence(training_data, next_sentence, boundary):
    # caution: what if next_sentence is out of boundary
    while next_sentence < boundary:
        if training_data[next_sentence] and training_data[next_sentence][0] != ".":
            break
        else:
            next_sentence += 1

    if next_sentence >= boundary:
        # print "in find next", next_sentence
        next_sentence = -1
    return next_sentence
# end of function


def decompose_a_pair(dictionary, index):
    word = dictionary[index][0]
    tag = dictionary[index][1]
    return word, tag
# end of function


# def experiment(filename, percentage, smoothing, order):

################################################# testing functions ########################################
def test_trigram_viterbi(hmm, sentence):
    return trigram_viterbi(hmm, sentence)

def test_partition_sentences(list_of_words):
    return partition_sentences(list_of_words)
# Input: "test_viterbi_tagged.txt"
# Expected output: [['The', '/', 'DT', 'can', '/', 'NN', 'can', '/', 'MD', 'run', '/', 'verb', '.'], ['/', '.'], ['', 'The', '/', 'DT', 'sun', '/', 'NN', 'rises', '/', 'verb', '.']]


def test_update_hmm(emission_matrix, sentence, epsilon=0.00001):
    update_hmm(emission_matrix, sentence, epsilon=0.00001)
    print emission_matrix


def test_build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
    return build_hmm(training_data, unique_tags, unique_words, order, use_smoothing)


def test_compute_transition_matrix(order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2=None):
    return compute_transition_matrix(order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2)
# Input: "test_viterbi_tagged.txt", order = 3
# Expected output: defaultdict(<function <lambda> at 0x02B99CF0>, {'MD': defaultdict(<function <lambda> at 0x02B99D30>, {'verb': defaultdict(<type 'float'>, {'.': 1.0})}), 'DT': defaultdict(<function <lambda> at 0x02B99D70>, {'NN': defaultdict(<type 'float'>, {'MD': 0.5, 'verb': 0.5})}), 'NN': defaultdict(<function <lambda> at 0x02B99DB0>, {'MD': defaultdict(<type 'float'>, {'verb': 1.0}), 'verb': defaultdict(<type 'float'>, {'.': 1.0})})})
# Input: "test_viterbi_tagged.txt", order = 2
# Expected output: defaultdict(<function <lambda> at 0x02C8AC30>, {'MD': defaultdict(<type 'float'>, {'verb': 1.0}), 'DT': defaultdict(<type 'float'>, {'NN': 1.0}), 'verb': defaultdict(<type 'float'>, {'.': 1.0}), 'NN': defaultdict(<type 'float'>, {'MD': 0.5, 'verb': 0.5})})


def test_compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    return compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
# Input: "test_viterbi_tagged.txt", order = 3
# Expected output: [0.6, 0.4, 0.0]
# order = 2
# Expected output: [0.42857142857142855, 0.5714285714285714]
# Input : "training.txt", order = 3 todo
# Expected output: [0.11725640453016421, 0.39507145854342945, 0.4876721369264064]
# order = 2
# Expected output: [0.32179275790165707, 0.6782072420983429]


def test_compute_emission_probabilities(unique_words, unique_tags, W, C):
    print compute_emission_probabilities(unique_words, unique_tags, W, C)
# test input: "test_training_data1.txt"
# expect output:
# defaultdict(<function <lambda> at 0x02B86770>, {':': defaultdict(<type 'int'>, {'--': 0.0}),
# 'NN': defaultdict(<type 'int'>, {'Test': -0.22184874961635637, 'text': -0.6989700043360187,'can': -0.6989700043360187}),
# 'VBD': defaultdict(<type 'int'>, {'sucked': 0.0}),
# 'CC': defaultdict(<type 'int'>, {'and': 0.0}),
# '.': defaultdict(<type 'int'>, {'.': 0.0}),
# 'verb': defaultdict(<type 'int'>, {'can': 0.0}), 'CD': defaultdict(<type 'int'>, {'2': 0.0})})
# end of function


def test_compute_initial_distribution(training_data, order):
    return compute_initial_distribution(training_data, order)
# test input: "test_training_data.txt"
# expect output:
# defaultdict(<function <lambda> at 0x02C24530>, {'DT': defaultdict(<type 'int'>, {'JJ': 0.25}),
# 'NN': defaultdict(<type 'int'>, {'VBD': 0.25, 'NN': 0.25, 'CD': 0.25})})

# end of function


def test_compute_counts(training_data, order, print_mode=True):
    num_of_tokens, count_tag_word, count_tags, count_tag2, count_tag3 = compute_counts(training_data, order)
    if print_mode:
        print "number of tokens", num_of_tokens
        for tag1 in count_tag2:
            for tag2 in count_tag2[tag1]:
                print "count: tag1 {0}, tag2 {1} is {2}".format(tag1, tag2, count_tag2[tag1][tag2])
        if order == 3:
            for tag1 in count_tag3:
                for tag2 in count_tag3[tag1]:
                    for tag3 in count_tag3[tag1][tag2]:
                        print "count: tag1 {0}, tag2 {1}, tag3 {2} is {3}"\
                            .format(tag1, tag2, tag3, count_tag3[tag1][tag2][tag3])
    # print count_tag_word, count_tags, count_tag2
    return num_of_tokens, count_tag_word, count_tags, count_tag2, count_tag3
    # test input: "test_training_data.txt"
    # expect output:
    # number of tokens 61
    # count: tag1 NNPS, tag2 NNP, tag3 NNP is 1
    # count: tag1 NN, tag2 NN, tag3 : is 1
    # count: tag1 NN, tag2 VBD, tag3 DT is 1
    # count: tag1 NN, tag2 VBD, tag3 . is 1
    # count: tag1 NN, tag2 CC, tag3 verb is 1
    # count: tag1 NN, tag2 CD, tag3 . is 1
    # count: tag1 NN, tag2 IN, tag3 DT is 1
    # count: tag1 NN, tag2 :, tag3 NN is 1
    # count: tag1 NN, tag2 NNS, tag3 CC is 1
    # count: tag1 NN, tag2 NNS, tag3 IN is 1
    # count: tag1 VBD, tag2 DT, tag3 NN is 1
    # count: tag1 CC, tag2 DT, tag3 NNP is 1
    # count: tag1 CC, tag2 verb, tag3 . is 1
    # count: tag1 CC, tag2 NN, tag3 NNS is 1
    # count: tag1 CC, tag2 NNP, tag3 NNP is 1
    # count: tag1 JJS, tag2 NNS, tag3 IN is 1
    # count: tag1 ,, tag2 CC, tag3 DT is 1
    # count: tag1 ,, tag2 DT, tag3 IN is 1
    # count: tag1 ,, tag2 WDT, tag3 VBP is 1
    # count: tag1 CD, tag2 ,, tag3 CC is 1
    # count: tag1 CD, tag2 ,, tag3 WDT is 1
    # count: tag1 VBP, tag2 NN, tag3 NNS is 1
    # count: tag1 WDT, tag2 VBP, tag3 NN is 1
    # count: tag1 JJ, tag2 NNS, tag3 IN is 1
    # count: tag1 JJ, tag2 JJ, tag3 NNS is 1
    # count: tag1 IN, tag2 NNS, tag3 . is 1
    # count: tag1 IN, tag2 DT, tag3 NNP is 1
    # count: tag1 IN, tag2 CD, tag3 , is 2
    # count: tag1 IN, tag2 JJS, tag3 NNS is 1
    # count: tag1 IN, tag2 NNP, tag3 NNP is 1
    # count: tag1 DT, tag2 IN, tag3 CD is 1
    # count: tag1 DT, tag2 NN, tag3 IN is 1
    # count: tag1 DT, tag2 JJ, tag3 JJ is 1
    # count: tag1 DT, tag2 NNP, tag3 NNPS is 1
    # count: tag1 DT, tag2 NNP, tag3 NNP is 1
    # count: tag1 :, tag2 NN, tag3 CC is 1
    # count: tag1 NNS, tag2 CC, tag3 NN is 1
    # count: tag1 NNS, tag2 IN, tag3 NNS is 1
    # count: tag1 NNS, tag2 IN, tag3 JJS is 1
    # count: tag1 NNS, tag2 IN, tag3 NNP is 1
    # count: tag1 NNP, tag2 NNPS, tag3 NNP is 1
    # count: tag1 NNP, tag2 NN, tag3 VBD is 1
    # count: tag1 NNP, tag2 CC, tag3 NNP is 1
    # count: tag1 NNP, tag2 ,, tag3 DT is 1
    # count: tag1 NNP, tag2 IN, tag3 CD is 1
    # count: tag1 NNP, tag2 NNP, tag3 CC is 1
    # count: tag1 NNP, tag2 NNP, tag3 IN is 1
    # count: tag1 NNP, tag2 NNP, tag3 , is 1
    # count: tag1 NNP, tag2 NNP, tag3 NN is 1
    # count: tag1 NNP, tag2 NNP, tag3 NNP is 3
# end of function


def test_decompose_a_pair(training_data, index):
    print decompose_a_pair(training_data, index)
# end of function


def test_find_first_of_next_sentence(training_data, next_sentence, boundary):
    print find_first_of_next_sentence(training_data, next_sentence, boundary)
# end of function

# initialize training data
# tuple_of_3_returns = read_pos_file("mytest_tagged.txt")
tuple_of_3_returns = read_pos_file("training.txt")

training_data = tuple_of_3_returns[0]
unique_words = tuple_of_3_returns[1]
unique_tags = tuple_of_3_returns[2]
# order = 3
# smoothing = False
# num_of_tokens, count_tag_word, count_tags, count_tag12, count_tag123 = test_compute_counts(training_data, order, False)
# print "unique_tags", unique_tags
# print "num_tokens", num_of_tokens
# print "Cti", count_tags
# print "Cti-1,ti", count_tag12
# print "Cti-2,ti-1,ti", count_tag123
# lambdas = test_compute_lambdas(unique_tags, num_of_tokens, count_tags, count_tag12, count_tag123, order)
# print "lambdas", lambdas
# print reduce((lambda x, y: x + y), lambdas)
# print test_compute_transition_matrix(order, smoothing, count_tags, num_of_tokens, count_tag12, count_tag123)

# test_update_hmm(hmm.emission_matrix, words)
# print hmm.emission_matrix
# print hmm.transition_matrix
# print hmm.order
# print hmm.initial_distribution
# test_compute_emission_probabilities(unique_words, unique_tags, count_tag_word, count_tags)
# print test_compute_initial_distribution(training_data, order)
testdata_tagged = read_pos_file("testdata_tagged.txt")
# testdata_tagged = read_pos_file("mytest_tagged.txt")
answers = testdata_tagged[0]
# unique_tags = testdata_tagged[1]
# unique_words = testdata_tagged[2]
#
untagged_words = read_untagged_data("testdata_untagged.txt")
# untagged_words = read_untagged_data("mytest_untagged.txt")
# print test_partition_sentences(untagged_words)

# sentence = ['The', 'cat', 'ran'] #, '.']#, 'away', 'today', '.']
# training_data = [('The','DT'),('cat','NN'),('ran','VBD')] #, ('.', '.')] #,('away','RB'),('today','NN'),('.','.')]
# unique_tags = ['DT','NN','VBD'] #,'.'] #,'RB']
# unique_words = ['The', 'cat', 'ran'] #, '.']
order = 3
smoothing = True

trained_hmm = build_hmm(training_data, unique_tags, unique_words, order, smoothing)
# print trained_hmm.emission_matrix

update_hmm(trained_hmm.emission_matrix, untagged_words, list(unique_words))
# print trained_hmm.emission_matrix
# for state0 in trained_hmm.transition_matrix:
#     for state1 in trained_hmm.transition_matrix[state0]:
#         print state0, state1, trained_hmm.transition_matrix[state0][state1]
# print trained_hmm.order
# print trained_hmm.initial_distribution
sentences = partition_sentences(untagged_words)
# print test_trigram_viterbi(trained_hmm, sentences[0])
print check_accuracy(trained_hmm, untagged_words, answers, order)