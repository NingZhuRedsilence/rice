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

# provided code
def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of tuples, where each tuple
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	f = open(str(filename), "r")
	# count = 0 #
	for line in f:
		# count += 1 #
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		# if not word or not tag: #
			# print count #
		file_representation.append( (word, tag) )
		unique_words.add(word)
		unique_tags.add(tag)
  	f.close()
	return file_representation, unique_words, unique_tags


def compute_counts(training_data, order):
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

    # fill base cases and inductive cases based on order
    for i in range(end):
        pair = training_data[i]
        if pair:
            num_of_tokens += 1
            word2, tag2 = decompose_a_pair(training_data, i)
            count_tag_word[tag2][word2] += 1
            count_tag[tag2] += 1

            if i >= 1:
                word1, tag1 = decompose_a_pair(training_data, i - 1)
                count_tag12[tag1][tag2] += 1
                if order == 3 and i >= order - 1:
                    word0, tag0 = decompose_a_pair(training_data, i - 2)
                    count_tag123[tag0][tag1][tag2] += 1

    return num_of_tokens, count_tag_word, count_tag, count_tag12, count_tag123
# end of function


def compute_initial_distribution(training_data, order):
    """Takes in training data in the form of a list of pairs (word, part-of-speech tag) and an int "order",
        returns initial distribution of the training data for a HMM of order "order"
    :param training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
    :param order:  the order of the HMM
    :return: a dictionary of, if order == 2: {tag: initial probability}; if order == 3: {tag1: {tag2: initial probability}}
    """
    # initialize matrix for initial distribution

    count_sentence = 1
    # fill base cases and inductive cases based on order
    end = len(training_data)
    word, tag = decompose_a_pair(training_data, 0)
    if order == 2:
        init_distribution = defaultdict(float)
        init_distribution[tag] += 1
    elif order == 3:
        init_distribution = defaultdict(lambda: defaultdict(float))
        word1, tag1 = decompose_a_pair(training_data, 1)
        init_distribution[tag][tag1] += 1

    else:
        raise Exception("Invalid order!")

    for i in range(order - 1, end):
        # repetitive code, should modularize
        if training_data[i] :
            word, tag = decompose_a_pair(training_data, i)
            if word == ".":
                next_sentence = i + 1
                next_sentence = find_first_of_next_sentence(training_data, next_sentence, end)
                if next_sentence >= 0:
                    count_sentence += 1
                    word, tag = decompose_a_pair(training_data, next_sentence)
                    if order == 2:
                        init_distribution[tag] += 1
                    elif next_sentence >= order - 1:
                        word1, tag1 = decompose_a_pair(training_data, next_sentence + 1)
                        init_distribution[tag][tag1] += 1
        else:
            print "training data at line {0} is empty".format(i)
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
    """ Takes in a list of unique words, tags, and counts of pairs of tags and words and tags and
    return a matrix of emission probability
    :param unique_words: a list of unique words found in the file.
    :param unique_tags: a list of unique POS tags found in the file.
    :param W: count_tag_word from compute_counts()
    :param C: count_tag from compute_counts()
    :return: e_dict, emission matrix as a dictionary whose keys are the tags
    """
    e_dict = defaultdict(lambda: defaultdict(float))
    for tag, word_dict in W.items():
        for word in word_dict:
            e_dict[tag][word] = (float(W[tag][word])/C[tag])

    return e_dict
# end of function


def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    """ Takes in a list of unique_tags, an integer that is the number of total tokens (non-unique),
    dictionaries of counts of tags C1, counts of 2-tag sequences C2, and if order = 3, counts of 3-tag sequences C3,
    and integer order; returns a list of 3 floats, lambda1, lambda2, lambda3 using algorithm given in document.
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

                        if alpha0 == max(alpha1, alpha0, alpha2):
                            lambdas[0] += C3[tag_i_2][tag_i_1][tag_i]
                        elif alpha1 == max(alpha1, alpha0, alpha2):
                            lambdas[1] += C3[tag_i_2][tag_i_1][tag_i]
                        else:
                            lambdas[2] += C3[tag_i_2][tag_i_1][tag_i]
    sum_lambdas = lambdas[0] + lambdas[1] + lambdas[2]
    result = [lambdas[0]/sum_lambdas, lambdas[1]/sum_lambdas, lambdas[2]/sum_lambdas]
    return result
# end of function


def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
    """Takes a list of (word, POS-tag) pairs, lists of unique tags and unique words in training data, an integer for
    order of the model, and a boolean for whether uses smoothing or not; returns a HMM object that has initial
    distribution, emission_matrix, transition_matrix and order
    :param training_data: a list of (word, POS-tag) pairs
    :param unique_tags: unique tags
    :param unique_words: unique words
    :param order: either 2 or 3
    :param use_smoothing: true or false, whether or not to use computed lambdas for smoothing
    :return: a HMM object
    """
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
    """Takes A hidden markov model object (2nd order), the sequence of words that should be tagged; returns a tagged
     sentence in the form of a list of(word, POS-tag) pairs
    :param hmm: A hidden markov model object (2nd order HMM)
    :param sentence: the sequence of words that should be tagged.
    :return: A tagged sentence in the form of a list of(word, POS-tag) pairs
    """
    # check input
    if not sentence:
        return []

    # initialize variables
    emission = hmm.emission_matrix
    init_distribution = hmm.initial_distribution
    transit_matrix = hmm.transition_matrix
    all_states = emission.keys()

    v_matrix = defaultdict(lambda: defaultdict(float))
    bp_matrix = defaultdict(lambda: defaultdict(str))
    length = len(sentence)

    z_list = [None] * length
    # fill base cases
    for state in all_states:
        word = sentence[0]
        v_matrix[state][0] = log_wrapper(init_distribution[state]) + log_wrapper(emission[state][word])

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
            v_matrix[curr_state][i]= max_prob + log_wrapper(emission[curr_state][sentence[i]])
            bp_matrix[curr_state][i] = arg_max

    # get the last state for z_list
    max_prob = float("-inf")
    arg_max_last = all_states[0]
    for last_state in emission:
        temp_prob = v_matrix[last_state][length-1]
        if temp_prob > max_prob:
            max_prob = temp_prob
            arg_max_last = last_state

    z_list[length-1] = arg_max_last

    for i in range(length-2, -1, -1):
        z_list[i] = bp_matrix[z_list[i+1]][i+1]

    list_pairs = []
    for i in range(length):
        list_pairs.append((sentence[i], z_list[i]))
    return list_pairs
# end of function


def trigram_viterbi(hmm, sentence):
    """Takes A hidden markov model object (3rd order), the sequence of words that should be tagged; returns a tagged
     sentence in the form of a list of(word, POS-tag) pairs
    :param hmm: A hidden markov model object, (3rd order HMM)
    :param sentence: the sequence of words that should be tagged.
    :return: A tagged sentence in the form of a list of(word, POS-tag) pairs
    """
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
    length = len(sentence)

    # base cases for v
    word0 = sentence[0]
    word1 = sentence[1]
    for state0 in all_states:
        for state1 in all_states:
            if init_distribution[state0][state1] != 0 and emission[state0][word0] != 0 and emission[state1][word1] != 0:
                v_matrix[state0][state1][1] = log_wrapper(init_distribution[state0][state1])\
                                          + log_wrapper(emission[state0][word0]) + log_wrapper(emission[state1][word1])
            else:
                v_matrix[state0][state1][1] = float('-inf')

    for i in range(2, length):
        for prior_state1 in all_states: # wrong result: curr_state; correct result: prior_state1, curr_state
            for curr_state in all_states: # wrong result: prior_state0; correct result: curr_state, prior_state1
                max_prob = float('-inf')
                arg_max0 = prior_state1 # correct result: prior_state1
                for prior_state0 in all_states:
                # wrong result: prior_state1 (if arg_max captures prior_state0, but correct if prior_state1);
                # correct result: prior_state0
                    # arg_max1 = prior_state1
                    if transit_matrix[prior_state0][prior_state1][curr_state] != 0:
                        temp = v_matrix[prior_state0][prior_state1][i-1] + \
                           log_wrapper(transit_matrix[prior_state0][prior_state1][curr_state])
                    else:
                        temp = float('-inf')
                    if temp > max_prob:
                        max_prob = temp
                        arg_max0 = prior_state0 # correct result: prior_state0
                if emission[curr_state][sentence[i]] != 0:
                    v_matrix[prior_state1][curr_state][i] = max_prob + log_wrapper(emission[curr_state][sentence[i]])
                else:
                    v_matrix[prior_state1][curr_state][i] = float('-inf')
                bp_matrix[prior_state1][curr_state][i] = arg_max0

    max_prob = float('-inf')
    arg_max_last = all_states[0]
    arg_max_second_last = all_states[0]
    for prior_state1  in v_matrix:
        for last_state in v_matrix:
            if v_matrix[prior_state1][last_state][length - 1] != 0:
                temp = v_matrix[prior_state1][last_state][length - 1]
                if temp > max_prob:
                    max_prob = temp
                    arg_max_last = last_state
                    arg_max_second_last = prior_state1
            else:
                v_matrix[prior_state1][last_state][length - 1] = float('-inf')
    z_list = [None] * length
    z_list[length - 1] = arg_max_last
    z_list[length - 2] = arg_max_second_last

    for i in range(length-3, -1, -1):
        z_list[i] = bp_matrix[z_list[i+1]][z_list[i+2]][i+2]

    list_pairs = []
    for i in range(length):
        list_pairs.append((sentence[i], z_list[i]))
    return list_pairs


################################################# Helper functions ##########################################
def truncate_training_data(word_tag_pairs, percentage):
    """ Takes a list of word-tag pairs, cut the list according to the given percentage while keeping the last sentence intact
    in the returned list.
    :param word_tag_pairs: a list of word-tag pairs
    :param percentage: an integer. For 1%, percentage would be 1, for 10%, percentage would be 10
    :return: a list of word-tag pairs that has the first percentage of word_tag_pairs
    """
    total_length = len(word_tag_pairs)
    training_length = int(total_length * percentage/100)
    print
    while (word_tag_pairs[training_length-1][0] != "." and training_length < total_length):
        # print training_length
        training_length += 1

    return word_tag_pairs[: training_length]


def experiment(training_file, percentage, testing_file, answer_key_file, order, smoothing):
    """ Takes a string for training file name, an integer for percentage of training data, strings for test data
    file name and answer file name, an integer for order of HMM and a boolean value for using smoothing or not; no return
    The function trains a HHM of given order on the given percentage of training file with or without smoothing,
     tags testing data, then compute accuracy of tagging by comparing tagging result with answer from another file,
     and write the result to a file.
    :param training_file: a string for training file name
    :param percentage: an integer for percentage of training data
    :param testing_file: a string for test data file name
    :param answer_key_file: a string for file name that contains answers
    :param order: order: either 2 or 3
    :param smoothing: true or false, whether or not to use computed lambdas for smoothing
    :return: None
    """
    training_tuple = read_pos_file(training_file)
    word_tag_pairs = training_tuple[0]
    training_pairs = truncate_training_data(word_tag_pairs, percentage)
    unique_words = set()
    unique_tags = set()
    if len(training_pairs) > 1:
        for i in range(len(training_pairs)):
            word = training_pairs[i][0]
            tag = training_pairs[i][1]
            unique_words.add(word)
            unique_tags.add(tag)
    # if not word or not tag: #
    # print count #

    trained_hmm = build_hmm(training_pairs, unique_tags, unique_words, order, smoothing)
    untagged_words = read_untagged_data(testing_file)
    update_hmm(trained_hmm.emission_matrix, untagged_words, list(unique_words))
    sentences = partition_sentences(untagged_words)
    answers = read_pos_file(answer_key_file)[0]
    accuracy = check_accuracy(trained_hmm, sentences, answers, order)
    report = "training data is {0}%, test_data is from {1}, order is {2}, used smoothing {3}, accuracy is {4}\n". \
        format(percentage, testing_file, order, smoothing, accuracy)
    file = open("result.txt", 'a')
    file.write(report)


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


def check_accuracy(hmm, list_of_sentences, word_tag_in_test, order=2):
    """ Takes a HMM object, a list of lists of words (sentences each with periods at the end), a list of word-tag pairs
    that is the correct part-of-speech tagging for the sentences, and an integer for order; returns a float that is the
    percentage of tagging that agree with the given word-tag pairs.
    :param hmm: a HMM object
    :param list_of_sentences: a list of lists of words (sentences each with periods at the end)
    :param word_tag_in_test: a list of word-tag pairs
    that is the correct part-of-speech tagging for the sentences
    :param order: an integer for order
    :return: a float that is the percentage of tagging that agree with the given word-tag pairs.
    """
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

    # print len(total_tagges_from_hmm), len(word_tag_in_test)
    for i in range(0, len(word_tag_in_test)):
        # print i
        if total_tagges_from_hmm[i] == word_tag_in_test[i]:
            corrects += 1
    # print total_tagges_from_hmm
    return corrects * 100.0 / len(word_tag_in_test)


def log_wrapper(number):
    """ A function that takes a number and return the log value of it, or -inf if the number is zero
    :param number: a number that could  be integer or float
    :return: a float
    """
    if number == 0:
        return float("-inf")
    else:
        return log(number)

def update_hmm(emission_matrix, sentence, words_in_training, epsilon=0.00001):
    """ Takes an emission matrix of a HMM, a list of words with a period at the end, a set of unique words in training
    data and a float;
    Modifies the emission_matrix for each word that is in sentence but not in words_in_training such
    that each tag has a small probability of generating a given word, emission probabilities for all existing words
    are increased by the same amount and then all values are normalized such that the matrix stays stochastic.
    :param emission_matrix: emission matrix of a HMM
    :param sentence:  a list of words with a period at the end
    :param words_in_training: a set of unique words in training data
    :param epsilon: a very small float, default to be 0.00001 if no value is given to the function
    :return: None
    """
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
        for word in emission_matrix[state]:
            total += emission_matrix[state][word]
        # normalize
        for word in emission_matrix[state]:
            emission_matrix[state][word] = emission_matrix[state][word]/total


def read_untagged_data(filename):
    """ Reads words from a text file, returns a list of elements if the file content is separated by white spaces
    :param filename: string of file name containing
    :return: a list of elements
    """
    f = open(str(filename), "r")
    words = []
    sentences = []
    for line in f: # line = paragraph
        if len(line.split(" ")) < 1:
            continue
        else:
            words.extend(line.replace("\t", "").strip().split(" "))
    f.close()
    return words


def compute_transition_matrix(order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2):
    """ Takes order, smoothing, count_tags, num_tokens, tag_sequence1, tag_sequence2; return a dictionary of transition probabilities
    :param order: integer for order of an HMM
    :param smoothing: boolean for whether to use smoothing
    :param count_tags: a dictionary for counts of each tag in training data
    :param num_tokens: an integer for number of tokens in training data
    :param tag_sequence1: a dictionary for counts of 2-tag sequences in training data
    :param tag_sequence2: a dictionary for counts of 3-tag sequences in training data
    :return: result_matrix: dictionary that contains transition matrix of an HMM
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
    print "lambdas", lambdas
    if order == 2:
        result_matrix = defaultdict(lambda: defaultdict(float))

        for tag_i_1 in count_tags: #tag_sequence1.items():
            for tag_i in count_tags:
                if count_tags[tag_i_1] == 0:
                    trans_prob = lambdas[0] * float(count_tags[tag_i])/num_tokens
                else:
                    trans_prob = lambdas[1] * float(tag_sequence1[tag_i_1][tag_i])/count_tags[tag_i_1]
                + lambdas[0] * float(count_tags[tag_i])/num_tokens
                result_matrix[tag_i_1][tag_i] = trans_prob
        return result_matrix

    elif order == 3:
        result_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for tag_i_2, tag_i_1_dict in tag_sequence2.items():
            for tag_i_1, tag_i_dict in tag_i_1_dict.items():
                for tag_i in tag_i_dict:
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


################################################# testing functions ########################################
def test_truncate_training_data(word_tag_pairs, percentage):
    return truncate_training_data(word_tag_pairs, percentage)
# Input: training.txt, 10
# Expected output (len(test_truncate_training_data), len(training_data)): 216488 2164655

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
# Input : "training.txt", order = 3
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


def test_decompose_a_pair(training_data):
    print decompose_a_pair(training_data)
# end of function


def test_find_first_of_next_sentence(training_data, next_sentence, boundary):
    print find_first_of_next_sentence(training_data, next_sentence, boundary)
# end of function

# initialize training data
# tuple_of_3_returns = read_pos_file("mytest_tagged.txt")
# tuple_of_3_returns = read_pos_file("training.txt")
#
# training_data = tuple_of_3_returns[0]
# unique_words = tuple_of_3_returns[1]
# unique_tags = tuple_of_3_returns[2]
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
# testdata_tagged = read_pos_file("testdata_tagged.txt")
# testdata_tagged = read_pos_file("mytest_tagged.txt")
# answers = testdata_tagged[0]
# unique_tags = testdata_tagged[1]
# unique_words = testdata_tagged[2]
#
# untagged_words = read_untagged_data("testdata_untagged.txt")
# untagged_words = read_untagged_data("mytest_untagged.txt")
# print test_partition_sentences(untagged_words)

# sentence = ['The', 'cat', 'ran'] #, '.']#, 'away', 'today', '.']
# training_data = [('The','DT'),('cat','NN'),('ran','VBD')] #, ('.', '.')] #,('away','RB'),('today','NN'),('.','.')]
# unique_tags = ['DT','NN','VBD'] #,'.'] #,'RB']
# unique_words = ['The', 'cat', 'ran'] #, '.']
# order = 2
# smoothing = False

# trained_hmm = build_hmm(training_data, unique_tags, unique_words, order, smoothing)
# print trained_hmm.emission_matrix
#
# update_hmm(trained_hmm.emission_matrix, untagged_words, list(unique_words))
# print trained_hmm.emission_matrix

# print trained_hmm.transition_matrix

# # print trained_hmm.initial_distribution
# sentences = partition_sentences(untagged_words)
# print bigram_viterbi(trained_hmm, sentences[0])
# print test_trigram_viterbi(trained_hmm, sentences[0])
# print check_accuracy(trained_hmm, sentences, answers, order)
# #
# input = {'perc': [1], 'order' : [3], 'smoothing': [True, False]}
# # input1 = {'perc': [100], 'order' : [2, 3], 'smoothing': [True, False]}
# for perc in input['perc']:
#     for order in input['order']:
#         for smoothing in input['smoothing']:
#             experiment("training.txt", perc, "testdata_untagged.txt", "testdata_tagged.txt", order, smoothing)