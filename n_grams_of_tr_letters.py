from nltk import ngrams
from collections import Counter
import re
import random
import html2text
import math  # math.log(X) is 'the log base e of X'
import pickle

UNK = 'UNK'  # It for unknown letter set. (It means that they have not seen before)
N = [1, 2, 3, 4, 5]  # It means 1_Grams, 2_Grams, ..., 5_Grams
TOP_5 = 5  # It is for generating a sentence with top 5 probabilistic ngrams.


# It is a class that hold some information about the ngram.
# It also has necessary functions to create ngrams, probability, etc.
class NGrams:
    def __init__(self, n, training_file):
        self.n = n  # It is number of gram.
        self.data = []  # It stores the texts of corpora.
        self.counter_dict = {}  # It stores the ngrams with their counters. (It is assigned by 'from collections import Counter')
        self.probabilities = {}  # Smart ways of storing probabilities of the Ngrams.
        self.corpora_training_file = training_file  # It is corpora name for training.
        self.corpora_test_file = ""  # It is corpora name for testing.
        self.p_0 = 0
        self.check_n()  # It checks if n is valid.
        self.data = read_corpora(self.corpora_training_file)  # It reads the training corpora and stores it self.data.
        self.fill_ngrams_probabilities_dict()  # It fills the dictionary which holds the probabilities of the Ngrams.

    # It checks that if the value of n is valid.
    # If it is not valid, then the function terminates the program.
    def check_n(self):
        if self.n not in N:
            print("{}_Grams can not be computed! \nOnly the [1-5]_Grams can be computed!".format(self.n))
            exit()

    # This function calculates the probabilities of ngrams based on the Markov Assumption
    def fill_ngrams_probabilities_dict(self):
        n_gram = ngrams(self.data, self.n)  # It generates ngrams 'from nltk import ngrams is used for that'
        self.counter_dict = Counter(n_gram)  # It finds the counter of each ngrams and holds them in a dict 'from collections import Counter is used fro that'
        self.probabilities[UNK] = 0  # { 'UNK': 0 } It is for unkown ngrams.
        self.good_turing_smoothing()  # It does good turing smoothing.

        # If the system is 1-gram no need to Markov Assumption.
        if self.n == 1:
            for _ngram, _counter in self.counter_dict.items():
                self.probabilities[_ngram] = _counter / len(self.counter_dict)
        # Otherwise use the Markov Assumption.
        else:
            markov_asm_n = self.n - 1
            markov_n_gram = ngrams(self.data, markov_asm_n)
            markov_asm_counter = Counter(markov_n_gram)
            for _ngram, _counter in self.counter_dict.items():
                self.probabilities[_ngram] = _counter / markov_asm_counter.get(_ngram[0:-1])

    # This function does GT smoothing
    # It creates reconstructed count of each ngrams
    def good_turing_smoothing(self):
        gt_counter_dict = {}  # { _count: occurs } It holds how many time '_count' occurs.
        count_one_time = 0  # It holds the count of ngram which occurs only one time.

        # Filling gt_counter_dict with _count and how many times '_count' occurs
        for _count in self.counter_dict.values():
            if gt_counter_dict.get(_count) is None:
                gt_counter_dict[_count] = 1
            else:
                gt_counter_dict[_count] = gt_counter_dict.get(_count) + 1
            if _count == 1:
                count_one_time = count_one_time + 1

        for _ngram, _count in self.counter_dict.items():
            if gt_counter_dict.get(_count + 1) is not None:
                c_add_1 = _count + 1
                N_c_add_1 = gt_counter_dict.get(_count + 1)
                N_c = gt_counter_dict.get(_count)
                r_count = c_add_1 * (N_c_add_1 / N_c)
                self.counter_dict[_ngram] = r_count  # Assign reconstructed counter to the counter_dict
        self.p_0 = count_one_time / len(self.counter_dict)

    # This function generates random sentences with top 5 ngrams.
    # Sort the probabilities with inverse order and picks top 5 of them randomly.
    def generate_random_sentences(self):
        self.probabilities = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)
        random_sentences = []

        for j in range(TOP_5):
            sentence = ['', '', '', '', '']
            random_nums = random.sample(range(0, TOP_5), TOP_5)  # Random sampling without replacement.
            for i in range(len(random_nums)):
                sentence[i] = sentence[i].join(self.probabilities[random_nums[i]][0])
            random_sentences.append(sentence)
        print_sentences(random_sentences)
        self.probabilities = dict(self.probabilities)

    # It is an evaluation test with perplexity.
    # It use the %5 of the corpora and find the perplexity of each word.
    def evaluation_with_perplexity(self, test_file):
        self.corpora_test_file = test_file
        test_data = read_corpora(test_file)  # It read the test file and stores in test_data.
        test_data = test_data.split(" ")  # It splits the text with ".". It means that the text is a sentence.
        perplexity_dict = {}  # It holds each word and its perplexity. { word: perplexity}

        # For each word it calculates the perplexity of the word
        for word in test_data:
            _N = len(word)
            temp_prob = self.probability(word)  # It calculates the probability of the word.
            if _N != 0 and temp_prob != 0:
                pp_prob = pow((1 / temp_prob), (1 / _N))  # Perplexity formula.
                perplexity_dict[word] = pp_prob  # Store the perplexity

        print_perplexity_dict(perplexity_dict)
        # return perplexity_dict

    # It finds the probability of the word.
    def probability(self, word):
        result = 0  # Probability addition
        # result = 1.0  # Probability multiplication
        _N = len(word)

        # If the length of test word is greater than or equal to self.n
        if _N >= self.n:
            # It finds the ngrams of the test word and calculates the probability.
            for i in range(1 + _N - self.n):
                next_gram = ngrams(word[i: i + self.n], self.n)
                for current_gram in next_gram:
                    # If the ngram of the word is not in our corpora that means it is unkown or zero unseen.
                    if self.probabilities.get(current_gram) is None:
                        str_n_gram = ''.join(current_gram)  # tuple to string operation
                        # It means that it is in corpora but unseen combination
                        if str_n_gram in self.data:
                            result = result + math.log(self.p_0)
                            # result = result * self.p_0
                        # It means that it is Out Of Vocabulary
                        else:
                            result = result + math.log((self.probabilities[UNK] + 1) / len(self.probabilities))
                            # result = result * (self.probabilities[UNK] + 1) / len(self.probabilities)
                            self.probabilities[UNK] = self.probabilities[UNK] + 1

                    # Otherwise find the probability
                    else:
                        result = result + math.log(self.probabilities.get(current_gram))
                        # result = result * self.probabilities.get(current_gram)

        # If the length of test word is smaller than self.n,
        # even if it occurs then we can not find it in our corpora.
        # So that it searches the word in the ngram.
        elif _N < self.n:
            str_n_gram = ""
            include_count = 0
            for _ngram in self.counter_dict.keys():
                str_n_gram = ''.join(_ngram)  # tuple to string operation
                # If we found it in ngrams
                if word in str_n_gram:
                    include_count = include_count + self.counter_dict.get(_ngram)

            # If the ngram of the word is not in our corpora that means it is unkown or zero unseen.
            if include_count == 0:
                # It means that it is in corpora but unseen combination
                if word in self.data:
                    result = result + math.log(self.p_0)
                    # result = result * self.p_0
                # It means that it is in corpora but unseen combination
                else:
                    self.probabilities[UNK] = self.probabilities.get(UNK) + 1
                    result = result + math.log(self.probabilities.get(UNK) / len(self.probabilities))
                    # result = self.probabilities.get(UNK) / len(self.probabilities)
            # Otherwise
            else:
                result = result + math.log(include_count / len(self.probabilities))
                # result = include_count / len(self.probabilities)

        return math.exp(result)
        # return result

    # This is for saving the model.
    # We don not need to do statistic operation for each test.
    # Once do that operation after that use that pickle from where you stored it.
    def save_probabilities(self):
        file_name = self.corpora_training_file.split("/")
        file_name = file_name[len(file_name) - 1]
        NAME = "D:/Masaüstü/Models/{}-Gram-{}-training-file".format(self.n, file_name)
        pickle_out = open(NAME, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


# This function reads the corpora for both training set and test set
# The only acceptable letters are [a-zçğıöşü ] for training.
# There is not any upper case latter or any numerical letter or any punctuation mark.
# If the corpora can not be read, the function terminates the program.
def read_corpora(corpora_name):
    file_handle = None
    try:
        file_handle = open(corpora_name, "r", encoding="utf8", errors='ignore')
    except FileNotFoundError as e:
        print(e)
        exit()

    data = file_handle.read()
    data = html2text.html2text(data)
    data = data.lower()
    data = re.sub(r'[^a-zçğıöşü ]', '', data)
    file_handle.close()
    return data


# It is print function.
# It calls print_one_sentence or print_all_sentences.
def print_sentences(random_sentences):
    which_one = random.randint(0, TOP_5 - 1)
    # print_one_sentence(random_sentences[which_one])
    print_all_sentences(random_sentences)


# It print one sentence of the random sentences.
def print_one_sentence(one_sentence):
    print("Printing one random sentence...")
    for word in one_sentence:
        print(word, end=' ')
    print()


# It prints all random sentences (amount of 5).
def print_all_sentences(all_sentences):
    print("Printing all random sentences...")
    for sentence in all_sentences:
        for word in sentence:
            print(word, end=' ')
        print()


# It prints the perplexities.
def print_perplexity_dict(perplexity_dict):
    for _word, _perp in perplexity_dict.items():
        print("{", _word, ":", _perp, "}")
