from nltk import ngrams
from collections import Counter
import re
import random
import html2text


UNK = 'UNK'  # It for unknown letter set. (It means that they have not seen before)
N = [1, 2, 3, 4, 5]  # It means 1_Grams, 2_Grams, ..., 5_Grams


class NGrams:
    def __init__(self, n, corpora_name):
        self.n = n  # It is number of gram.
        self.data = []  # It stores the texts of corpora.
        self.counter_dict = {}  # It stores the ngrams with their counters. (It is assigned by 'from collections import Counter')
        self.probabilities = {}  # Smart ways of storing probabilities of the Ngrams.
        self.corpora_name = corpora_name  # It is corpora name.
        self.gt = 0  # !!!!!!!!!!!

        self.check_n()  # It checks if n is valid.
        self.read_corpora()  # It reads the corpora.
        self.fill_ngrams_probabilities_dict()  # It fills the dictionary which holds the probabilities of the Ngrams.

    # It checks that if the value of n is valid.
    # If it is not valid, then the function terminates the program.
    def check_n(self):
        if self.n not in N:
            print("{}_Grams can not be computed! \nOnly the [1-5]_Grams can be computed!".format(self.n))
            exit()

    # This function reads the corpora(dataset) and assigns it to self.data.
    # General format is created.
    # The only acceptable letters are [a-zçğıöşü ].
    # There is not any upper case latter or any numerical letter or any punctuation mark.
    # If the corpora can not be read, the function terminates the program.
    def read_corpora(self):
        file_handle = None
        try:
            file_handle = open(self.corpora_name, "r", encoding="utf8", errors='ignore')
        except FileNotFoundError as e:
            print(e)
            exit()

        data = file_handle.read()
        # data = html2text.html2text(data)
        data = data.lower()
        data = re.sub(r'[^a-zçğıöşü ]', "", data)
        file_handle.close()
        self.data = data

    def fill_ngrams_probabilities_dict(self):
        n_gram = ngrams(self.data, self.n)
        self.counter_dict = Counter(n_gram)
        self.probabilities[UNK] = 0  # { 'UNK': 0 }
        self.good_turing_smoothing()

        for _ngram, _counter in self.counter_dict.items():
            self.probabilities[_ngram] = _counter / len(self.counter_dict)

        self.probabilities = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)

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
                self.counter_dict[_ngram] = r_count
        self.gt = count_one_time / len(self.counter_dict)
