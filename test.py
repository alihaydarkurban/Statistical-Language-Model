import n_grams_of_tr_letters as ng

if __name__ == '__main__':

    corpora_training_file = "corpora_95"
    corpora_test_file = "corpora_test"
    n_gram = ng.NGrams(5, corpora_training_file)
    n_gram.generate_random_sentences()
    n_gram.evaluation_with_perplexity(corpora_test_file)