import n_grams_of_tr_letters as ng

def generate_statistical_model():
    training_file_names = [
        "D:/Masaüstü/Models/Datasets/05mb_corpora_95"
        "D:/Masaüstü/Models/Datasets/1mb_corpora_95"
        "D:/Masaüstü/Models/Datasets/10mb_corpora_95"
    ]

def test_statictical_model():
    test_file_names = [
        "D:/Masaüstü/Models/Datasets/05mb_corpora_5"
        "D:/Masaüstü/Models/Datasets/1mb_corpora_5"
        "D:/Masaüstü/Models/Datasets/10mb_corpora_5"
    ]



if __name__ == '__main__':

    # x = ng.random.sample(range(0,5),5)
    # print(x)
    # exit()

    # text = "ali haydar kurban"
    #
    # medo = "medine"
    # medo = tuple(medo)
    #
    # print(medo[:-1])
    #
    # a = ng.ngrams(text, 5)
    #
    # for x in a:
    #     S = ''.join(x)
    #     print(S)
    #     if "ali" in S:
    #         print("YASASIN")
    #
    #
    #
    # exit()
    # # cname = "C:/NLP_CORPORA/text"
    #
    # corpora_training_file = "C:/NLP_CORPORA/CORPORA_2/Samples_1/corpora_95"
    # corpora_test_file = "C:/NLP_CORPORA/CORPORA_2/Samples_1/short.txt"

    corpora_training_file = "D:/Masaüstü/Models/CORPORA_2/Samples_1/05mb_corpora_95"
    corpora_test_file = "D:/Masaüstü/Models/CORPORA_2/Samples_1/short.txt"

    # # cname = "D:/Masaüstü/_WIKI_00/w_3/100.txt"
    # # X = ng.ngram(cname,5)
    # # print(X.model[:5])
    #
    # pickle_name = "D:/Masaüstü/Models/5-Gram-corpora_95-training-file"
    # n_gram = ng.pickle.load(open(pickle_name, "rb"))

    n_gram = ng.NGrams(5, corpora_training_file)
    n_gram.save_probabilities()
    n_gram.generate_random_sentences()
    n_gram.evaluation_with_perplexity(corpora_test_file)

    # print(n_gram.probability("ali"))
    #
    # sentence = "ali haydar kurban"
    # a = ngrams(sentence, 2)
    # for i in a:
    #     print(i)
    # my_dict = {'ali': 23, 'aynur': 56, 'muzaffer': 58, "kubra": 26}
    #
    # my_dict = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
    # print("shoul be list", my_dict)
    #
    # my_dict = dict(my_dict)
    # print("shoul be dict", my_dict)
    #
    # for key, val in my_dict.items():
    #     print(key)
    #     print(val)
    #     print("=============")