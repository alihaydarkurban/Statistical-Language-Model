import n_grams_of_tr_letters as ng


if __name__ == '__main__':

    # cname = "C:/NLP_CORPORA/text"

    cname = "C:/NLP_CORPORA/CORPORA_2/Samples_1/corpora_95.txt"

    # cname = "D:/Masaüstü/_WIKI_00/w_3/100.txt"
    # X = ng.ngram(cname,5)
    # print(X.model[:5])

    n_gram = ng.NGrams(5, cname)
    n_gram.generate_random_sentences()
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