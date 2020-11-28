import n_grams_of_tr_letters as ng
from nltk import ngrams
# import re
#
#
# def read_file(filename):
#     f = open(filename, "r", encoding="utf8", errors='ignore')
#     data = f.read()
#     data = data.lower()
#     data = re.sub(r'[^a-zçğıöşü ]', '', data)
#     data = data.lower()
#     f.close()
#     return data
#
#
# arr = read_file("C:/NLP_CORPORA/CORPORA_1/Samples_1/corpora_5.txt")
# print(arr[:200])

if __name__ == '__main__':

    cname = "C:/NLP_CORPORA/CORPORA_1/Samples_1/corpora_95.txt"

    # X = ng.ngram(cname,5)
    # print(X.model[:5])


    n_gram = ng.NGrams(5, cname)
    print(n_gram.probabilities[:100])

    #
    # # sentence = "ali haydar kurban"
    # # a = ngrams(sentence, 2)
    # # for i in a:
    # #     print(i)
    # my_dict = {'ali': 23, 'aynur': 56, 'muzaffer': 58, "kubra": 26}
    #
    # my_dict = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
    # print(my_dict)