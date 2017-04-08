import numpy as np
import pandas
import nltk

sentence_pairs = np.array([["das haus", "the house"], ["das buch", "the book"], ["ein buch", "a book"]])
translation_prob = {'das': {'house': 0.25, 'the': 0.25, 'book': 0.25, 'a': 0.25},
                    'haus': {'house': 0.25, 'the': 0.25, 'book': 0.25, 'a': 0.25},
                    'buch': {'house': 0.25, 'the': 0.25, 'book': 0.25, 'a': 0.25},
                    'ein': {'house': 0.25, 'the': 0.25, 'book': 0.25, 'a': 0.25}}
count_conditional = {'das': {'house': 0, 'the': 0, 'book': 0, 'a': 0},
                     'haus': {'house': 0, 'the': 0, 'book': 0, 'a': 0},
                     'buch': {'house': 0, 'the': 0, 'book': 0, 'a': 0},
                     'ein': {'house': 0, 'the': 0, 'book': 0, 'a': 0}}
translation_prob = pandas.DataFrame.from_dict(translation_prob)
all_german = translation_prob.axes[1]
all_english = translation_prob.axes[0]
count_conditional = pandas.DataFrame.from_dict(count_conditional)
print(count_conditional)
count = 0
total = pandas.Series(index=all_german, data=0.0)
s_total = pandas.Series(index=all_english, data=0.0)


def zero(a):
    return 0.0


while count <= 30:
    count_conditional = count_conditional.applymap(func=zero)
    total = total.apply(func=zero)
    for sentence_pair in sentence_pairs:
        for eng_word in nltk.tokenize.casual_tokenize(sentence_pair[1]):
            s_total[eng_word] = 0.0
            for german_word in nltk.tokenize.casual_tokenize(sentence_pair[0]):
                s_total[eng_word] += translation_prob[german_word][eng_word]
        for eng_word in nltk.tokenize.casual_tokenize(sentence_pair[1]):
            for german_word in nltk.tokenize.casual_tokenize(sentence_pair[0]):
                count_conditional.loc[eng_word, german_word] += translation_prob[german_word][eng_word] / s_total[
                    eng_word]
                total[german_word] += translation_prob[german_word][eng_word] / s_total[eng_word]
    print("coiukjb")
    print(count_conditional)
    print("total")
    print(total)
    for german_word in all_german:
        for eng_word in all_english:
            translation_prob.loc[eng_word, german_word] = count_conditional[german_word][eng_word] / total[german_word]
    print(translation_prob)
    count += 1
