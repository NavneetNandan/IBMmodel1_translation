import numpy as np
import pandas
import nltk

sentence_pairs = np.array([["das haus", "the house"], ["das buch", "the book"], ["ein buch", "a book"]])
translation_prob = {}
count_conditional = {}
all_english = set()
all_german = set()
for eng_sentence in sentence_pairs[:, 1]:
    for eng_word in nltk.tokenize.casual_tokenize(eng_sentence):
        all_english.add(eng_word)
for german_sentence in sentence_pairs[:, 0]:
    for german_word in nltk.tokenize.casual_tokenize(german_sentence):
        all_german.add(german_word)
uniform_translation_prob = 1 / len(all_english)
for german_word in all_german:
    translation_prob[german_word] = {}
    count_conditional[german_word] = {}
    for eng_word in all_english:
        translation_prob[german_word][eng_word] = uniform_translation_prob
        count_conditional[german_word][eng_word] = 0.0
translation_prob = pandas.DataFrame.from_dict(translation_prob)
count_conditional = pandas.DataFrame.from_dict(count_conditional)
total = pandas.Series(index=all_german, data=0.0)
s_total = pandas.Series(index=all_english, data=0.0)

def zero(a):
    return 0.0

count = 0
while count <= 5:
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
    print("count_conditional")
    print(count_conditional)
    print("total")
    print(total)
    for german_word in all_german:
        for eng_word in all_english:
            translation_prob.loc[eng_word, german_word] = count_conditional[german_word][eng_word] / total[german_word]
    print(translation_prob)
    count += 1
