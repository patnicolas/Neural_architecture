__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

default_stop_words_file = "../../stopwords/english"

class StopWords(object):
    def __init__(self, stop_words_file = default_stop_words_file):
        acc = []
        with open(stop_words_file, 'r') as stop_words:
            for line in stop_words.read().splitlines():
                acc.append(line)
        self.stopwords = set(acc)

    def remove(self, words):
        return filter(lambda w: w not in self.stopwords, words)


stop_words_file = "../../stopwords/english"
stop_words = StopWords(stop_words_file)
print(stop_words.stopwords)

new_words = ["hello", "the"]
cleaned_words = stop_words.remove(new_words)
for w in cleaned_words:
    print(w)

