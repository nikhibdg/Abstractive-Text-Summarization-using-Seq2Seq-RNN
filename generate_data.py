import numpy as np

filename = 'glove.6B.50d.txt'
import csv
import string
from nltk import word_tokenize


def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)

text_files = []
summary_files = []

texts = []
summaries = []

with open('Reviews.csv', 'rb') as csvfile:
    Reviews = csv.DictReader(csvfile)
    k = 1
    i = 0
    for row in Reviews:
        if i % 100000 == 0 and i != 0:
            np.savetxt('raw_text.txt_' + str(k), texts, delimiter='\n', fmt="%s")
            np.savetxt('raw_summary.txt_' + str(k), summaries, delimiter='\n', fmt="%s")
            texts = []
            summaries = []
            k = k + 1
            print k

        i = i + 1
        texts.append(" ".join(word_tokenize(clean(row['Text']))))
        summaries.append(" ".join(word_tokenize(clean(row['Summary']))))

np.savetxt('raw_text.txt_' + k, texts, delimiter='\n', fmt="%s")
np.savetxt('raw_summary.txt_' + k, summaries, delimiter='\n', fmt="%s")


text_files = ["raw_text.txt_1","raw_text.txt_2","raw_text.txt_3","raw_text.txt_4","raw_text.txt_5","raw_text.txt_68454"]
summary_files = ["raw_summary.txt_1","raw_summary.txt_2","raw_summary.txt_3","raw_summary.txt_4","raw_summary.txt_5","raw_summary.txt_68454"]

with open('raw_summary.txt', 'w') as outfile:
    for fname in summary_filessummary_files:
        with open(fname) as infile:
            outfile.write(infile.read())

with open('raw_text.txt', 'w') as outfile:
    for fname in text_files:
        with open(fname) as infile:
            outfile.write(infile.read())