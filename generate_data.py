import numpy as np
filename = 'glove.6B.50d.txt'
import csv
import string
from nltk import word_tokenize

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)


texts = []
summaries = []
with open('Reviews.csv', 'rb') as csvfile:
    Reviews = csv.DictReader(csvfile)
    i = 0
    for row in Reviews:
        i +=1
        if i==100000:
            break

        texts.append(" ".join(word_tokenize(clean(row['Text']))))
        summaries.append(" ".join(word_tokenize(clean(row['Summary']))))

np.savetxt('text.txt', texts, delimiter='\n', fmt="%s")
np.savetxt('summary.txt', summaries, delimiter='\n', fmt="%s")
