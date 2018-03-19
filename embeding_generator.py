import numpy as np
filename = 'glove.6B.50d.txt' 


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('GloVe Loaded.')
    file.close()
    return vocab,embd

vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embd[0])

import csv
import nltk as nlp
from nltk import word_tokenize
import string

summaries = []
texts = []

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)     

with open('Reviews.csv', 'rb') as csvfile: 
    Reviews = csv.DictReader(csvfile)
    i = 0
    for row in Reviews:
        i +=1
        if i==1000:
            break

        clean_text = clean(row['Text'])
        clean_summary = clean(row['Summary'])
        print(i)
        summaries.append(word_tokenize(clean_summary))
        texts.append(word_tokenize(clean_text))

def np_nearest_neighbour(x):
    #returns array in embedding that's most similar (in terms of cosine similarity) to x

    xdoty = np.multiply(embedding,x)
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)

    return embedding[np.argmax(cosine_similarities)]


def word2vec(word):
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

def vec2word(vec):
    for x in range(0, len(embedding)):
            if np.array_equal(embedding[x],np.asarray(vec)):
                return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))

# word = "king"
# print("Vector representation of '"+str(vec2word(word2vec("kingdom")))+"':\n")
# print(np.dot(np.array(word2vec("king")), np.array(word2vec("kingdom"))))
# print(np.dot(np.array(word2vec("king")), np.array(word2vec("queen"))))


vec_summaries = []

for summary in summaries:

    vec_summary = []

    for word in summary:
        vec_summary.append(word2vec(word))

    vec_summary.append(word2vec('eos'))

    vec_summary = np.asarray(vec_summary)
    vec_summary = vec_summary.astype(np.float32)

    vec_summaries.append(vec_summary)


vec_texts = []

for text in texts:

    vec_text = []

    for word in text:
        vec_text.append(word2vec(word))

    vec_text.append(word2vec('eos'))
    vec_text = np.asarray(vec_text)
    vec_text = vec_text.astype(np.float32)

    vec_texts.append(vec_text)


import pickle
with open('vec_summaries', 'wb') as fp:
    pickle.dump(vec_summaries, fp)
with open('vec_texts', 'wb') as fp:
    pickle.dump(vec_texts, fp)
