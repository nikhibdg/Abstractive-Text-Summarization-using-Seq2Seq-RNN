import numpy as np
filename = 'glove.6B.50d.txt'
import matplotlib.pyplot as plt


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


def summary_embeds():
    embd = []
    i = 0;
    file = open("summary_vocab.txt",'r')
    for word in file.readlines():
        i+=1
        print word.rstrip()
        embd.append(word2vec(word.rstrip()))

    np.savetxt('summary_vocab_embedings.txt', embd, delimiter=' ')

def text_embeds():
    embd = []
    i = 0;
    file = open("text_vocab.txt", 'r')
    for word in file.readlines():
        i += 1
        print word.rstrip()
        embd.append(word2vec(word.rstrip()))

    np.savetxt('text_vocab_embedings.txt', embd, delimiter=' ')

summary_embeds()
text_embeds()