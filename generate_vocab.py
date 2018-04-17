import nltk as nlp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv
from nltk import word_tokenize
import string

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

glove_vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embd[0])



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
        return embedding[glove_vocab.index(word)]


def summary_vocab():
    raw = open('raw_summary.txt').read()
    tokens = nlp.word_tokenize(raw)

    fdist = nlp.FreqDist(tokens)

    embd = []
    vocab = []
    vocab.append("unk")
    embd.append(word2vec("unk"))

    vocab.append("BEGIN")
    embd.append(word2vec("zebra"))

    vocab.append("END")
    embd.append(word2vec("eos"))

    print embd

    for word, frequency in fdist.most_common(50000):
        if frequency >= 3 and word in glove_vocab:
            embd.append(word2vec(word))
            vocab.append(word)

    print "size of summary vocab :: ", len(vocab)
    np.savetxt('summary_vocab.txt', vocab, delimiter='\n', fmt="%s")
    np.savetxt('summary_vocab_embedings.txt', embd, delimiter=' ')


def text_vocab():
    raw = open('raw_text.txt').read()
    tokens = nlp.word_tokenize(raw)

    fdist = nlp.FreqDist(tokens)

    embd = []
    vocab = []

    vocab.append("unk")
    embd.append(word2vec("unk"))

    vocab.append("BEGIN")
    embd.append(word2vec("zebra"))

    vocab.append("END")
    embd.append(word2vec("eos"))

    for word, frequency in fdist.most_common(50000):
        if frequency >= 5 and word in glove_vocab:
            embd.append(word2vec(word))
            vocab.append(word)

    print "size of text vocab :: ", len(vocab)
    np.savetxt('text_vocab.txt', vocab, delimiter='\n', fmt="%s")
    np.savetxt('text_vocab_embedings.txt', embd, delimiter=' ')

# print "END to end",np.dot( word2vec("eos"), word2vec("eos"))
# print "BEGIN to begin",np.dot( word2vec("zebra"), word2vec("eos"))

summary_vocab()
text_vocab()