import nltk
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from nltk import word_tokenize
import string

filename = 'glove.6B.50d.txt'

pos_tag_list = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]


def pos_tag_embedd():
    pos_embedd_dict = {}
    for tag in pos_tag_list:
        pos_embedd_dict[tag] = np.random.rand(10,)

    return pos_embedd_dict


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
pos_embedd_dict = pos_tag_embedd()



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
    raw = open('features_summary.txt').read()
    tokens = raw.split()

    fdist = nltk.FreqDist(tokens)

    embd = []
    vocab = []
    pos = nltk.pos_tag(["unk"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("unk", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("unk")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("unk"))

    pos = nltk.pos_tag(["zebra"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("zebra", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("zebra")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("zebra"))

    pos = nltk.pos_tag(["eos"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("END", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("eos")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("eos"))

    #print embd

    for word, frequency in fdist.most_common(50000):
        if frequency >= 3 and word in glove_vocab:

            pos_word = word.split("/")
            tag = pos_word[1]
            new_word = pos_word[0]
            pos_vec = pos_embedd_dict[tag]
            word_vec = word2vec(new_word)
            feature_vec = np.concatenate((word_vec, pos_vec))
            embd.append(feature_vec)
            vocab.append(word)

    print "size of summary vocab :: ", len(vocab)
    np.savetxt('features_summary_vocab.txt', vocab, delimiter='\n', fmt="%s")
    np.savetxt('features_summary_vocab_embedings.txt', embd, delimiter=' ')


def text_vocab():
    raw = open('features_text.txt').read()
    tokens = raw.split()

    fdist = nltk.FreqDist(tokens)

    embd = []
    vocab = []
    pos = nltk.pos_tag(["unk"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("unk", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("unk")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("unk"))

    pos = nltk.pos_tag(["zebra"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("zebra", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("zebra")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("zebra"))

    pos = nltk.pos_tag(["eos"])
    tag = pos[0][1]
    vocab.append("{}/{}".format("END", tag))
    pos_vec = pos_embedd_dict[tag]
    word_vec = word2vec("eos")
    feature_vec = np.concatenate((word_vec, pos_vec))
    embd.append(feature_vec)
    #embd.append(word2vec("eos"))

    #print embd

    for word, frequency in fdist.most_common(50000):
        if frequency >= 3 and word in glove_vocab:

            pos_word = word.split("/")
            tag = pos_word[1]
            new_word = pos_word[0]
            pos_vec = pos_embedd_dict[tag]
            word_vec = word2vec(new_word)
            feature_vec = np.concatenate((word_vec, pos_vec))
            embd.append(feature_vec)
            vocab.append(word)


    print "size of text vocab :: ", len(vocab)
    np.savetxt('features_text_vocab.txt', vocab, delimiter='\n', fmt="%s")
    np.savetxt('features_text_vocab_embedings.txt', embd, delimiter=' ')


summary_vocab()
text_vocab()
