import nltk as nlp
import numpy as np
import matplotlib.pyplot as plt


def summary_vocab():
    raw = open('raw_summary.txt').read()
    tokens = nlp.word_tokenize(raw)

    fdist = nlp.FreqDist(tokens)

    vocab = []

    vocab.append("unk")
    vocab.append("eos")
    vocab.append("sos")

    for word, frequency in fdist.most_common(50000):
        if frequency >= 3:
            vocab.append(word)

    print "size of summary vocab :: ", len(vocab)
    np.savetxt('summary_vocab.txt', vocab, delimiter='\n', fmt="%s")

def text_vocab():
    raw = open('raw_text.txt').read()
    tokens = nlp.word_tokenize(raw)

    fdist = nlp.FreqDist(tokens)

    vocab = []

    vocab.append("unk")
    vocab.append("eos")
    vocab.append("sos")

    for word, frequency in fdist.most_common(50000):
        if frequency >= 5:
            vocab.append(word)

    print "size of text vocab :: ", len(vocab)
    np.savetxt('text_vocab.txt', vocab, delimiter='\n', fmt="%s")

summary_vocab()
text_vocab()