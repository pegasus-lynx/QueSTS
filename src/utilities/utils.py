import numpy as np
import nltk
import re
import os

from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def get_tf_multi(processed, vocabs, normalized=True):
    tokens = []

    for line in processed:
        tokens.extend(line)

    return get_tf(tokens, vocabs, normalized)

def get_tf(line, vocabs, normalized=True):
    dim = len(vocabs["words"])
    tf = np.zeros((dim,1))
    ntokens = len(line)

    for token in line:     
        tf[vocabs["words"][token]] += 1

    if normalized:
        tf /= ntokens

    return tf

def get_tfs(lines, vocabs, normalized=True):
    dim = len(vocabs["words"])    
    tfs = np.zeros((len(lines), dim))

    for i,line in enumerate(lines):
        tfs[i] = get_tf(line, vocabs, normalized)

    return tfs

def get_idf(processed, vocabs):
    n = len(processed)
    dim = len(vocabs["words"])
    idf = np.zeros((dim,1))

    for tokens in processed:
        for token in set(tokens):
            idf[vocabs["words"][token]][0] += 1

    # Hack for avoiding erro due to division by zero
    for p in range(dim):
        if idf[p][0] == 0:
            idf[p][0] = n

    return np.math.log(n/idf, 10, keepdims=True)