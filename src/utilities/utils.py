import numpy as np
import re
import os

from utilities.config import PROC_DIR
from utilities.utils import get_tf
from utilities.preprocess import read_datafile
from os.path import  join

def get_tf_doc(processed, vocabs, normalized=True):
    tokens = []

    for line in processed:
        tokens.extend(line)

    return get_tf(tokens, vocabs, normalized)

def get_tf(line, vocabs, normalized=True):
    dim = len(vocabs["words"])
    tf = np.zeros((dim,1))
    ntokens = len(line)

    for token in line:  
        if token not in vocabs["words"].keys():
            token = "<oov>"   
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

def get_isf(files, vocabs):
    nsent = 0
    dim = len(vocabs["words"])
    isf = np.zeros((dim,1))

    for file in files:
        filepath = join(PROC_DIR,file)
        _, processed = read_datafile(filepath)
        nsent += len(processed)
        for tokens in processed:
            for token in set(tokens):
                isf[vocabs["words"][token]][0] += 1

    for p in range(dim):
        if isf[p][0] == 0:
            isf[p][0] = nsent

    return np.math.log(nsent/isf, 10, keepdims=True)

def get_query_wts(tokens, isf, vocabs):
    
    weights = np.zeros((len(tokens),1))
    tf = get_tf(tokens, vocabs)
    vec = tf*isf

    for i,token in enumerate(tokens):
        weights[i][0] = vec[vocabs["words"][token]]

    return weights

def query_similarity_weights(query, igraph, vocabs, isf):
    
    nwts = np.zeros((len(query), len(igraph)))

    qvecs = query.expand(vocabs)

    for i,qvec in enumerate(qvecs):
        for j,node in enumerate(igraph.nodes):
            nwts[i][j] = node.vsimilarity(qvec)

    return nwts

def differs(wts_prev, wts_curr, diff=0.00001):
    delta =  np.absolute(wts_curr - wts_prev)
    if np.sum(delta) < diff:
        return True
    return False 

def transpose(mat):

    row = len(mat)
    col = len(mat[0])

    tmat = [ [None]*row ]*col

    for r in range(row):
        for c in range(col):
            tmat[c][r] = mat[r][c]

    return tmat