import numpy as np
import os

from utilities.preprocess import tokenize_query
from utilities.utils import get_query_wts

class Query(object):

    def __init__(self, text):
        self.text = text
        self.tokens = tokenize_query(text)
        self.weights = np.zeros((len(self.tokens), 1))

    def __len__(self):
        return len(self.tokens)

    def weights(self):
        return self.weights

    def set_weights(self, vocabs, isf):
        self.weights = get_query_wts(self.tokens, isf, vocabs)

    def expand(self, vocabs):
        vectors = []
        dim = len(vocabs["words"])

        for i,token in enumerate(self.tokens):
            try: 
                p = vocabs["words"][token]
            except Exception as e:
                p = vocabs["words"]["<oov>"]
            
            vec = np.zeros((dim, 1))
            vec[p][0] = self.weights[i][0]
            vectors.append(vec)

        return vectors