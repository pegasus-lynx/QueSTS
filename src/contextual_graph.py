import numpy as np
from utilities.config import PROC_DIR, CGRAPH_DIR
from utilities.preprocess import read_datafile
from utilities.utils import get_idf

from os.path import join

class ContextualGraphNode(object):
    
    def __init__(self, vector = None, line = None, tokens = None):
        self.line = line
        self.tokens = tokens
        self.vec = vector

    def vector(self):
        return self.vec

    def similarity(self, node):

        dot = np.dot(self.vector(), node.vector())

        norm_self = np.linalg.norm(self.vector())
        norm_node = np.linalg.norm(node.vector())

        return dot / (norm_self * norm_node)


class ContextualGraph(object):
    
    def __init__(self, file, vocabs=None):
        self.filename = file

        self.raw = None
        self.processed = None

        self.length = None

        self.nodes = None
        self.adj_mat = None
        self.doc_idf = None

        if vocabs is not None:
            self.build(vocabs)
    
    def build(self, vocabs):
        filepath = join(PROC_DIR, self.filename)

        self.raw, self.processed = read_datafile(filepath)
        self.length = len(self.raw)

        self.doc_idf = get_idf()

        for line, tokens in zip(self.raw, self.processed):
            node = self.make_node(line, tokens, vocabs)
            self.nodes.append(node)

        self.make_adj_mat()

    def make_node(self, line, tokens, vocabs):
        
        dim = len(vocabs["words"])
        
        node = ContextualGraphNode()
        node.vec = np.zeros((dim,1))
        
        node.line = line
        node.tokens = tokens
        
        for token in tokens:
            node.vec[vocabs["words"][token]][0] += 1

        node.vec = np.dot(node.vec, self.idf)
        return node

    def make_adj_mat(self):

        self.adj_mat = np.zeros((self.length, self.length))

        for p in range(self.length):
            for q in range(p):
                sim = self.nodes[p].similarity(self.nodes[q])
                if sim > 0.001:
                    self.adj_mat[p][q] = sim

    def save(self):
        filepath = join(CGRAPH_DIR, self.filename)

        dim = self.nodes[0].shape[0]
        nodes = np.zeros(self.length, dim)

        for i,node in enumerate(self.nodes):
            nodes[i] = node.vector()

        np.savez_compressed(filepath, nodes=nodes, adj_mat=self.adj_mat)

    @staticmethod
    def load(file):
        graph = ContextualGraph(file)

        textfile_path = join(PROC_DIR, file)
        datafile_path = join(CGRAPH_DIR, file)
        
        nodes = None
        with open(datafile_path, "r") as data:
            graph.adj_mat = data["adj_mat"]
            nodes = data["nodes"]

        graph.length = nodes.shape[0]

        raw, processed = read_datafile(textfile_path)

        for p in range(graph.length):
            node = ContextualGraphNode(nodes[p], raw[p], processed[p])
            graph.nodes.append(node)

        return graph