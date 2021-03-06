import numpy as np
from os.path import join

from utilities.config import PROC_DIR, CGRAPH_DIR
from utilities.preprocess import read_datafile
from utilities.utils import get_isf, get_tf

from graph_node import GraphNode

class ContextualGraph(object):
    
    def __init__(self, file, vocabs=None):
        self.filename = file

        self.raw = None
        self.processed = None

        self.length = None

        self.nodes = None
        self.adj_mat = None
        self.isf = None

        if vocabs is not None:
            self.build(vocabs)
    
    def build(self, vocabs, isf):
        filepath = join(PROC_DIR, self.filename)

        self.raw, self.processed = read_datafile(filepath)
        self.length = len(self.raw)

        self.isf = isf

        for line, tokens in zip(self.raw, self.processed):
            node = self.make_node(line, tokens, vocabs)
            self.nodes.append(node)

        self.make_adj_mat()

    def make_node(self, line, tokens, vocabs):
        
        dim = len(vocabs["words"])
        
        node = GraphNode()
        node.vec = np.zeros((dim,1))
        
        node.line = line
        node.tokens = tokens
        
        tf = get_tf(tokens, vocabs)

        node.vec = tf*self.isf
        return node

    def make_adj_mat(self):

        self.adj_mat = np.zeros((self.length, self.length))

        for p in range(self.length):
            for q in range(p):
                sim = self.nodes[p].similarity(self.nodes[q])
                if sim > 0.001:
                    self.adj_mat[p][q] = sim
                    self.adj_mat[q][p] = sim

    def parents(self):
        return np.argmax(self.adj_mat, axis=1).tolist()

    def save(self):
        filepath = join(CGRAPH_DIR, self.filename)

        dim = self.nodes[0].shape[0]
        nodes = np.zeros(self.length, dim)

        for i,node in enumerate(self.nodes):
            nodes[i] = node.vector()

        np.savez_compressed(filepath, nodes=nodes, adj_mat=self.adj_mat)

    def save_json(self):
        pass

    @staticmethod
    def load(file):
        graph = ContextualGraph(file)

        textfile_path = join(PROC_DIR, file)
        datafile_path = join(CGRAPH_DIR, file.replace("txt", "npz"))
        
        nodes = None
        with open(datafile_path, "r") as data:
            graph.adj_mat = data["adj_mat"]
            nodes = data["nodes"]

        graph.length = nodes.shape[0]

        raw, processed = read_datafile(textfile_path)

        for p in range(graph.length):
            node = GraphNode(nodes[p], raw[p], processed[p])
            graph.nodes.append(node)

        return graph

    @staticmethod
    def load_json(file):
        pass

    def __len__(self):
        return len(self.nodes)