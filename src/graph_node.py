import numpy as np

class GraphNode(object):
    
    def __init__(self, vector = None, line = None, tokens = None):
        self.id = None
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