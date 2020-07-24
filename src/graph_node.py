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

        dot = np.dot(self.vec.T, node.vector())
        dot = float(np.squeeze(dot))

        norm_self = np.linalg.norm(self.vec)
        norm_node = np.linalg.norm(node.vector())

        return dot / (norm_self * norm_node)

    def vsimilarity(self, vector):

        dot = np.dot(self.vec.T, vector)
        dot = float(np.squeeze(dot))

        norm_self = np.linalg.norm(self.vec)
        norm_vec = np.linalg.norm(vector)

        return dot / (norm_self * norm_vec)

class TreeNode(object):

    def __init__(self, node, parent=None):
        self.parent = parent
        self.childs = []

        self.node = node

    def add_child(self, node):
        self.childs.append(node)
        node.parent = self