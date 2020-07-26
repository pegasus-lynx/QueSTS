import numpy as np

class GraphNode(object):
    
    def __init__(self, vector = None, line = None, tokens = None):
        self.id = None
        
        self.line = line
        self.tokens = tokens
        
        self.vec = vector

    def has(self, term):
        if term in self.tokens:
            return True
        return False

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

    def __eq__(self, other):
        if isinstance(other,GraphNode):
            if self.tokens == other.tokens and self.id == other.id:
                return True
        return False

    def __neq__(self, other):
        return not self.__eq__(other)

class TreeNode(object):

    def __init__(self, node, parent=None):
        self.parent = parent
        self.childs = []

        self.node = node

    def add_child(self, node):
        self.childs.append(node)
        node.parent = self

    def __eq__(self, other):
        
        if self.node != other.node:
            return False

        if len(self.childs) != len(other.childs):
            return False

        if len(self.childs) == 0:
            return True

        for ix in range(len(self.childs)):
            if self.childs[ix].node != other.childs[ix].node:
                return False

        return True
        
    def __ne__(self, other):
        if self.node != other.node:
            return True

        if len(self.childs) != len(other.childs):
            return True

        if len(self.childs) == 0:
            return False

        for ix in range(len(self.childs)):
            if self.childs[ix].node != other.childs[ix].node:
                return True

        return False
            