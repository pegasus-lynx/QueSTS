import numpy as np
from os.path import join

from graph_node import GraphNode
from contextual_graph import ContextualGraph
from utilities.config import IGRAPH_DIR, VOCAB_DIR
from utilities.preprocess import load_vocabs

class IntegratedGraph(object):

    def __init__(self, file, vocabs, graphs=None):
        self.filename = file

        self.ndocs = 0
        self.vocabs = vocabs
        self.cgraphs = []

        # Integrated Graph Structure
        self.eta = 0
        self.length = 0
        self.ids = []
        self.nodes = None
        self.adj_mat = None

        if graphs != None:
            self.add_graphs(graphs)

    def add_graphs(self, graphs):

        if type(graphs) == ContextualGraph:
            self.eta += len(graphs.nodes)
            self.cgraphs.append(graphs)
        elif type(graphs) == list:
            if len(graphs) == 0:
                return
            self.cgraphs.extend(graphs)
            for graph in graphs:
                self.eta += len(graph.nodes)
            
        self.cgraphs.sort(key=lambda r: r.length, reverse=True)
        self.ndocs = len(self.cgraphs)
        self.build()

    def build(self):
        if len(self.cgraphs) == 0:
            return

        self.nodes = None
        self.adj_mat = None

        basegraph = self.cgraphs[0]
        self.add_base_graph(basegraph)

        for i in range(1, len(self.cgraphs)):
            self.add_graph(self.cgraphs[i])

    def add_base_graph(self, graph):
        self.adj_mat = graph.adj_mat

        for i,node in enumerate(graph.nodes):
            node.id = i*self.eta
            self.ids.append(node.id)
            self.nodes.append(node)
            self.length += 1

    def add_graph(self, graph):
        parents = graph.parents()
        igparents = dict()

        for i,node in enumerate(graph.nodes):
            p = parents[i]

            # Parent of i precedes i
            if p < i: 
                if not self.similarity(node, parent=igparents[p]):
                    self.add_node(node)
                    igparents[i] = len(self.nodes)-1
                else:
                    igparents[i] = self.get_igparent(node)
            else:
                if not self.similarity(node):
                    self.add_node(node)
                    igparents[i] = len(self.nodes)-1
                else:
                    igparents[i] = self.get_igparent(node)

    def add_node(self, node):
        self.adj_mat = np.pad(self.adj_mat, ((0,0),(1,1)), mode='constant', constant_values=0 )
        
        self.length += 1
        p = self.length-1

        max_sim = 0
        max_simarg = -1

        for q in range(p):
            sim = self.nodes[q].similarity(node)
            if sim > 0.001:
                if sim > max_sim:
                    max_sim = sim
                    max_simarg = self.nodes[q].id
                self.adj_mat[p][q] = sim
                self.adj_mat[q][p] = sim

        node.id = self.get_id(max_simarg)
        self.ids.append(node.id)
        self.nodes.append(node)

    def similarity(self, node, parent=-1, threshold=0.7):
        if parent == -1:
            for inode in self.nodes:
                if inode.similarity(node) > threshold:
                    return True
        else:
            for i,inode in enumerate(self.nodes):
                if self.adj_mat[parent][i] != 0:
                    if inode.similarity(node) > threshold:
                        return True

        return False

    def get_id(self, base_id):
        while base_id in self.ids:
            base_id += 1
        return base_id

    def get_igparent(self, node):
        sims = [ node.similarity(x) for x in self.nodes ]
        max_sim = np.argmax(np.asarray(sims))
        return max_sim

    def nodes_sorted(self):
        
        sorted_nodes = []
        p = 0
        cnt = 0
        while cnt != len(self.nodes):
            interval_nodes = [ node for node in self.nodes if node.id >= p and node.id < p+self.eta]
            base_node = interval_nodes[0]
            interval_nodes.sort(key=lambda r: r.similarity(base_node), reverse=True)
            sorted_nodes.append(interval_nodes)
            cnt += len(interval_nodes)
            p += 1

        return sorted_nodes

    def get_node_index(self, gnode):
        for i,node in enumerate(self.nodes):
            if node == gnode:
                return i
        return -1

    def save(self):
        text_filepath = join(IGRAPH_DIR, self.filename)

        with open(text_filepath, "w") as f:
            for graph in self.cgraphs:
                f.write(graph.filename)
                f.write("\n")

    def save_json(self):
        pass

    @staticmethod
    def load(file):
        filepath = join(IGRAPH_DIR, file)

        cgraphs = []
        with open(filepath, "r") as f:
            for line in f:
                graph = ContextualGraph.load(line)
                cgraphs.append(graph)

        vocabs = dict()
        vocabs["files"], vocabs["inv_files"] = load_vocabs(join(VOCAB_DIR, "files.txt"))
        vocabs["words"], vocabs["inv_words"] = load_vocabs(join(VOCAB_DIR, "words.txt"))    

        return IntegratedGraph(file, vocabs, cgraphs)

    @staticmethod
    def load_json(file):
        pass

    def __len__(self):
        return len(self.nodes)

    # def traverse(self, root_index=0, beam=3, depth=4):
    #     vis = [False] * len(self.nodes)
    #     queue = [(root_index, -1)]
    #     vis[root_index] = True

    #     while len(queue) != 0:
    #         yield queue[0]
    #         pid = queue[0][0]


    #         pass