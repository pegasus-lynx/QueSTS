import numpy as np
from os.path import join

from utilities.config import VOCAB_DIR, CGRAPH_DIR, IGRAPH_DIR
from utilities.preprocess import load_vocabs
from utilities.utils import get_isf, get_query_wts, differs
from utilities.utils import query_similarity_weights

from integrated_graph import IntegratedGraph
from contextual_graph import ContextualGraph
from contextual_tree import CTree

from graph_node import GraphNode, TreeNode
from query import Query


class QueSTS(object):

    def __init__(self, igraph_file):
        self.vocabs = self.load_vocabs()
        
        files = self.vocabs["files"].keys()
        self.isf = get_isf(files,self.vocabs)
        
        self.igraph = IntegratedGraph.load(igraph_file)
        self.queries = []

        # Persistent objects for handling query processing
        self.ctrees = None
        self.node_wts = None

    def load_vocabs(self):
        vocabs = dict()
        vocabs["files"], vocabs["inv_files"] = load_vocabs(join(VOCAB_DIR, "files.txt"))
        vocabs["words"], vocabs["inv_words"] = load_vocabs(join(VOCAB_DIR, "words.txt"))
        return vocabs

    def process_query(self, text):
        query = Query(text)
        query.set_weights(self.vocabs, self.isf)
        self.queries.append(query)

        node_wts = self.get_node_weights(query)

        for i,token in enumerate(query.tokens):
            ctree_row = []
            for j,node in enumerate(self.igraph.nodes):
                ctree = CTree(node)
                ctree.construct()

    def get_node_weights(self, query, bias_factor=0.5):
        
        sim_ss = self.igraph.adj_mat
        sim_qn = query_similarity_weights(query, self.igraph, self.vocabs, self.isf)
        
        sum_sim_ss = np.sum(sim_ss, axis=0, keepdims=True)
        sum_sim_qn = np.sum(sim_qn, axis=1, keepdims=True)

        nwts = sim_qn / sum_sim_qn
        nwts_next = nwts

        iterate = True
        mask = sim_ss > 0

        while iterate:
            temp = nwts_next

            nwts_next =  bias_factor * ( sim_qn / sum_sim_qn )
            nwts_next += (1-bias_factor) * np.sum(sim_ss / sum_sim_ss, axis=0, keepdims=True) * nwts

            nwts = temp
            iterate = differs(nwts, nwts_next)

        return nwts
