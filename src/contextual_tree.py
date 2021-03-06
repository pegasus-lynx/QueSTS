import numpy as np
from graph_node import TreeNode

class CTree(object):

    def __init__(self, query_token, node_wts, igraph_node, beam_width=3, depth=4):
        self.beam = beam_width
        self.depth = depth

        self.alpha = 1
        self.beta = 1

        self.qterm = query_token
        self.node_wts = node_wts
        
        self.gnode = igraph_node
        self.nindex = 0

        self.rnode = None
        self.weight = None
        self.score = None

    def construct(self, igraph):
        queue = []
        vis = [False] * len(igraph)
        lev = [-1] * len(igraph)
        par = [-1] * len(igraph)

        self.nindex = igraph.get_node_index(self.gnode)
        self.score = 0

        queue.append(self.nindex)
        vis[self.nindex] = True
        lev[self.nindex] = 0

        bck = []

        self.score += self.beta * self.node_wts[self.nindex]

        while len(queue) != 0:
            curr = queue[0]

            if igraph.nodes[curr].has(self.qterm):
                bck.append(curr)
                break

            if lev[curr] > self.depth:
                break

            node_prominence = list(zip(self.get_node_prominence(igraph), range(len(igraph))))
            node_prominence.sort(key=lambda r: r[0], reverse=True)

            cnt = 0
            for wt, ix in node_prominence:
                if wt != 0:
                    if cnt == self.beam:
                        vis[ix] = True
                        continue
                    if not vis[ix]:
                        vis[ix] = True
                        par[ix] = curr
                        lev[ix] = lev[curr] + 1
                        cnt += 1
                        queue.append(ix)

        
        if len(bck) != 0:
            while par[bck[-1]] != -1:
                bck.append(par[bck[-1]])

            bck.reverse()
            self.add_nodes(bck, igraph)

    def traverse(self):
        pass

    def add_nodes(self, bck, igraph):
        vis = [False] * len(igraph)

        vis[bck[0]] = True
        curr = TreeNode(self.gnode)
        self.rnode = curr

        next_id = -1

        for lev, i in enumerate(range(len(bck)-1)):
            for p in range(len(igraph)):
                if igraph.adj_mat[i][p] != 0:
                    if not vis[p]:
                        vis[p] = True
                        tnode = TreeNode(igraph.nodes[p], curr)
                        curr.childs.append(tnode)

                        self.score += self.alpha*(igraph[i][p]) / np.sqrt(lev+1)
                        self.score += self.beta*(self.node_wts[p])/np.sqrt(lev+1)
                        
                        if p == bck[i+1]:
                            next_id = tnode.node.id

            curr.childs.sort(key=lambda r:r.node.id)

            for node in curr.childs:
                if node.node.id == next_id:
                    curr = node

    def get_node_prominence(self, igraph, node_wts):
        adj_list = igraph.adj_mat[self.nindex]
        mask = adj_list != 0

        return  (self.alpha*(adj_list) + self.beta*(mask * self.node_wts)).tolist()
