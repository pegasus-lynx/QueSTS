from graph_node import TreeNode

class SGraph(object):

    def __init__(self,node):
        self.gnode = node
        self.rnode = None

    def score(self):
        pass

    def merge(self, ctrees):
        self.rnode = TreeNode(self.gnode)
        for ctree in ctrees:
            self.add_tree(ctree)

    def add_tree(self, ctree):
        queue = [self.rnode]
        nqueue = [ctree.rnode]

        while len(nqueue) != 0:
            curr = queue[0]
            ncurr = nqueue[0]

            cids = [x.node.id for x in curr.childs]
            for child in ncurr.childs:
                if child.node.id not in cids:
                    curr.childs.add(child)
                    queue.append(child)
                    nqueue.append(child)
                else:
                    for tnode in curr.child:
                        if tnode.node.id == child.node.id:
                            queue.append(tnode)
                            nqueue.append(child)
                            break
                        