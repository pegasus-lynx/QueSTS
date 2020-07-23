from utilities.config import DATA_DIR, PROC_DIR, VOCAB_DIR
from utilities.preprocess import get_files, read_datafile, load_vocabs
from contextual_graph import ContextualGraph
from integrated_graph import IntegratedGraph

from os.path import join

if __name__ == "__main__":

    vocabs = dict()
    vocabs["files"], vocabs["inv_files"] = load_vocabs(join(VOCAB_DIR, "files.txt"))
    vocabs["words"], vocabs["inv_words"] = load_vocabs(join(VOCAB_DIR, "words.txt"))    

    files = vocabs["files"].keys()
    cgraphs = []

    for file in files:        
        cg = ContextualGraph(file)
        cg.build(vocabs)
        cgraphs.append(cg)
        cg.save()

    igraph = IntegratedGraph("igraph.txt", vocabs, cgraphs)
    igraph.save()