from utilities.config import PROC_DIR, VOCAB_DIR, DATA_DIR

import os
from os.path import isfile, join

if __name__ == "__main__":
    
    # Setting up the directories

    if not os.path.exists(PROC_DIR):
        os.makedirs(PROC_DIR)
        os.makedirs(VOCAB_DIR)
        print("PROC_DIR : " + PROC_DIR + " created")
        print("VOCAB_DIR : " + VOCAB_DIR + " created")
    else:
        print("PROC_DIR : " + PROC_DIR + " exists")


    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)
        print("VOCAB_DIR : " + VOCAB_DIR + " created")
    else:
        print("VOCAB_DIR : " + VOCAB_DIR + " exists")


    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("DATA_DIR : " + DATA_DIR + " created")
    else:
        print("DATA_DIR : " + DATA_DIR + " exists")