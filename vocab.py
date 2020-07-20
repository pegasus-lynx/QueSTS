from utilities.config import DATA_DIR, PROC_DIR, VOCAB_DIR
from utilities.preprocess import get_files, save_filenames, get_vocabs,
                                 save_vocabs

from os.path import join

WORDS = VOCAB_DIR + "words.txt"
FILES = VOCAB_DIR + "files.txt"

if __name__ == "__main__":

    files = get_files(DATA_DIR)
    save_filenames(FILES, files)

    vocabs = set()

    for file in files:
        filepath = join(DATA_DIR, file)
        raw = split_doc(filepath)

        processed = process_doc(raw)

        vocabs.update(get_vocabs(processed))

    save_vocabs(WORDS, list(vocabs))
