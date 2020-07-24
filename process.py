from utilities.config import DATA_DIR, PROC_DIR, VOCAB_DIR
from utilities.preprocess import get_files, save_filenames, get_vocabs
from utilities.preprocess import save_vocabs, split_doc, process_doc, save_processed

from os.path import join

WORDS = VOCAB_DIR + "words.txt"
FILES = VOCAB_DIR + "files.txt"

if __name__ == "__main__":

    files = get_files(DATA_DIR)
    save_filenames(FILES, files)

    vocabs = set()
    vocabs.add("<oov>")

    for file in files:
        filepath = join(DATA_DIR, file)
        
        raw = split_doc(filepath)
        processed = process_doc(raw)

        processed_file = join(PROC_DIR, file)
        save_processed(processed_file, processed, raw)

        vocabs.update(get_vocabs(processed))

    save_vocabs(WORDS, list(vocabs))
