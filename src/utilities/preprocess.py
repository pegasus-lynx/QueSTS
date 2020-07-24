import numpy as np
import nltk
import re
import os

from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def split_doc(file):
    raw_sentences = []
    with open(file, "r") as f:
        for line in f:
            sentences = re.split('\. ', line)
            for sentence in sentences:
                sentence = sentence.replace('\xa0', ' ')
                raw_sentences.append(sentence)

    return raw_sentences

def process_doc(raw_sentences, lemmatizer = WordNetLemmatizer()):
    processed = []
    for sentence in raw_sentences:
        words = [w.lower() for w in word_tokenize(sentence)]
        
        tokens = []
        for word in words:
            tokens.extend(re.split('[^a-zA-Z]', word))
        
        lemmatized_tokens = [
            lemmatizer.lemmatize(token) for token in tokens 
            if not token in stopwords.words('english')]
        
        processed.append(list(filter(lambda token: len(token), lemmatized_tokens)))
    
    return processed

def tokenize_query(query, lemmatizer = WordNetLemmatizer()):
    
    tokens = []
    words = query.strip().split()
    
    for word in words:
        tokens.append(re.split('[^a-zA-Z]', word.lower()))

    lemmatized_tokens = [
        lemmatizer.lemmatize(token) for token in tokens 
        if not token in stopwords.words('english')]

    return lemmatized_tokens

def make_vocab(processed_tokens):
    words = set()
    for tokens in processed_tokens:
        words.update(tokens)
    return words

def save_vocabs(file, words):
    with open(file, "w") as f:
        for word in words:
            f.write(word+"\n")

def load_vocabs(file):
    vocabs = dict()
    inv_vocabs = dict()
    
    with open(file,"r") as f:
        for i,line in enumerate(f):
            line = line.strip()
            vocabs[line] = i
            inv_vocabs[i] = line

    return (vocabs, inv_vocabs)

def save_processed(file, processed, raw_sentences):
    with open(file, "w") as f:
        for text, tokens in zip(raw_sentences, processed):
            f.write(text + "\n")
            f.write(" ".join(tokens) + "\n")
            f.write("\n")

def get_files(dir_name):
    return [ f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]

def save_filenames(file, filenames):
    with open(file, "w") as f:
        for name in filenames:
            f.write(name)
            f.write("\n")

def read_datafile(file):
    raw = []
    processed = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) != 0:
                if len(raw) != len(processed):
                    processed.append(line.split(" "))
                else:
                    raw.append(line)

    return (raw, processed)
