import string
import tensorflow as tf
import numpy as np
import sys
from sklearn.preprocessing import normalize

window_size = 3
text_file = "C:/Users/sanketn/Documents/IU/Deep Learning/coco_val_sentences.txt"

def preprocess_word(word):
    word = word.lower().strip()
    for punc in string.punctuation:
        word = word.replace(punc,"")
    return word

word_training_pairs = []

all_sentences = open(text_file).readlines()

for sentence in all_sentences:
    sentence_split = [preprocess_word(word) for word in sentence.split()]
    for i,target in enumerate(sentence_split):
        for j in range(1,4):
            if not i+j >= len(sentence_split):
                word_training_pairs.append((target,sentence_split[i+j]))
            if not i-j >= len(sentence_split):
                word_training_pairs.append((target,sentence_split[i-j]))    

id2word = list(set([pair[0] for pair in word_training_pairs]))
word2id = {w:i for i,w in enumerate(id2word)}

word_training_pairs = [(word2id[pair[0]],word2id[pair[1]]) for pair in word_training_pairs]
print(len(word_training_pairs))