import string
import tensorflow as tf
import numpy as np
import sys
from sklearn.preprocessing import normalize

window_size = 3
text_file = "C:/Users/sanketn/Documents/IU/Deep Learning/aethism.txt"

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

tf.reset_default_graph()
vocab_size = len(id2word)
embed_size = 50
batch_size = 1024

#Variables
word_embedding = tf.Variable(tf.random_uniform([vocab_size,embed_size]))
softmax_weight = tf.Variable(tf.random_uniform([embed_size,vocab_size]))

#Input
train_pairs = tf.placeholder(tf.int32,shape=[None,2])
train_inputs = train_pairs[:,0]
train_outputs = train_pairs[:,1]

#Model
word_embed = tf.nn.embedding_lookup(word_embedding,train_inputs)
prediction = tf.matmul(word_embed,softmax_weight)

#loss
loss = tf.losses.sparse_softmax_cross_entropy(train_outputs,prediction)

#optimizer
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

loss_hist = []
iter=0
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,len(word_training_pairs),batch_size):
        batch_data = word_training_pairs[i:i+batch_size]
        _,loss_value = sess.run([optimizer,loss],feed_dict={train_pairs:batch_data})
        loss_hist.append(loss_hist)
        sys.stdout.write("\r%d %f"%(iter,loss_value))
        sys.stdout.flush()
        iter+=1
    W = sess.run(word_embedding)
    
np.save("W.npy",W)
    
f = open("words.txt",'w')
for w in id2word:
    f.write(w+"\n")
f.close()

W = np.load("W.npy")
id2word = [w.strip() for w in open("words.txt")]
word2id = {w:i for i,w in enumerate(id2word)}

W = normalize(W)    

def print_similar_words(word,topk=10):
    word_vec = W[word2id[word]]
    sim = np.dot(W,word_vec)
    sim_idx = np.argsort(sim)[::-1]
    
    for idx in sim_idx[1:topk+1]:
        print(id2word[idx]) 
        
print_similar_words("universe")          