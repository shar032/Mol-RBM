""" Restriced Boltzmann Machine in TF 2.0 """
import numpy as np
import pandas as pd
import tensorflow as tf

from Learning_rates import LR
from RBM_utils import get_char_idx_mappers, get_val_train_sets, RBM_process


# FIRST CREATE RBM MODEL USING REGULAR FUNCTIONAL PROGRAMMING AND THEN MAKE IT OBJECT ORIENTED

data = pd.read_csv('train_set.csv')
smiles_list = data['SMILES'] # has 11,800 SMILES molecules, all of length 110 including padding spaces characters

max_smiles_len = 110

smiles_list = list(filter(lambda smile : len(smile) <= max_smiles_len, smiles_list)) # list remains the same
# but just filter to ensure and cross check

# get character and index mappers for SMILES 
index_to_char, char_to_index, vocab_unique_chars = get_char_idx_mappers(smiles_list)


rbm_process = RBM_process(max_smiles_len, index_to_char, char_to_index, vocab_unique_chars, n_hidden)

# Define constants and variables
n_visible = max_smiles_len*len(index_to_char)
n_hidden = round(0.68*n_visible)
lr = tf.constant(0.09, tf.float32)


# Define energy function
@tf.function
def gibbs_energy(b_v, v, b_h, h, W):
    b_v = b_v[0]
    b_h = b_h[0]
    h = h[0]
    v = v[0]
    h_t = h
    f = tf.tensordot(b_v, v, 1).numpy()
    s = tf.tensordot(b_h, h, 1).numpy()
    t = tf.tensordot(v, tf.matmul(W, tf.reshape(h_t, [n_hidden, -1])), 1).numpy()
    return (f + s + t)[0]


input_SMILES_visible_vectors = tf.convert_to_tensor([rbm_process.one_hot_SMILES_to_visible_vector(smile) for smile
                                                     in smiles_list[0:5000] if len(rbm_process.one_hot_SMILES_to_visible_vector(smile)) > 0])
    
input_SMILES_visible_vectors = tf.cast(input_SMILES_visible_vectors, tf.float32)

# Define and initialise the weights and biases for both layers and other functions 
W = tf.Variable(tf.compat.v2.random.normal([n_visible, n_hidden], 0.01, dtype = tf.float32), name = "W")
b_h = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name = "b_h"))
b_v = tf.Variable(tf.zeros([1, n_visible], tf.float32, name = "b_v"))
num_epochs = 10
batch_size = 28

@tf.function
def sample(probs):
    return tf.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1, dtype = tf.float32))

@tf.function
def gibbs_step(x_k):
    h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + b_h))
    x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v))
    return h_k

@tf.function
def gibbs_sample(k, x_k):
    for i in range(k):
        x_out = gibbs_step(x_k)
    return x_out

@tf.function
def train_RBM_using_CD(x):
    h = sample(tf.sigmoid(tf.matmul(x, W) + b_h)) 
    x_ = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))
    x_probs = tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v)
    h_ = sample(tf.sigmoid(tf.matmul(x_, W) + b_h))
    
    dW  = tf.multiply(lr, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_), h_)))
    db_v = tf.multiply(lr, tf.reduce_sum(tf.subtract(x, x_), 0, True)) 
    db_h = tf.multiply(lr, tf.reduce_sum(tf.subtract(h, h_), 0, True)) 
    
    W.assign_add(dW)
    b_v.assign_add(db_v)
    b_h.assign_add(db_h)
    
    return [W, b_v, b_h, h, x_, h_, x_probs]

'''Setup training loop for stochastic steepest ascent using contrastive divergence '''
[gibbs_energies, cosine_similarities, similarities_each_unit, reconstruction_loss] = [[] for _ in range(4)]

for epoch in range(1, num_epochs + 1):
    


    input_SMILES_visible_vectors = tf.random.shuffle(input_SMILES_visible_vectors)
    for i in range(len(input_SMILES_visible_vectors)):
        
        params = train_RBM_using_CD(tf.reshape(input_SMILES_visible_vectors[i], [-1, n_visible]))
        if epoch == 1 and i % 100 == 0:
            gibbs_energies.append((epoch, i, second_gibbs_trial(params[1], input_SMILES_visible_vectors[i], params[2], params[3], params[0])))
            reconstruction_loss.append((epoch, i, rbm_process.reconstruction_loss(input_SMILES_visible_vectors[i].numpy(), params[6][0].numpy())))
            
            
            print("Epoch : {}, Data point : {}".format(epoch, i))
            print("Gibbs Energy : {}".format(second_gibbs_trial(params[1], input_SMILES_visible_vectors[i], params[2], params[3], params[0])))
            print("Reconstruction Loss: {}\n".format(rbm_process.reconstruction_loss(input_SMILES_visible_vectors[i].numpy(), params[6][0].numpy())))
            print('-'*50)
            
