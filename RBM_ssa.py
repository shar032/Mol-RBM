""" Restricted Boltzmann Machine (training using stochastic 
    steepest ascent and contrastive divergance) """

import tensorflow as tf

from Learning_rates import LR
from RBM_utils import RBM_process


class RBM(RBM_process, LR):
    
    def __init__(self, n_visible, max_SMILES_len, index_to_char, char_to_index, 
                 vocab_unique_chars, n_hidden):
        RBM_process.__init__(self, max_SMILES_len, index_to_char, char_to_index, 
                             vocab_unique_chars, n_hidden)
        LR.__init__(self)
        
        self.n_visible = n_visible
        self.W = tf.Variable(tf.compat.v2.random.normal([n_visible, n_hidden], 0.01, dtype = tf.float32), name = "W")
        self.b_h = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name = "b_h"))
        self.b_v = tf.Variable(tf.zeros([1, n_visible], tf.float32, name = "b_v"))
    
    @tf.function
    def _gibbs_step(self, x_k):
        h_k = self.sample(tf.sigmoid(tf.matmul(x_k, self.W) + self.b_h))
        x_k = self.sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(self.W)) + self.b_v))
        return h_k
    
    @tf.function
    def _gibbs_sample(self, k, x_k):
        for i in range(k):
            x_out = self._gibbs_step(x_k)
        return x_out
    
    
    @tf.function
    def train_RBM_with_CD(self, x, lr):
        h = self.sample(tf.sigmoid(tf.matmul(x, self.W) + self.b_h)) 
        x_ = self.sample(tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b_v))
        h_ = self.sample(tf.sigmoid(tf.matmul(x_, self.W) + self.b_h))
        
        dW  = tf.multiply(lr, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_), h_)))
        db_v = tf.multiply(lr, tf.reduce_sum(tf.subtract(x, x_), 0, True)) 
        db_h = tf.multiply(lr, tf.reduce_sum(tf.subtract(h, h_), 0, True)) 
        
        self.W.assign_add(dW)
        self.b_v.assign_add(db_v)
        self.b_h.assign_add(db_h)
        
        return {'W': self.W, 'b_v': self.b_v, 'b_h': self.b_h, 'h': h}

    @tf.function
    def train(self, input_SMILES_visible_vectors, base_lr, max_lr,
              num_epochs, step_size, data_interval_stats):
        # To implement: early stopping to prevent overfitting by comparing 
        # train and validation set mean Gibbs energy and breaking from loop
        # when validation energy exceeds train energy by a threshold 
        
        for epoch in range(1, num_epochs + 1):
            
            lr = self.cyclical_triangular(base_lr, max_lr, epoch, step_size)
            input_SMILES_visible_vectors = tf.random.shuffle(input_SMILES_visible_vectors)
            
            for i in range(input_SMILES_visible_vectors.shape[0]):
                params = self.train_RBM_with_CD(tf.reshape(input_SMILES_visible_vectors[i], [-1, self.n_visible]), lr)
                
                if i % data_interval_stats == 0:
                    
                    print("Epoch : {}, Data point : {}".format(epoch, i))
                    print("Gibbs Energy : {}".format(self.get_gibbs_energy(params['b_v'], input_SMILES_visible_vectors[i], params['b_h'], params['h'], params['W'])))
                    print('-'*50)
                    
    
   