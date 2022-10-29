""" Restricted Boltzmann Machine (training using mini-batch 
    steepest ascent and contrastive divergence) """

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
    def _get_grads(self, x, lr):
        x = tf.reshape(x, [-1, self.n_visible])
        h = self.sample(tf.sigmoid(tf.matmul(x, self.W) + self.b_h)) 
        x_ = self.sample(tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b_v))
        x_probs = tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b_v)
        h_ = self.sample(tf.sigmoid(tf.matmul(x_, self.W) + self.b_h))
        
        dWx  = tf.multiply(lr, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_), h_)))
        db_vx = tf.multiply(lr, tf.reduce_sum(tf.subtract(x, x_), 0, True)) 
        db_hx = tf.multiply(lr, tf.reduce_sum(tf.subtract(h, h_), 0, True)) 
        return [dWx, db_vx, db_hx, h, x_, h_, x_probs]
    
    @tf.function
    def _train_RBM_batch(self, x_vec, lr):
        
        all_grads_lists = [self._get_grads(x_vec[i], lr) for i in range(x_vec.shape[0])]
    
        [dWs, db_vs, db_hs, hs, x_s, h_s, x_probs_s] = [[all_grads_lists[i][j] for i in range(len(all_grads_lists))] for j in range(7)]
        
        self.W.assign_add(tf.reduce_mean(dWs, 0))
        self.b_v.assign_add(tf.reduce_mean(db_vs, 0))
        self.b_h.assign_add(tf.reduce_mean(db_hs, 0))
        
        return {'W': self.W, 'b_v': self.b_v, 'b_h': self.b_h, 'h': hs[-1]}
    
    @tf.function
    def train(self, num_epochs, base_lr, max_lr, batch_size, input_SMILES_visible_vectors, 
              step_size, data_interval_stats):
        # To implement: early stopping to prevent overfitting by comparing 
        # train and validation set mean Gibbs energy and breaking from loop
        # when validation energy exceeds train energy by a threshold
        
        for epoch in range(1, num_epochs + 1):
            
            lr = self.cyclical_triangular(base_lr, max_lr, epoch, step_size)
            input_SMILES_visible_vectors = tf.random.shuffle(input_SMILES_visible_vectors)
            
            for i in range(input_SMILES_visible_vectors.shape[0] - batch_size):
                
                if i % batch_size == 0:
                    
                    batch = input_SMILES_visible_vectors[i:i+batch_size]
                    params = self._train_RBM_batch(batch, lr)
                    
                    if i % data_interval_stats == 0:
                        print("Epoch : {}, Data point : {}".format(epoch, i))
                        print("Gibbs Energy : {}".format(self.get_gibbs_energy(params['b_v'], batch[-1], params['b_h'], params['h'], params['W'])))
                        print('-'*50)
    
    
    
    