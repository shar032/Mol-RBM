""" Train RBM using stochastic steepest ascent """
import pandas as pd
import tensorflow as tf

from Chem_utils import Chem_utils
from RBM_ssa import RBM
from RBM_utils import get_char_idx_mappers, RBM_process

data = pd.read_csv('train_set.csv')
smiles_list = data['SMILES'] # Around 11,800 SMILES strings

max_SMILES_len = 110 # Maximum SMILES length (including padding character)
smiles_list = list(filter(lambda smile : len(smile) <= max_SMILES_len, smiles_list))

# Check SMILES
chem_utils = Chem_utils()
smiles_list = list(filter(lambda smile: chem_utils.check_SMILES, smiles_list))

# Get mappers
index_to_char, char_to_index, vocab_unique_chars = get_char_idx_mappers(smiles_list)

# Define RBM hyper parameters
n_visible = max_SMILES_len*len(index_to_char)
n_hidden = round(0.68*n_visible)
base_lr = tf.constant(0.005, tf.float32)
max_lr = tf.constant(0.009, tf.float32)
num_epochs = 75
data_interval_stats = 100
step_size = 10

# Instantiate RBM_process class
rbm_process = RBM_process(max_SMILES_len, index_to_char, char_to_index, vocab_unique_chars, n_hidden)

# Process input data
input_SMILES_visible_vectors = tf.convert_to_tensor([rbm_process.one_hot_SMILES_to_visible_vector(smile) for smile
                                                     in smiles_list if len(rbm_process.one_hot_SMILES_to_visible_vector(smile)) > 0])

input_SMILES_visible_vectors = tf.cast(input_SMILES_visible_vectors, tf.float32)

# Instantiate RBM 
rbm = RBM(n_visible, max_SMILES_len, index_to_char, char_to_index, vocab_unique_chars, n_hidden)

# Train RBM
rbm.train(input_SMILES_visible_vectors, base_lr, max_lr, num_epochs, step_size, data_interval_stats)


