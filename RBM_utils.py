""" Functions and classes for RBM """
import numpy as np
import random
import tensorflow as tf

from scipy.spatial import distance

def get_char_idx_mappers(SMILES_list):
    """ Functions mapping SMILES character to token/index identifier
    and back
    
    Args:
        SMILES_list (list): list of SMILES strings in dataset
    
    Returns
        (tuple): index to character and character to index mapper
    """
    # Get vocabulary of unique characters set
    vocab_unique_chars = set()
    for s in SMILES_list:
        for char in s:
            vocab_unique_chars.add(char)
    # Map index identifier to SMILES character
    index_to_char = np.array(sorted(vocab_unique_chars))
    # Map SMILES character to index identifier
    char_to_index = {y:x for (x, y) in enumerate(index_to_char)}
    
    return index_to_char, char_to_index, vocab_unique_chars

def get_val_train_sets(dataset, split):
    """ Split dataset of SMILES strings into training and validation sets
    
    Args:
        dataset (list): list of SMILES strings in dataset
        split (float): fraction to split into validation set
    
    Returns:
        (tuple): trainset and validation set
    """
    dataset_len = len(dataset)
    val_idxs = random.sample(range(dataset_len), round(split*dataset_len))
    val_set = [dataset[i] for i in val_idxs]
    train_set = list(filter(lambda s: s not in val_set, dataset))
    return train_set, val_set


class RBM_process:
    """ RBM processing functions """
    def __init__(self, max_SMILES_len, index_to_char, 
                 char_to_index, vocab_unique_chars, n_hidden):
        
        self.max_SMILES_len = max_SMILES_len
        self.index_to_char = index_to_char
        self.char_to_index = char_to_index
        self.vocab_unique_chars = vocab_unique_chars
        self.vocab_len = len(vocab_unique_chars)
        self.n_hidden = n_hidden
        
    def one_hot_SMILES_to_visible_vector(self, smile):
        """ One hot encode SMILES string to initialize RBM visible vector
        
        Args:
            smile (str): SMILES string
        
        Returns:
            (numpy.ndarray): one hot encoded vector representing SMILES, 
            including one hot codes for padding character to equalize lengths 
            up to maximum length
        """
        out_vec = np.empty(shape = [0,0])
        for char in smile:
            block = np.zeros(len(self.index_to_char))
            block[self.char_to_index[char]] = 1.0
            new = np.concatenate((out_vec, block), axis = None)
            out_vec = new
        return out_vec
    
    def probability_visible_vector_to_SMILES(self, out_vector):
        """ Convert RBM visible vector to SMILES
        
        Args:
            out_vector (numpy.ndarray): reconstructed RBM visible vector after
            multiple Gibbs steps (array of sigmoid activated float32 probability
            values)
        
        Returns:
            (str): SMILES string representation of visible vector
        """
        max_prob_indexes_each_block = list()
        for i in range(len(out_vector) - len(self.index_to_char)):
            if i % len(self.index_to_char) == 0:
                for j in range(len(self.index_to_char)):
                    if out_vector[i:i+len(self.index_to_char)][j] == max(out_vector[i:i+len(self.index_to_char)]):
                        max_prob_indexes_each_block.append(j)
        return ' '.join([self.index_to_char[ind] for ind in max_prob_indexes_each_block]).replace(' ', "")             
      
    def reconstruction_loss(self, in_visible, updated_visible):
        """ Reconstruction (MSE) loss of updated visible vector (with probabilities)
        against input binary visible vector (recommended by Hinton et. al)
        - RBM visible vector is not a probability distribution
        
        Args:
            in_visible (numpy.ndarray): initial visible vector
            updated_visible (numpy.ndarray): updated visible vector after
            multiple Gibbs steps
            
        Returns:
            (float): reconstruction loss
        """
        return np.mean((in_visible - updated_visible)**2)
    
    def random_SMILES_to_visible_vector(self):
        """ Convert random invalid SMILES string to visible vector as an
        initialization method
        
        Returns:
            (numpy.ndarray): visible vector for RBM initialization
        """
        vec = np.empty(shape = [0,0])
        for _ in range(self.max_SMILES_len):
            block = np.zeros(self.vocab_len)
            block[random.randint(0, self.vocab_len-1)] = 1.0
            new = np.concatenate((vec,block), axis=None)
            vec = new
        return vec
        
    def cosine_similarity(self, vec1, vec2):
        """ Cosine similarity between two vectors
        
        Args:
            vec1 (numpy.ndarray)
            vec2 (numpy.ndarray)
            
        Returns:
            (float): cosine similarity between vectors
        """
        return 1 - distance.cosine(vec1, vec2)
    
    def euclidean_distance(self, vec1, vec2):
        """ Euclidean distance between points
        (useful in norm calculations for visible and molecule feature 
        vectors for relative comparisons)
        
        Args:
            vec1 (numpy.ndarray)
            vec2 (numpy.ndarray)
        
        Returns:
            (float): euclidean distance
        """
        return distance.euclidean(vec1, vec2)
    
    def gen_SMILES_characters_added(self, seed_SMILES, generated_SMILES):
        """ Characters added to generated SMILES
        
        Args:
            seed_SMILES (str): RBM initializing SMILES string
            generated_SMILES (str): SMILES representation of updated visible
            vector after multiple Gibbs steps
        
        Returns:
            (int): number of new characters introduced in generated SMILES
            string
        """
        seed_set = set(seed_SMILES)
        gen_set = set(generated_SMILES)
        return len(gen_set.difference(seed_set))
    
    
    @tf.function
    def get_gibbs_energy(self, b_v, v, b_h, h, W):
        """ Gibbs ebergy of an instance of RBM configuration """
        f = tf.tensordot(b_v[0], v, 1)
        s = tf.tensordot(b_h[0], h[0], 1)
        t = tf.tensordot(v, tf.matmul(W, tf.reshape(h, [self.n_hidden, -1])), 1)
        return (f + s + t)[0]
        
    
    @tf.function
    def sample(self, probs):
        """ Stochastically sample binary vector from probabilities """
        return tf.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1, dtype = tf.float32))
    


