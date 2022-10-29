""" Learning rate functions for NN models """
import numpy as np

class LR:
    """ Learning rate decay types """
    
    def constant_lr(self, lr: float) -> float:
        
        """ Constant learning rate
        Args:
            lr (float): current learning rate
        Returns:
            (float): updated learning rate 
        """
        return lr
    
    def time_decay(self, lr: float, epoch: int, decay_rate: float) -> float:
        
        """ Time based learning rate decay
        """
        return lr/(1 + decay_rate*epoch)
    
    def step_decay(self, lr: float, epoch_interval: int, epoch: int, 
                   drop_frac: float) -> float:
        
        """ Step learning rate decay
        """
        if epoch % epoch_interval == 0:
            return lr * drop_frac
        return lr
    
    def exponential_decay1(self, lr0: float, epoch: int, k: float) -> float:
        
        """ Exponential learning rate decay 1
        """
        return lr0 * pow(k, epoch)
    
    def exponential_decay2(self, lr0: float, epoch: int, k: float) -> float:
        
        """ Exponential learning rate decay 2
        """
        return lr0 * np.exp(-k * epoch)
    
    def cyclical_triangular(self, base_lr: float, max_lr: float, epoch: int,
                            step_size: int) -> float:
        
        """ Step-wise cyclical learning rate decay with triangular policy (oscillate 
        between base and max value bounds in step sizes)
        """
        half_cycle = step_size // 2
        if epoch % step_size < half_cycle:
            return base_lr
        return max_lr
        
    def cyclical_triangular2(self, base_lr: float, max_lr: float, epoch: int, 
                             step_size: int, num_epochs: int) -> float:
        
        """ Step-wise cyclical learning rate decay with triangular 2 policy (oscillate 
        between base and max value bounds in step sizes with max bound halved 
        after each cycle)
        """
        half_cycle = step_size // 2
        if max_lr > base_lr and epoch in range(0, num_epochs, step_size):
            max_lr *= 0.5
            
        if epoch % step_size < half_cycle:
            return base_lr
        return max_lr
    
