# ImaginaryTimeEvolution.py

'''This code evolve the density matrix in the temperature space so that
   we can obtain the thermal density matrix
'''

import numpy as np
import tensorflow as tf
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Global variables 

X = None
y = None

layers_dnn = []
weights_dnn = []
bias_dnn = []
sq = None
loss = None
opimizer = None
training_op = None
init = None
saver = None


            
            
            
            
