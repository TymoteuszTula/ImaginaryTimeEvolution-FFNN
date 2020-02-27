# ImaginaryTimeEvolution.py

'''This code evolve the density matrix in the temperature space so that
   we can obtain the thermal density matrix
'''

import numpy as np
import tensorflow as tf
import time
import ITEClassesFunctions as ITEC
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():
    # Parameters
    neuron_vector = [10, 80, 80, 80, 80]
    activation_vector1 = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu,
                          tf.math.exp]
    activation_vector2 = ['rel', 'rel', 'rel', 'rel', 'exp']
    mc_batchsize = 1000
    n_epochs = 200
    no_of_sites = 5
    no_of_particles = 5
    U = 1
    delta_beta = 0.001
    
    
    start = time.time()
    ITEC.graph_initialization(neuron_vector, activation_vector1,
                         learning_rate = 0.002)
    ITEC.learning_procedure_evol(mc_batchsize, n_epochs, no_of_sites,
                       activation_vector2, no_of_particles,
                       save = False, is_first=True, U=U, 
                       delta_beta=delta_beta)
    end = time.time()
    
    print(end-start)

main()


            
            
            
            
