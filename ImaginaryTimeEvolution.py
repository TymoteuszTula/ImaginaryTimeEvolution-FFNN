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
    mc_batchsize = 2000
    n_epochs = 2000
    no_of_sites = 5
    no_of_particles = 5
    U = 1
    delta_beta = 0.001
    beta_it = 1000
    
    
    start = time.time()
    ITEC.graph_initialization(neuron_vector, activation_vector1,
                         learning_rate = 0.001)
    ITEC.learning_procedure_evol(mc_batchsize, n_epochs, no_of_sites,
                       activation_vector2, no_of_particles,
                       save = True, save_name=f"{delta_beta:.3f}",
                       is_first=True, U=U, 
                       delta_beta=delta_beta)
    
    for iterations in range(beta_it):
        ITEC.learning_procedure_evol(mc_batchsize, n_epochs, no_of_sites,
                       activation_vector2, no_of_particles,
                       save = True, save_name=f"{((iterations + 2) * delta_beta):.3f}",
                       is_first=False, U=U, 
                       delta_beta=delta_beta,
                       previous_point = f"{((iterations + 1) * delta_beta):.3f}")
    
    end = time.time()
    
    print(end-start)

main()


            
            
            
            
