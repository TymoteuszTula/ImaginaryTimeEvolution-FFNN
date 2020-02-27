# ImaginaryTimeEvolution.py

'''This code evolve the density matrix in the temperature space so that
   we can obtain the thermal density matrix
'''

import numpy as np
import tensorflow as tf
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class NeuralNetworkGenerator:
    
    def __init__(self, weights, bias, activation_vector, no_of_sites):
        self.weights = weights
        self.bias = bias
        self.no_weights = len(weights)
        self.activ = []
        self.no_of_sites = no_of_sites
        
        for i_weight in range(self.no_weights):
            if activation_vector[i_weight] == 'sig':
                self.activ.append(self.sigmoid)
            if activation_vector[i_weight] == 'elu':
                self.activ.append(self.elu)
            if activation_vector[i_weight] == 'rel':
                self.activ.append(self.relu)
            if activation_vector[i_weight] == 'exp':
                self.activ.append(self.exp)
        
    # Activation functions
    def elu(self, x):
        return (x < 0) * (np.exp(x) - 1) + (x>=0) * x
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return (x < 0) * 0 + (x >=0) * x
    
    def exp(self, x):
        return np.exp(x)
    
    # Neural Network output
    def nn_out(self, n_input):
        out = self.activ[0](np.dot(n_input, self.weights[0]) + self.bias[0])
        for i_weight in range(1, self.no_weights - 1):
            out = self.activ[i_weight](np.dot(out, self.weights[i_weight]) +
                                       self.bias[i_weight])
        return self.activ[-1](np.dot(out, self.weights[-1]) + self.bias[-1])
    
    # Calculating target output - identity matrix
    def target_output(self, n_input):
        if np.array_equal(n_input[:self.no_of_sites], n_input[self.no_of_sites:]):
            return 1
        else:
            return 0
        
    # Calculating target output - Hubbard model
    def target_outputHM(self, n_input):
        
        
    # Monte Carlo Evolution
    def mc_step(self, n_input):
        r1 = np.random.randint(2 * self.no_of_sites)
        r2 = np.random.randint(self.no_of_sites) + self.no_of_sites * (
               r1 // self.no_of_sites)
        n_output = np.copy(n_input)
        if n_output[r1] != 0:
            n_output[r1] -= 1
            n_output[r2] += 1

        target_output1 = self.target_output(n_input)
        target_output2 = self.target_output(n_output)
        prob1 = (target_output1 - self.nn_out(n_input[np.newaxis]))**2
        prob2 = (target_output2 - self.nn_out(n_output[np.newaxis]))**2
        if np.random.rand() < (prob2 / prob1):
            return n_output, target_output2
        else:
            return n_input, target_output1
        

def graph_initialization(neuron_vector, activation_vector, 
                         learning_rate = 0.005):
    ''' Initialises tensorflow graph to represent a fully connected neural 
        network specified by neuron_vector variable:
        
        neuron_vector - 1d list of number of neurons at each layer. First
                        entry corresponds to input layer and last entry to
                        the output layer-1. Size of the list corresponds to
                        total number of layer-1.
                        
        activation_vector - 1d list of possible activation functions for 
                            each layer. The length of this list is the lenght
                            of neuron_vector.
                            
        learning_rate - Learning rate of the network.
    '''
    
    no_layers = len(neuron_vector) + 1
    global X, y, sq, loss, optimizer, training_op, init, saver
    
    # Placeholders for input X and target output y
    X = tf.placeholder(tf.float32, shape=(None, neuron_vector[0]), name="X")
    y = tf.placeholder(tf.float32, shape=(None), name="y")
    
    
    # Neural Network architecture
    with tf.name_scope("DNN"):
        layers_dnn.append(tf.layers.dense(X, neuron_vector[1], name="hidden1",
                                          activation=activation_vector[0]))
        with tf.variable_scope("hidden1", reuse=True):
            weights_dnn.append(tf.get_variable("kernel"))
            bias_dnn.append(tf.get_variable("bias"))
            
        for layer in range(2, no_layers-1):
            layers_dnn.append(tf.layers.dense(layers_dnn[-1], 
                              neuron_vector[layer], name="hidden" + str(layer),
                              activation=activation_vector[layer-1]))
            with tf.variable_scope("hidden" + str(layer), reuse=True):
                weights_dnn.append(tf.get_variable("kernel"))
                bias_dnn.append(tf.get_variable("bias"))
                
        layers_dnn.append(tf.layers.dense(layers_dnn[-1], 1,
                          name="output", activation=activation_vector[-1]))
        with tf.variable_scope("output", reuse=True):
            weights_dnn.append(tf.get_variable("kernel"))
            bias_dnn.append(tf.get_variable("bias"))
            
    
    with tf.name_scope("LOSS"):
        sq = tf.square(y - layers_dnn[-1][:,0])
        loss = tf.reduce_mean(sq, name="loss")
        
    with tf.name_scope("TRAIN"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def tensor_eval(tensor):
    return tensor.eval()

    
def learning_procedure(mc_batchsize, n_epochs, no_of_sites, activation_vector,
                       no_of_particles, save = False):
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            batch = np.zeros((mc_batchsize, 2*no_of_sites))
            target_batch = np.zeros(mc_batchsize)
            n_start1 = np.random.randint(no_of_sites, size=(no_of_particles))
            n_start2 = np.random.randint(no_of_sites, size=(no_of_particles))
            nn_instance = NeuralNetworkGenerator(list(map(tensor_eval, 
                                                          weights_dnn)),
                                                 list(map(tensor_eval,
                                                          bias_dnn)),
                                                 activation_vector, 
                                                 no_of_sites)
            for site in range(no_of_sites):
                batch[0,site] = np.sum(n_start1 == site)
                batch[0,site + no_of_sites] = np.sum(n_start2 == site)
            target_batch[0] = nn_instance.target_output(batch[0,:])
            
            for instance in range(mc_batchsize-1):
                (batch[instance+1,:],
                target_batch[instance+1]) = nn_instance.mc_step(
                    batch[instance,:])
            
            sess.run(training_op, feed_dict={X: batch, y: target_batch})
            acc_train = loss.eval(feed_dict={X: batch, y: target_batch})
            
            print(np.sum(target_batch))
            print(epoch, "Train accuracy:", acc_train)
            
            
            
            
