# ImaginaryTimeEvolution.py

'''This code contains definitions of classes and functions used
   by other scripts.
'''

import numpy as np
import tensorflow as tf
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


class NeuralNetworkGenerator:
    
    def __init__(self, weights, bias, activation_vector, no_of_sites,
                 delta_beta = 0, U = 0, evolution = False):
        self.weights = weights
        self.bias = bias
        self.no_weights = len(weights)
        self.activ = []
        self.delta_beta = delta_beta
        self.U = U
        self.no_of_sites = no_of_sites
        self.evolution = evolution
        
        for i_weight in range(self.no_weights):
            if activation_vector[i_weight] == 'sig':
                self.activ.append(self.sigmoid)
            if activation_vector[i_weight] == 'elu':
                self.activ.append(self.elu)
            if activation_vector[i_weight] == 'rel':
                self.activ.append(self.relu)
            if activation_vector[i_weight] == 'exp':
                self.activ.append(self.exp)
                
        self.jump_matrix = np.zeros((2*no_of_sites, no_of_sites))
        jump_vector1 = np.r_[1, -1, np.zeros(no_of_sites - 2)]
        jump_vector2 = np.r_[-1, 1, np.zeros(no_of_sites - 2)]
        
        for site in range(no_of_sites):
            self.jump_matrix[2*site, :] = np.roll(jump_vector1, site)
            self.jump_matrix[2*site + 1, :] = np.roll(jump_vector2, site)
        
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
        
    def target_output2(self, n_input):
        return np.prod(np.equal(n_input[:,:self.no_of_sites],
                                n_input[:,self.no_of_sites:]), axis=1)
        
    # Calculating target output - Hubbard model
    def target_outputHM(self, n_input):
        n_1 = n_input[:self.no_of_sites][np.newaxis]
        n_2 = n_input[self.no_of_sites:][np.newaxis]
        neigh_sites_1 = n_1 + self.jump_matrix
        neigh_sites_2 = n_2 + self.jump_matrix

        result = (self.nn_out(n_input[np.newaxis]) - self.delta_beta *
                   self.U / 4 * (self.nn_out(np.c_[n_1, n_2])
                            * np.sum(n_1 *(n_1 - 1)) +
                           self.nn_out(np.c_[n_1, n_2])
                           * np.sum(n_2 *(n_2 - 1))) +                    
                 self.delta_beta / 2 * np.sum(self.nn_out(np.c_[np.repeat(
                     n_1, 2*self.no_of_sites, axis=0), neigh_sites_2]) + 
                     self.nn_out(np.c_[neigh_sites_1, np.repeat(
                     n_2, 2*self.no_of_sites, axis=0)]), axis=0))
        
        #result = (self.target_output2(n_input[np.newaxis]) - self.delta_beta *
                    #self.U / 4 * (self.target_output2(np.c_[n_1, n_2])
                             #* np.sum(n_1 *(n_1 - 1)) +
                            #self.target_output2(np.c_[n_1, n_2])
                            #* np.sum(n_2 *(n_2 - 1))) +                    
                  #self.delta_beta / 2 * np.sum(self.target_output2(np.c_[np.repeat(
                      #n_1, 2*self.no_of_sites, axis=0), neigh_sites_2]) + 
                      #self.target_output2(np.c_[neigh_sites_1, np.repeat(
                      #n_2, 2*self.no_of_sites, axis=0)]), axis=0))
                      
        return result
        
    # Monte Carlo Evolution
    def mc_step(self, n_input, neuralNGenerator):
        r1 = np.random.randint(2 * self.no_of_sites)
        r2 = np.random.randint(self.no_of_sites) + self.no_of_sites * (
               r1 // self.no_of_sites)
        n_output = np.copy(n_input)
        if n_output[r1] != 0:
            n_output[r1] -= 1
            n_output[r2] += 1
        
        if self.evolution:
            target_output1 = neuralNGenerator.target_outputHM(n_input)
            target_output2 = neuralNGenerator.target_outputHM(n_output)
        else:
            target_output1 = neuralNGenerator.target_output(n_input)
            target_output2 = neuralNGenerator.target_output(n_output)
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
                    batch[instance,:], nn_instance)
            
            sess.run(training_op, feed_dict={X: batch, y: target_batch})
            acc_train = loss.eval(feed_dict={X: batch, y: target_batch})
            
            print(np.sum(target_batch))
            print(epoch, "Train accuracy:", acc_train)
            
        print(np.c_[target_batch, layers_dnn[-1].eval(feed_dict={X: batch})])
        if save:
            save_path = saver.save(sess, "./delta/delta_func_bosons_np=" +
                                str(no_of_particles) + "ns=" +
                                str(no_of_sites) + ".ckpt")
            
def learning_procedure_evol(mc_batchsize, n_epochs, no_of_sites, 
                            activation_vector, no_of_particles, U,
                            delta_beta, save=False, save_name='0',
                            is_first=False, previous_point='0'):
    with tf.Session() as sess:
        if is_first:
            saver.restore(sess, "./delta/delta_func_bosons_np=" +
                           str(no_of_particles) + "ns=" +
                           str(no_of_sites) + ".ckpt")
        else:
            saver.restore(sess, "./evolutionNN/evolution_bosons_np=" +
                                str(no_of_particles) + "ns=" +
                                str(no_of_sites) + "beta=" + 
                                previous_point + ".ckpt")
        
        old_nn = NeuralNetworkGenerator(list(map(tensor_eval, 
                                                weights_dnn)),
                                                 list(map(tensor_eval,
                                                 bias_dnn)),
                                                 activation_vector, 
                                                 no_of_sites,
                                                 delta_beta=delta_beta,
                                                 U=U, evolution=True)
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
                                                 no_of_sites,
                                                 evolution=True)
            for site in range(no_of_sites):
                batch[0,site] = np.sum(n_start1 == site)
                batch[0,site + no_of_sites] = np.sum(n_start2 == site)
            target_batch[0] = old_nn.target_output(batch[0,:])
            
            for instance in range(mc_batchsize-1):
                (batch[instance+1,:],
                target_batch[instance+1]) = nn_instance.mc_step(
                    batch[instance,:], old_nn)
                
            sess.run(training_op, feed_dict={X: batch, y: target_batch})
            acc_train = loss.eval(feed_dict={X: batch, y: target_batch})
            
            print(np.sum((target_batch > 0) * (target_batch < 0.5) ))
            print(epoch, "Train accuracy:", acc_train)
            
        print(np.c_[target_batch, layers_dnn[-1].eval(feed_dict={X: batch})])
        if save:
            save_path = saver.save(sess, "./evolutionNN/evolution_bosons_np=" +
                                str(no_of_particles) + "ns=" +
                                str(no_of_sites) + "beta=" + 
                                save_name + ".ckpt")                                         
    
    
    
    
    
    
    
    
    
    
            
            
