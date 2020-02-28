# ExactDMEvaluation.py

'''This code contains the exact evaluation of a given density matrix for
   a bosonic Hubbard model.
'''

from scipy.linalg import expm, sinm, cosm
import numpy as np

def main():
    # parameters
    no_of_sites = 5
    no_of_particles = 5
    beta = 0.001
    U = 20
    # creating the states
    column_vectors = []
    col_vect = np.arange(no_of_sites)[np.newaxis].T
    for part in range(no_of_particles):
        column_vectors.append(col_vect)
        col_vect = np.repeat(col_vect, no_of_sites, axis=0)
     
    v_par = column_vectors[-1]
    for part in range(no_of_particles-1):
        v_par = np.c_[np.tile(column_vectors[-part-2],(
            no_of_sites**(part+1),1)),v_par]
        
    v_sites_temp = np.sum(v_par == 0, axis=1)
    for site in range(no_of_sites-1):
        v_sites_temp = np.c_[v_sites_temp, np.sum(v_par==site+1, axis=1)]
        
    v_sites = np.unique(v_sites_temp, axis=0)
    no_of_states = v_sites.shape[0]
    
    # creating hamiltonian
    
    jump_matrix = np.zeros((2*no_of_sites, no_of_sites))
    
    jump_vector1 = np.r_[1, -1, np.zeros(no_of_sites - 2)]
    jump_vector2 = np.r_[-1, 1, np.zeros(no_of_sites - 2)]
    
    for site in range(no_of_sites):
        jump_matrix[2*site, :] = np.roll(jump_vector1, site)
        jump_matrix[2*site + 1,:] = np.roll(jump_vector2, site)
        
    H = np.zeros((no_of_states, no_of_states))
    
    for state1 in range(no_of_states):
        for state2 in range(no_of_states):
            if state1 == state2:
                H[state1,state2] = U / 2 * np.sum(
                    v_sites[state1,:] * (v_sites[state1,:] - 1))
            
            a1 = v_sites[state1,:] + jump_matrix
            a2 = v_sites[state2,:]
            
            if np.any(np.all(a1 == a2, axis=1)):
                H[state1,state2] = -1
            
    
    ro = expm(-beta * H)
    
    print(np.any(ro < 0))
    
    
        
    
    
main()
