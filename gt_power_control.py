import numpy as np
from sim_functions import *

def payoff_function(alpha, power_vec, channel):

    num_ue = power_vec.shape[0]

    mu_vec = np.zeros(num_ue)
    
    # Gets the better channel coefficient of each interest UE
    better_ch = np.argmax(channel[0], axis=1)
    
    for ue in range(num_ue):
        
        # Interest signal received power
        interest = dbm2lin(power_vec[ue]) * channel[:, ue, better_ch[ue]]**alpha

        # Interference and inverse interference signal received power sums
        interference = 0
        inv_interference = 0
        
        print('\n ue', ue)
        
        for i_ue in range(num_ue):

            interference += dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha
            inv_interference += 1/(dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha)

        interference -= interest
        inv_interference -= 1/interest
        
        # Calculate mu value of each UE
        gamma = interference / interest
        lamb = interest * inv_interference
        
        print('gamma', gamma)
        print('lambda', lamb)
        
        mu_vec[ue] = gamma + lamb
        
    return mu_vec

def minimizer_power(alpha, prev_power, channel):

    num_ue = prev_power.shape[0]
    power_vec = np.zeros(num_ue)
    
    better_ch = np.argmax(channel[0], axis=1)
    
    for ue in range(num_ue):
        
        # Interest signal received power
        interest = dbm2lin(prev_power[ue]) * channel[:, ue, better_ch[ue]]**alpha
        
        # Interference and inverse interference signal received power sums
        interference = 0
        inv_interference = 0
        
        for i_ue in range(num_ue):

            interference += dbm2lin(prev_power[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha
            inv_interference += (channel[:, ue, better_ch[ue]]**(2*alpha)) / (dbm2lin(prev_power[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha)
            
        interference -= interest
        inv_interference -= (channel[:, ue, better_ch[ue]]**(2*alpha)) / interest

        power_vec[ue] = np.sqrt(interference*(inv_interference)**(-1))

    return lin2dbm(power_vec)

def game_pas(pmax, alpha, num_ue, epsilon, channel):
    
    power_vec = np.ones(num_ue) * pmax

    power_evolution = []

    iter = 1
    
    while True:

        #print('\n', iter)

        min_power = minimizer_power(alpha, power_vec, channel)
        #min_power = np.clip(min_power, lin2dbm(1e-5), pmax)
        
        #print(min_power, '\n')

        # p(l)        
        prev_power_vec = power_vec.copy()
        power_evolution.append(prev_power_vec)
        
        for ue in range(num_ue):

            # p* for each ue
            aux_power_vec = prev_power_vec.copy()
            aux_power_vec[ue] = min_power[ue]
            
            #print(prev_power_vec)
            #print(aux_power_vec, '\n')

            #print(payoff_function(alpha, prev_power_vec, channel)[ue] - payoff_function(alpha, aux_power_vec, channel)[ue])

            if payoff_function(alpha, prev_power_vec, channel)[ue] - payoff_function(alpha, aux_power_vec, channel)[ue] > epsilon:  
                # p(l) = p* if payoff function is minimized, doesnt change if it isnt
                power_vec[ue] = aux_power_vec[ue]

        iter += 1

        #print(power_vec)

        if (power_vec == prev_power_vec).all():

            break

    return power_vec, power_evolution