import numpy as np
from sim_functions import *

def payoff_function(alpha, power_vec, channel):

    num_ue = power_vec.shape[0]

    mu_vec = np.zeros(num_ue)
    
    better_ch = np.argmax(channel[0], axis=1)
    
    for ue in range(num_ue):
        
        # Interest signal received power
        interest = dbm2lin(power_vec[ue]) * channel[:, ue, better_ch[ue]]**alpha

        # Interference and inverse interference signal received power sums
        interference = 0
        inv_interference = 0
        
        for i_ue in range(num_ue):

            interference += dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha
            inv_interference += 1/(dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha)

        interference -= interest
        inv_interference -= 1/interest

        # Calculate mu value of each UE
        gamma = interference / interest
        lamb = interest * inv_interference
        
        mu_vec[ue] = gamma + lamb
        
    return mu_vec

def minimizer_power(alpha, prev_power, channel):

    num_ue = prev_power.shape[0]
    power_vec = np.zeros(num_ue)
    
    better_ch = np.argmax(channel[0], axis=1)
    
    for ue in range(num_ue):
        
        # Interest signal received power
        interest = dbm2lin(power_vec[ue]) * channel[:, ue, better_ch[ue]]**alpha
        
        # Interference and inverse interference signal received power sums
        interference = 0
        inv_interference = 0
        
        for i_ue in range(num_ue):

            interference += dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha
            inv_interference += (channel[:, ue, better_ch[ue]]**(2*alpha)) / (dbm2lin(power_vec[i_ue]) * channel[:, i_ue, better_ch[ue]]**alpha)
            
        interference -= interest
        inv_interference -= (channel[:, ue, better_ch[ue]]**(2*alpha)) / interest

        power_vec[]


def game_pas(p_max, alpha, num_ue, epsilon):
    
    power_vec = np.ones(num_ue)
    
    iter = 0
    
    while True:
        
            
    
    