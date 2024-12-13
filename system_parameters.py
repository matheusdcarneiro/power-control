import numpy as np
from sim_functions import *

def get_channel(dis_matrix, shadowing_matrix, rayleigh_matrix):
    ''' 
    Returns the channel  matrix [in linear] between an UE and each AP. 

    Parameters
    ----------
    dis_matrix : numpy_ndarray
        The distance matrix between each UE and AP.
    shadowing_matrix : numpy_ndarray
        The shadowing matrix between each UE and AP.
    rayleigh_matrix : numpy_ndarray
        The rayleigh matrix between each UE and AP through each channel, representing fast fading effect.
    '''      

    ch = shadowing_matrix * (rayleigh_matrix**2) * (1e-4 / (dis_matrix**4))

    return ch

def get_noise(total_bandwidth, num_ch):
    ''' 
    Returns the noise power [in W] of system.

    Parameters
    ----------
    total_bandwidth : int, float
        Total bandwidth of system.
    num_ch : int
        Number of channels in the system.
    '''   

    return (total_bandwidth/num_ch) * 1e-20

def get_sinr(power_vec, channel, n_power):
    ''' 
    Returns the SINR vector [in linear] of each UE after doing the association UE-AP based on the better
    channel coefficient.

    [!] None channel allocation is being made considering scenario is NOMA. Will be done if necessary.

    Parameters
    ----------
    power_vec : numpy_ndarray
        The (1, num_ue) vector in which each entry is the power of each UE.
    channel : numpy_ndarray
        The channel matrix between each UE and AP.
    n_power : numpy_ndarray
        The noise power of system.
    '''      

    num_ch, num_ue, num_ap = channel.shape

    # Association based on better channel coefficient, entry position represents
    # UE and entry value represents AP associated
    better_ch = np.argmax(channel[0], axis=1)

    # Creates the SINR vector
    sinr_vec = np.zeros(num_ue)

    # Stores the SINR fo each UE
    for ue in range(num_ue):

        # Interest signal received power
        interest = power_vec[ue] * channel[:, ue, better_ch[ue]]

        # Interference signal received power sum
        interference = 0

        for i_ue in range(num_ue):

            interference += power_vec[i_ue] * channel[:, i_ue, better_ch[ue]]

        interference -= interest

        # Calculate SINR of each UE
        sinr_vec[ue] = interest / (interference + n_power)

    return sinr_vec

