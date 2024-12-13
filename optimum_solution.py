import numpy as np
from itertools import product

def available_power_vec(pmin, pmax, step, num_ue):
    ''' 
    Returns the available transmit powers vectors of system.

    Parameters
    ----------
    pmin : int, float
        Minimum transmit power of each UE.
    pmax : int, float
        Maximum transmit power of each UE.
    step : int, float
        Discretization of set of powers available for each UE.
    num_ue : int
        Number of UEs in system.
    '''   

    # Available powers to be set
    t_powers = np.linspace(pmin, pmax, int(((pmax - pmin)/step)+1))

    # Set of possible power vecs of system
    power_vecs = list(product(t_powers, repeat=num_ue))

    return power_vecs