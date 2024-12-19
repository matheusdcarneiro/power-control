from system_parameters import *
from sim_functions import *
import numpy as np

def esnp_maxmin(channel, pmax, pmin, step, delta, epsilon, time, n_power):
    ''' 
    Exhaustive Search with Neighborhood Preference: Returns a SINR vector after performing a power
    control algorithm that alternate between doing global and local search in order to maximize 
    the objective function, maxmin SINR in this case.

    Parameters
    ----------
    channel : numpy_ndarray
        The channel matrix between each UE and AP.
    pmax : int, float
        The maximum power of system.
    pmin : int, float
        The minimum power of system.
    step : int, float
        The step to define the available values of power.
    delta : int, float
        The step that defines the range of values in local search.
    epsilon : int, float
        The probability of doing a global or a local search in each iteration.
    time : int
        The total number of iterations.
    n_power : numpy_ndarray
        The noise power of system.
    '''      

    # Available power values
    t_powers = np.linspace(pmin, pmax, int(((pmax - pmin)/step)+1))

    num_ue = channel.shape[1]

    # SINR vector after convergence and for each step t
    conv_sinr = np.zeros(num_ue)
    step_sinr = np.zeros(num_ue)

    # power vector after convergence and for each step t
    power_vec = pmax * np.ones(num_ue)
    conv_power = np.zeros(num_ue)

    # objective function: min SINR
    obj_func = 0

    for t in range(time):

        # Gets the objective function given the power vector in step t 
        step_sinr = get_sinr(power_vec, channel, n_power)
        min_sinr = np.min(step_sinr)

        # Checks if objective function is being maximized in step t
        if min_sinr > obj_func:

            # Update objective function, power vector and SINR vector
            obj_func = min_sinr.copy()
            conv_sinr = step_sinr.copy()
            conv_power = power_vec.copy()

        # Generate random value to sort if power vector will be a random value within the available
        # power values or will be a value in the neighborhood of the current one
        rand_value = np.random.rand()

        if rand_value < epsilon:

            for power in range(num_ue):
                # Get a random vector within the available power values
                power_vec[power] = t_powers[np.random.randint(t_powers.shape[0])]

        else: 

            # Get a value in the neighboorhood of current vector based on the delta
            power_vec = conv_power + delta * (np.random.rand(num_ue) - 0.5)

        for power in range(num_ue):
            # Keep power values between minimum and maximum power
            power_vec = np.clip(power_vec, pmin, pmax)

    post_conv_power = np.zeros(num_ue)

    for p in range(num_ue):

        post_conv_power[p] = t_powers[np.argmin(np.absolute(conv_power[p] - t_powers))]

    post_conv_sinr = get_sinr(post_conv_power, channel, n_power)

    return post_conv_sinr, post_conv_power

def esnp_maxsum(channel, pmax, pmin, step, delta, epsilon, time, n_power):
    ''' 
    Exhaustive Search with Neighborhood Preference: Returns a SINR vector after performing a power
    control algorithm that alternate between doing global and local search in order to maximize 
    the objective function, maxsum SINR in this case.

    Parameters
    ----------
    channel : numpy_ndarray
        The channel matrix between each UE and AP.
    pmax : int, float
        The maximum power of system.
    pmin : int, float
        The minimum power of system.
    step : int, float
        The step to define the available values of power.
    delta : int, float
        The step that defines the range of values in local search.
    epsilon : int, float
        The probability of doing a global or a local search in each iteration.
    time : int
        The total number of iterations.
    n_power : numpy_ndarray
        The noise power of system.
    '''     

    # Available power values
    t_powers = np.linspace(pmin, pmax, int(((pmax - pmin)/step)+1))

    num_ue = channel.shape[1]

    # SINR vector after convergence and for each step t
    conv_sinr = np.zeros(num_ue)
    step_sinr = np.zeros(num_ue)

    # power vector after convergence and for each step t
    power_vec = pmax * np.ones(num_ue)
    conv_power = np.zeros(num_ue)

    # objective function: SINR sum
    obj_func = 0

    for t in range(time):

        # Gets the objective function given the power vector in step t 
        step_sinr = get_sinr(power_vec, channel, n_power)
        sum_sinr = np.sum(step_sinr)

        # Checks if objective function is being maximized in step t
        if sum_sinr > obj_func:

            # Update objective function, power vector and SINR vector
            obj_func = sum_sinr.copy()
            conv_sinr = step_sinr.copy()
            conv_power = power_vec.copy()

        # Generate random value to sort if power vector will be a random value within the available
        # power values or will be a value in the neighborhood of the current one
        rand_value = np.random.rand()

        if rand_value < epsilon:

            for power in range(num_ue):
                # Get a random vector within the available power values
                power_vec[power] = t_powers[np.random.randint(t_powers.shape[0])]

        else: 

            # Get a value in the neighboorhood of current vector based on the delta
            power_vec = conv_power + delta * (np.random.rand(num_ue) - 0.5)

        for power in range(num_ue):
            # Keep power values between minimum and maximum power
            power_vec = np.clip(power_vec, pmin, pmax)

    post_conv_power = np.zeros(num_ue)

    for p in range(num_ue):

        post_conv_power[p] = t_powers[np.argmin(np.absolute(conv_power[p] - t_powers))]

    post_conv_sinr = get_sinr(post_conv_power, channel, n_power)

    return post_conv_sinr, post_conv_power

def esnp_maxproduct(channel, pmax, pmin, step, delta, epsilon, time, n_power):
    ''' 
    Exhaustive Search with Neighborhood Preference: Returns a SINR vector after performing a power
    control algorithm that alternate between doing global and local search in order to maximize 
    the objective function, max product between min SINR and SINR sum in this case.

    Parameters
    ----------
    channel : numpy_ndarray
        The channel matrix between each UE and AP.
    pmax : int, float
        The maximum power of system.
    pmin : int, float
        The minimum power of system.
    step : int, float
        The step to define the available values of power.
    delta : int, float
        The step that defines the range of values in local search.
    epsilon : int, float
        The probability of doing a global or a local search in each iteration.
    time : int
        The total number of iterations.
    n_power : numpy_ndarray
        The noise power of system.
    '''     


    # Available power values
    t_powers = np.linspace(pmin, pmax, int(((pmax - pmin)/step)+1))

    num_ue = channel.shape[1]

    # SINR vector after convergence and for each step t
    conv_sinr = np.zeros(num_ue)
    step_sinr = np.zeros(num_ue)

    # power vector after convergence and for each step t
    power_vec = pmax * np.ones(num_ue)
    conv_power = np.zeros(num_ue)

    # objective function: min SINR * SINR sum
    obj_func = 0

    for t in range(time):

        # Gets the objective function given the power vector in step t 
        step_sinr = get_sinr(power_vec, channel, n_power)
        prod_sinr = np.sum(step_sinr) * np.min(step_sinr)

        # Checks if objective function is being maximized in step t
        if prod_sinr > obj_func:

            # Update objective function, power vector and SINR vector
            obj_func = prod_sinr.copy()
            conv_sinr = step_sinr.copy()
            conv_power = power_vec.copy()

        # Generate random value to sort if power vector will be a random value within the available
        # power values or will be a value in the neighborhood of the current one
        rand_value = np.random.rand()

        if rand_value < epsilon:

            for power in range(num_ue):
                # Get a random vector within the available power values
                power_vec[power] = t_powers[np.random.randint(t_powers.shape[0])]

        else: 

            # Get a value in the neighboorhood of current vector based on the delta
            power_vec = conv_power + delta * (np.random.rand(num_ue) - 0.5)

        for power in range(num_ue):
            # Keep power values between minimum and maximum power
            power_vec = np.clip(power_vec, pmin, pmax)

    post_conv_power = np.zeros(num_ue)

    for p in range(num_ue):

        post_conv_power[p] = t_powers[np.argmin(np.absolute(conv_power[p] - t_powers))]

    post_conv_sinr = get_sinr(post_conv_power, channel, n_power)

    return post_conv_sinr, post_conv_power