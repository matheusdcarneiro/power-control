def channel(dis_matrix, shadowing_matrix, rayleigh_matrix):
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

    ch = shadowing_matrix * (rayleigh_matrix**2) * (10e-4 / (dis_matrix**4))

    return ch