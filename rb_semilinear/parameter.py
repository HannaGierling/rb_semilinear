from fenics import *
from typing import Literal
import numpy as np

def get_P(P_range: list[float], 
          P_strategy: Literal["decade", "decade_log", "log", "lin", "thesis",
                              "lin_n_log"], 
          P_ns: int = 50, 
          random: bool = False, endpoint: bool = True) -> np.ndarray:
    """
    Generate a sorted array of parameter values based on 
    distribution strategy.

    Parameters:
    -----------
    P_range : list
        A list containing lower and upper bounds [P_min, P_max] for the parameters.
    P_opt : str
        Strategy for sampling
        - "decade"     : Linear within each decade.
        - "decade_log" : Logarithmic within each decade.
        - "log"        : Logarithmically 
        - "lin"        : Linearly 
        - "lin_n_log"  : Union of 'log' and 'lin' sampling 
        - "thesis"     : selection of "decade" sampling as used in Thesis, Section 5 

    P_ns : int, optional
        Number of samples (used for "log" and "lin")
    random : bool, optional
        If True, generate samples randomly 
    endpoint : bool, optional
        Whether to include the endpoint in linear/log spacing.

    Returns:
    --------
    mus : np.ndarray
        Sorted array of parameter values 
    """
    log_start, log_end = np.log10(P_range[0]), np.log10(P_range[1])
    mus = []

    if P_strategy == "decade":
        points_per_decade = 9
        n_decades = int(log_end - log_start)
        for k in np.linspace(log_start, log_end, n_decades, endpoint=False):
            decade_values = np.linspace(10**k, 10**(k + 1), points_per_decade, endpoint=False)
            mus.extend(decade_values)
        mus.append(10**log_end)

    elif P_strategy == "decade_log":
        points_per_decade = 9
        n_decades = int(log_end - log_start)
        for k in np.linspace(log_start, log_end, n_decades + 1, endpoint=False):
            decade_values = np.logspace(k, k + 1, points_per_decade, endpoint=False)
            mus.extend(decade_values)
        mus.append(10**log_end)

    elif P_strategy == "log":
        if random:
            rng = np.random.default_rng()
            mean = np.log(np.sqrt(P_range[0] * P_range[1]))
            mus = rng.lognormal(mean, 1, P_ns)
        else:
            mus = np.logspace(log_start, log_end, num=P_ns, endpoint=endpoint)

    elif P_strategy == "lin":
        if random:
            rng = np.random.default_rng()
            mus = rng.uniform(P_range[0], P_range[1], P_ns)
        else:
            mus = np.linspace(P_range[0], P_range[1], num=P_ns, endpoint=endpoint)

    elif P_strategy == "thesis":
        mus = get_P(P_range, "decade", 9)
        mus = np.append( mus[4:-1:9],mus[0:-1:9])
        mus = np.append(mus, 1)
    
    elif P_strategy == "lin_n_log":
        if P_ns % 2 != 0:
            print("WARNING:  If P_ns is not divisible by 2, the resulting"+\
                        " number of samples may slightly differ from P_ns")
        ns = int(P_ns/2)
        mus = get_P(P_range, "lin", ns)
        mus = np.append(mus, get_P(P_range, "log", ns))
        
    else:
        raise Exception(f"Sampling strategy '{P_strategy}' not implemented.")

    return np.sort(np.array(mus))


