import numpy as np

def _appendDict(dict1:dict, dict2:dict):
    """
    Appends the values of 'dict2' to corresponding arrays in 'dict1'

    Parameters
    ----------
    dict1 : dict
        dictionary with numpy arrays as values (will be modified)
    dict2 : dict
        dictionary with values to append to dict1 arrays
    """
    if not dict1:
        for key, val in dict2.items():
            dict1[key] = np.append([], dict2[key])
    else:
        for key, val in dict1.items():
            dict1[key] = np.append(val, dict2[key])
