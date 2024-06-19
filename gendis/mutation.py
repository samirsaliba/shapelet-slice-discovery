import numpy as np
import random

##########################################################################
#                         Mutatation operators                           #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - shapelets (np.array)
# OUTPUT: 
#    - new_shapelets (np.array)

def smooth_shapelet(shapelets, toolbox):
    """Smooth a random shapelet"""
    rand_shapelet = np.random.randint(len(shapelets))
    shap = shapelets[rand_shapelet]

    window = 5
    shap = np.ravel(shap)
    shap_mva = np.convolve(shap, np.ones(window), 'valid') / window
    fill = np.full(shape=len(shap)-len(shap_mva), fill_value=shap_mva[-1])
    shap = np.concatenate([shap_mva, fill])
    shapelets[rand_shapelet] = shap
    return shapelets,

def add_shapelet(shapelets, toolbox):
    """Add a shapelet to the individual"""
    shapelets.append(toolbox.create(n_shapelets=1))
    return shapelets,

def remove_shapelet(shapelets, toolbox):
    """Remove a random shapelet from the individual"""
    if len(shapelets) > 1:
        rand_shapelet = np.random.randint(len(shapelets))
        shapelets.pop(rand_shapelet)

    return shapelets,

def replace_shapelet(shapelets, toolbox):
    """
    Replace a random shapelet in the individual with a newly created one.
    
    Parameters:
    shapelets (list of lists): The individual (list of shapelets)
    toolbox: The toolbox with the function to create a new shapelet
    
    Returns:
    shapelets (list of lists): The modified individual with one shapelet replaced
    """
    if shapelets:
        # Randomly select an index to remove
        remove_index = random.randint(0, len(shapelets) - 1)
        # Remove the selected shapelet
        del shapelets[remove_index]
    
    # Add a new shapelet
    shapelets.append(toolbox.create(n_shapelets=1))
    
    return shapelets,