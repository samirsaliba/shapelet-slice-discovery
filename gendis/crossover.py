import numpy as np

##########################################################################
#                         Crossover operators                            #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - ind1 (np.array)
#    - ind2 (np.array)
# OUTPUT: 
#    - new_ind1 (np.array)
#    - new_ind2 (np.array)

def crossover_AND(ind1, ind2):
    """
    Perform crossover by creating a new individual from the union of two individuals' shapelets.
    
    Parameters:
    ind1 (list of lists): First parent individual
    ind2 (list of lists): Second parent individual
    
    Returns:
    new_ind1 (list of lists): New individual created from the union of ind1 and ind2
    new_ind2 (list of lists): Duplicate of new_ind1 (for consistency with interface)
    """
    return ind2.extend(ind1), ind1

def crossover_uniform(ind1, ind2):
    """
    Perform uniform crossover with a 50% mixing ratio on shapelets.
    
    Parameters:
    ind1 (list of lists): First parent individual
    ind2 (list of lists): Second parent individual
    
    Returns:
    new_ind1 (list of lists): First new individual created by uniform crossover
    new_ind2 (list of lists): Second new individual created by uniform crossover
    """
    max_length = max(len(ind1), len(ind2))
    
    new_ind1 = []
    new_ind2 = []
    
    for i in range(max_length):
        if i < len(ind1) and i < len(ind2):
            shapelet1 = ind1[i]
            shapelet2 = ind2[i]
        elif i < len(ind1):
            shapelet1 = ind1[i]
            shapelet2 = None
        else:
            shapelet1 = None
            shapelet2 = ind2[i]
        
        if np.random.rand() < 0.5:
            if shapelet1 is not None:
                new_ind1.append(shapelet1)
            if shapelet2 is not None:
                new_ind2.append(shapelet2)
        else:
            if shapelet2 is not None:
                new_ind1.append(shapelet2)
            if shapelet1 is not None:
                new_ind2.append(shapelet1)
    
    return new_ind1, new_ind2
