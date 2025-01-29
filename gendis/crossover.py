from .individual import Shapelet, ShapeletIndividual


def point_crossover(parent1, parent2, X, np_random):
    """
    Point crossover between two ShapeletIndividuals.

    Args:
        parent1, parent2: ShapeletIndividual objects, each containing a list of shapelets.
        X: Time-series matrix where shapelets are extracted from.
        np_random: Instance of a random numpy generator.

    Returns:
        Tuple containing two new ShapeletIndividuals (child1, child2).
    """
    # Select a random shapelet from each individual for crossover
    idx1 = np_random.integers(len(parent1))
    idx2 = np_random.integers(len(parent2))

    shap_a = parent1[idx1]
    shap_b = parent2[idx2]

    timeseries_c = X[shap_a.index][shap_b.start : shap_b.start + len(shap_b)]
    shap_c = Shapelet(timeseries_c, index=shap_a.index, start=shap_b.start)

    timeseries_d = X[shap_b.index][shap_a.start : shap_a.start + len(shap_a)]
    shap_d = Shapelet(timeseries_d, index=shap_b.index, start=shap_a.start)

    # Create new individuals
    child1_shapelets = parent1[:]
    child2_shapelets = parent2[:]

    child1_shapelets[idx1] = shap_c
    child2_shapelets[idx2] = shap_d

    parent1.clear()
    parent1.extend(child1_shapelets)
    parent2.clear()
    parent2.extend(child2_shapelets)

    return parent1, parent2


def crossover_AND(ind1, ind2, **kwargs):
    """
    Perform crossover by creating a new individual from the union of two individuals' shapelets,
    ensuring that both parents' shapelets are included in the child.

    Parameters:
    ind1 (ShapeletIndividual): First parent individual
    ind2 (ShapeletIndividual): Second parent individual

    Returns:
    child1 (ShapeletIndividual): New individual created from the union of ind1 and ind2 shapelets
    child2 (ShapeletIndividual): New individual (clone of ind1) to preserve DEAP operator interface
    """
    # Create a new individual containing shapelets from both parents
    new_shapelets = list(ind1) + list(ind2)
    child = ShapeletIndividual(list(new_shapelets))

    return (
        child,
        ind1,
    )  # Returning child and one parent clone for compatibility with DEAP


def crossover_uniform(ind1, ind2, np_random, **kwargs):
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

        if np_random.rand() < 0.5:
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
