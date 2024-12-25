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
    # Single-shapelet crossover case
    if len(parent1) == 1 and len(parent2) == 1:
        shap_a = parent1[0]
        shap_b = parent2[0]

        # For child1, use A's index with B's start
        timeseries_a = X[shap_a.index][shap_b.start : shap_b.start + len(shap_b)]
        shap_c = Shapelet(timeseries_a, index=shap_a.index, start=shap_b.start)

        # For child2, use B's index with A's start
        timeseries_b = X[shap_b.index][shap_a.start : shap_a.start + len(shap_a)]
        shap_d = Shapelet(timeseries_b, index=shap_b.index, start=shap_a.start)

        # TODO fix this. def not ideal, but creating a new individual here is troublesome
        parent1.clear()
        parent1.extend([shap_a, shap_c])
        parent2.clear()
        parent2.extend([shap_b, shap_d])

        return parent1, parent2

    # General case: crossover for individuals with multiple shapelets
    crossover_point = np_random.integers(1, min(len(parent1), len(parent2)))
    child1_shapelets = parent1[:crossover_point]
    child2_shapelets = parent2[:crossover_point]

    # Handle remaining shapelets from parent2 to child1
    for i, shap in enumerate(parent1[crossover_point:], start=crossover_point):
        if i < len(parent2):  # Still within the bounds of parent2
            shap_b = parent2[i]
            timeseries = X[shap.index][shap_b.start : shap_b.start + len(shap_b)]
            new_shap = Shapelet(timeseries, index=shap.index, start=shap_b.start)
            child1_shapelets.append(new_shap)
        else:
            # Extend child1 with remaining shapelets from parent1
            child1_shapelets.extend(parent1[i:])
            break

    # Handle remaining shapelets from parent1 to child2
    for i, shap in enumerate(parent2[crossover_point:], start=crossover_point):
        if i < len(parent1):  # Still within the bounds of parent1
            shap_b = parent1[i]
            timeseries = X[shap.index][shap_b.start : shap_b.start + len(shap_b)]
            new_shap = Shapelet(timeseries, index=shap.index, start=shap_b.start)
            child2_shapelets.append(new_shap)
        else:
            # Extend child2 with remaining shapelets from parent2
            child2_shapelets.extend(parent2[i:])
            break

    # Return the new children
    # TODO fix this. def not ideal, but creating a new individual here is troublesome
    parent1.clear()
    parent1.extend(child1_shapelets)
    parent2.clear()
    parent2.extend(child2_shapelets)

    return parent1, parent2

    # child1 = ShapeletIndividual(child1_shapelets)
    # child2 = ShapeletIndividual(child2_shapelets)
    # return child1, child2


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
