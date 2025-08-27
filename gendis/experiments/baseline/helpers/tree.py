import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .plot import auto_savefig


def get_shapelet_tree_clf(
    X_distances, subgroup_mask, shapelet_ids, tree_depth, random_state
):
    """
    Train a simple classifier to separate subgroup vs rest using selected shapelets.

    Parameters
    ----------
    X_distances : np.ndarray
        Shapelet distance matrix (n_samples, n_shapelets)

    subgroup_mask : np.ndarray of bool
        Boolean array indicating subgroup membership

    shapelet_ids : list or tuple of int
        Indices of shapelets to use as features

    Returns
    -------
    clf : trained DecisionTreeClassifier
    """
    X = X_distances[:, shapelet_ids]
    y = subgroup_mask.astype(int)

    clf = DecisionTreeClassifier(max_depth=tree_depth, random_state=random_state)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    print("Classification report:\n", classification_report(y, y_pred))
    return clf


@auto_savefig
def plot_decision_tree_model(clf, feature_names, img_path=None):
    """
    Plot a scikit-learn DecisionTreeClassifier using consistent style.

    Parameters
    ----------
    clf : DecisionTreeClassifier
    feature_names : list of str
    img_path : str or None
        If given, saves plot to this path; otherwise shows it.
    """
    plt.figure(figsize=(12, 6))
    plot_tree(clf, feature_names=feature_names, filled=True, rounded=True)
    plt.tight_layout()
