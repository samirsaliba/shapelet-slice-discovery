import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

HISTPLOT_ARGS = {
    "stat": 'count', "bins": 20, "edgecolor": None, "multiple": 'dodge', #"alpha": 0.3, 
}

def test_subgroup_separation(df, X, y, subgroups, shap):
    _X = X.copy(deep=True)

    # Get subgroups for the shapelet
    # sg_indexes, not_sg_indexes = gendis.get_subgroups(X, y, shapelets=[shap])
    sg_indexes, not_sg_indexes = subgroups
    _X['in_sg'] = False
    _X.iloc[sg_indexes, _X.columns.get_loc('in_sg')] = True
    _X['error'] = y
    _X['label'] = df['label']
    
    # Plot setup
    axs_multiplier = 0.75
    f, axs = plt.subplots(1, 3, figsize=(3*axs_multiplier*6.4, axs_multiplier*4.8))
   
    # Shapelet plot
    axs[0].plot(shap)
    
    # Plot error distributions (in subgroup, off subgroup)
    sns.histplot(data=_X, x='error', hue='in_sg', ax=axs[1], **HISTPLOT_ARGS)
    
    # Plot distances
    sns.histplot(data=_X, x='distances', hue='label', ax=axs[2], **HISTPLOT_ARGS)
    
    plt.show()
    
    print(f"Subgroup 'accuracy': {accuracy_score(_X['label'], _X['in_sg'])}")
    print(f'confusion matrix, label x subgroup label')
    print(confusion_matrix(_X['label'], _X['in_sg']))
    
    # Print error stats
    print("Error stats")
    print("Samples in subgroup:")
    print(_X.loc[_X['in_sg']==1, 'error'].describe())
    print("\nSamples not in subgroup:")
    print(_X.loc[_X['in_sg']==0, 'error'].describe())

    return _X

def plot_k_series_with_shaps(X, k, shap, ascending=True, X_ordered=False):
    _X = X.copy(deep=True)
    
    # Plot setup
    axs_multiplier = 1
    f, axs = plt.subplots(
        k, 1, 
        sharex=True, 
        figsize=(2*axs_multiplier*6.4, axs_multiplier*4.8),
        gridspec_kw={'hspace': 0.1,}
    )
    
    if not X_ordered:
        _X.sort_values(by='distances', inplace=True, ascending=ascending)
    
    for i in range(k):
        data = _X.iloc[i]
        ts = data.iloc[0:150]
        
        # Plot timeseries
        axs[i].plot(ts)
        
        # Plot shapelet
        shap_x = np.arange(data.positions, data.positions+len(shap))
        axs[i].plot(shap_x, shap)
        
    plt.xticks(np.arange(0, len(ts)+1, 30.0))
    plt.tight_layout()
    plt.show()
        
def test_classification_separation(D, y, d_train, d_test, shap, shap_id, plot_sin_y=True):
    plots_n = 2 + int(plot_sin_y)
    axs_multiplier = 0.75
    f, axs = plt.subplots(1, plots_n, figsize=(plots_n*axs_multiplier*6.4, axs_multiplier*4.8))
    print(f"Testing separation for shapelet {shap_id}")
    
    # Shapelet plot
    axs[0].plot(shap)

    # Distances histogram
    sns.histplot(data=D, x=shap_id, hue='label', ax=axs[1],  **HISTPLOT_ARGS)
    
    # Distance versus sin_y
    if plot_sin_y:
        D.plot.scatter(x='sin_y', y=shap_id, c='DarkBlue', ax=axs[2])
    
    # Accuracy using the shapelet as feature
    single_shapelet_train = d_train.loc[:, shap_id].values.reshape(-1, 1)
    single_shapelet_test = d_test.loc[:, shap_id].values.reshape(-1, 1)

    y_train = y[d_train.index]
    y_test = y[d_test.index]
    
    lr = LogisticRegression()
    lr.fit(single_shapelet_train, y_train)
    y_pred = lr.predict(single_shapelet_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy = {acc}')
    print(confusion_matrix(y_test, y_pred))
    
    plt.show()