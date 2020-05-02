"""
This module contains functions for visualizing data and model results

FUNCTIONS

    plot_true_pred()
        Plots model prediction results directly from model_dict or input arrays.
        Generates 5 subplots, (1) true values with predicted values overlay, 
        each y variable on its own axis, (2) output variable 1 true vs. predicted
        on each axis,(3) output variable 2 true vs. predicted on each axis
        (4) output variable 1 true vs. residuals, (5) output variable 2 true
        vs. residuals

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



def plot_true_pred(model_dict=None, dataset='train', y_true=None, y_pred=None,
                   model_descr=None, y1_label=None, y2_label=None):
    """Plots model prediction results directly from model_dict or input arrays
    
    This plotting function only really requires that a model_dict from the
    generate_model_dict() function be used as input. However, through use of
    the y_true, y_pred, model_descr, and y1 and y2 label parameters, predictions
    stored in a shape (n,2) array can be plotted directly wihtout the use of
    a model_dict
    
    NOTE: This plotting function requires y to consist of 2 output variables.
          Therefore, it will not work with y data not of shape=(n, 2).
    
    :param model_dict: dictionary or None, if model results from the
                       generate_model_dict func is used, function defaults to
                       data from that dict for plot, if None plot expects y_true,
                       y_pred, model_descr, and y1/y2 label inputs for plotting
    :param dataset: string, 'train' or 'test', indicates whether to plot training or
                    test results if using model_dict as data source, and labels
                    plots accordingly if y_pred and y_true inputs are used (default
                    is 'train')
    :param y_true, y_pred: None or pd.DataFrame and np.array shape=(n,2) data sources
                           accepted and used for plotting if model_dict=None
                           (default for both is None)
    :param model_descr: None or string of max length 80 used to describe model in
                        title. If None, model_descr defaults to description in
                        model_dict, if string is entered, that string overrides the
                        description in model_dict, if using y_true/y_test as data
                        source model_descr must be specified as a string (default
                        is None)
    :param y1_label, y2_label: None or string of max length 40 used to describe
                               the 2 output y variables being plotted. These values
                               appear along the plot axes and in the titles of
                               subplots. If None, the y_variables names from the
                               model_dict are used. If strings are entered, those
                               strings are used to override the model_dict values.
                               If using y_true/y_test as data source, these values
                               must be specified (default is None for both label)
    :return: Generates 5 subplots, (1) true values with predicted values overlay, 
             each y variable on its own axis, (2) output variable 1 true vs. predicted
             on each axis,(3) output variable 2 true vs. predicted on each axis
             (4) output variable 1 true vs. residuals, (5) output variable 2 true
             vs. residuals (no objects are returned)
    """        
    # create placeholder var_labels list for easier handling of conditionals
    var_labels = [None, None]

    # extract required objects from model_dict if not None 
    if type(model_dict)==dict:
        y_true = model_dict['y_values'][dataset]
        y_pred = model_dict['predictions'][dataset]
        r2_scores = model_dict['score'][dataset]
        var_labels = [
            var.replace('_', ' ') for var in model_dict['y_variables']
        ]
        
        if model_descr==None:
            model_descr = model_dict['description']
    # calculate r2 scores if model_dict not provided
    else:
        r2_scores = r2_score(y_true, y_pred, multioutput='raw_values')
        
    # Set y labels or overwrite y labels if specified as not None
    if y1_label != None:
        var_labels[0] = y1_label
    if y2_label != None:
        var_labels[1] = y2_label

    # if y inputs are pandas dataframes, convert to numpy array
    if type(y_true)==pd.core.frame.DataFrame:
        y_true = y_true.copy().values        
    if type(y_pred)==pd.core.frame.DataFrame:
        y_pred = y_pred.copy().values
        
    # GENERATE PLOT 1
    fig, ax = plt.subplots(figsize=(12,6))
    
    plt.title(
        '{} predictions vs. true values for\n{}\n'.format(
            'TEST' if dataset.lower()=='test' else 'TRAINING', model_descr
        ),
        fontsize=18
    )
 
    plt.scatter(
        *y_true.T,
        color='silver',
        alpha=1,
        edgecolor='gray',
        marker='s',
        s=90,
        label='True values'
    )
    plt.scatter(
        *y_pred.T,
        color='c',
        alpha=1,
        edgecolor='k',
        marker='o',
        s=90,
        label='Predicted values'
    )
    
    ax.set_xlabel(var_labels[0], fontsize=12)
    ax.set_ylabel(var_labels[1], fontsize=12)

    ax.legend(fontsize=12, edgecolor='k')
            
    ax.grid(':', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # GENERATE SUBPLOTS 2 AND 3
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    
    plt.suptitle(
        'Predictions and residuals vs. true values by output variable',
        y=1.05,
        fontsize=16
    )
    
    for i, (ax, true, pred) in enumerate(zip(axes.flat, y_true.T, y_pred.T)):
        ax.scatter(
            true, pred,
            color='k',
            alpha=0.5,
            edgecolor='w',
            s=90
        )
        ax.set_title(
            '{}\n$R^2={:.3f}$'.format(var_labels[i], r2_scores[i]),
            fontsize=14
        )
        ax.set_xlabel('True value', fontsize=12)
        if i==0:
            ax.set_ylabel('Predicted value', fontsize=12)
        ax.axis('equal')
        ax.grid(':', alpha=0.4)
        
    plt.tight_layout()
    plt.show()

    # GENERATE SUBPLOTS 4 AND 5
    fig, axes = plt.subplots(1, 2, figsize=(12,3))
        
    for i, (ax, true, pred) in enumerate(zip(axes.flat, y_true.T, y_pred.T)):        
        ax.scatter(
            true, pred-true,
            color='k',
            alpha=0.5,
            edgecolor='w',
            s=90
        )
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title('Residuals', fontsize=14)        
        ax.set_xlabel('True value', fontsize=12)
        if i==0:
            ax.set_ylabel('Prediction error', fontsize=12)
        ax.grid(':', alpha=0.4)

    plt.tight_layout()
    plt.show()