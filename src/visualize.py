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

    plot_bdgt_sched_scaled()
        Plots 2 subplots for comparison, (1) the original budget vs schedule data
        plotted on the x and y axes, (2) the scaled versions of the budget vs
        schedule data plotted on the x and y values

    plot_change_trend()
        Plots 4 subplots showing project budget and duration forecast change trend
        over a specified time interval, or optionally for all available change
        records in cleansed master dataset

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


def plot_bdgt_sched_scaled(X, X_scaled, scale_descr, X_test=None, X_test_scaled=None):
    """Plots original vs scaled versions of budget and schedule input data
    
    :param X: Dataframe or 2D array with original budget and schedule train data
    :param X_scaled: Dataframe or 2D array with scaled budget and schedule train data
    :param scale_descr: Short string description of scaling transformation used
                        to title scaled data plot (e.g. 'Sigmoid Standardized')
    :param X_test: Optional, Dataframe or 2D array with original test data, which
                   will plot test data as overlay with training data (default is
                   X_test=None, which does not plot any overlay)
    :param X_test_scaled: Optional, Dataframe or 2D array with original test data,
                          which plots overlay similar to X_test (default is
                          X_test_scaled=None)
                          
    :return: Generates 1x2 subplotted scatterplots, no objects returned
    """
    corr = np.corrcoef(X.T)[0, 1]
    corr_scaled = np.corrcoef(X_scaled.T)[0, 1]

    fig, ax = plt.subplots(1,2, figsize=(12,6))
    
    plt.suptitle(
        'Original budget and duration values vs. {} scaled values'.format(
            scale_descr
            ),
        y=1.05,
        fontsize=18
    )
    
    for i, (data, data_test) in enumerate(
        zip([X, X_scaled], [X_test, X_test_scaled])
        ):
        ax[i].scatter(
            *data.values.T,
            color='k',
            alpha=0.5,
            edgecolor='w',
            s=80,
            label='training obs'
        )
        ax[i].set_title(
            'Original data\n({:.2f} pearson coefficient)'.format(corr) if i==0
            else '{} scaled\n({:.2f} pearson coefficient)'.format(
                scale_descr, corr_scaled
                ),
            fontsize=14
        )
        ax[i].set_xlabel('Budget', fontsize=12)
        if i==0:
            ax[i].set_ylabel('Duration (days)', fontsize=12)

        if np.all(X_test)!=None:
            ax[i].scatter(
                *data_test.values.T,
                color='tab:orange',
                alpha=1,
                edgecolor='w',
                marker='s',
                s=80,
                label='test obs'
            )
        
        ax[i].grid(':', alpha=0.4)
        
    ax[0].legend(fontsize=12, edgecolor='k', loc=4)        
    plt.tight_layout()
    plt.show()


def plot_change_trend(trend_data, pid_data, pid, interval=None):
    """Plots 4 subplots showing project budget and duration forecast change trend
    
    :param trend_data: pd.DataFrame, the cleaned dataset of all project change
                       records (i.e. 'Capital_Projects_clean.csv' dataframe)
    :param pid_data: pd.DataFrame, the prediction_interval dataframe produced
                     using this project's data generator function
                     (i.e. 'NYC_Capital_Projects_3yr.csv' dataframe)
    :param pid: integer, the PID for the project you wish to plot
    :param interval: integer or None, indicating the max Change_Year you wish
                     to plot, if None all change records are plotted for the
                     specified pid (default, interval=None)
                     
    :return: generates image of 4 subplots, no objects are returned
    """
    # sets default for converting datetimes in matplotlib
    from pandas.plotting import register_matplotlib_converters
    from matplotlib.dates import YearLocator, DateFormatter
    register_matplotlib_converters()
    
    years = YearLocator()
    years_fmt = DateFormatter('%Y')
     
    def set_date_axis(ax, years, years_fmt):
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
    
    # set sup_title reference fontsize
    fs=16

    # subset project record data (results from data generator)
    pid_record = pid_data.copy().loc[pid_data['PID']==pid]
    
    # subset project changes data (clean original dataset)
    changes_loc = (trend_data['PID']==pid) & (trend_data['Change_Year']<=interval) \
                  if interval else trend_data['PID']==pid
    pid_changes = trend_data.copy().loc[changes_loc]
    
    # convert datetime field to correct data type
    pid_changes['Date_Reported_As_Of'] = pd.to_datetime(pid_changes['Date_Reported_As_Of'])

    # calculate project duration array
    project_duration = pid_record['Duration_Start'].values[0] + \
                       np.cumsum(pid_changes['Latest_Schedule_Changes'].values)

    perc_bdgt_change = (
        pid_changes['Budget_Forecast'].values[-1] - pid_record['Budget_Start'].values[0]
    ) / pid_record['Budget_Start'].values[0]
    
    perc_sched_change = (
        project_duration[-1] - pid_record['Duration_Start'].values[0]
    ) / pid_record['Duration_Start'].values[0]
    
        # generate subplots
    fig, ax = plt.subplots(2,2, sharex=True, figsize=(12,6))
    
    plt.suptitle(
        'PID {}\n{}\nCategory: {}\nBorough: {}\nOriginal Budget: \${:.2f} million\n'\
        'Original Duration: {:,.0f} days'.format(
            pid,
            pid_record['Project_Name'].values[0][:88],
            pid_record['Category'].values[0],
            pid_record['Borough'].values[0],
            pid_record['Budget_Start'].values[0]/1e7,
            pid_record['Duration_Start'].values[0]
        ), fontsize=fs, y=1.25
    )

    # plot budget forecast
    ax[0,0].plot(
        pid_changes['Date_Reported_As_Of'], pid_changes['Budget_Forecast']/1e7, 'ko-'
    )
    ax[0,0].axhline(
        pid_record['Budget_Start'].values[0]/1e7, color='k', linestyle=':',
        label='Original forecasted'
    )
    ax[0,0].set_title(
        'Total budget forecast\n({:.2%} total change)'.format(perc_bdgt_change),
        fontsize=fs-2
    )
    ax[0,0].set_ylabel('USD (millions)', fontsize=fs-4)
    ax[0,0].legend(edgecolor='k', fontsize=fs-6)

    # plot budget forecast percent change
    ax[1,0].plot(
        pid_changes['Date_Reported_As_Of'],
        ((pid_changes['Latest_Budget_Changes'])/
        (
            pid_changes['Budget_Forecast']-pid_changes['Latest_Budget_Changes']
        ).replace(0,1))*100,
        'ko-'
    )
    ax[1,0].axhline(0, color='gray', linestyle='-', alpha=0.4)
    ax[1,0].set_title('Percentage budget change', fontsize=fs-2)
    ax[1,0].set_ylabel('percent change', fontsize=fs-4)

    ax[1,0].set_xlabel('project change date', fontsize=fs-4)

    # plot duration forecast
    ax[0,1].plot(
        pid_changes['Date_Reported_As_Of'], project_duration/1e3, 'ko-'
    )
    ax[0,1].axhline(
        pid_record['Duration_Start'].values[0]/1e3, color='k', linestyle=':',
    )
    ax[0,1].set_title(
        'Total forecasted project duration\n({:.2%} total change)'.format(perc_sched_change),
        fontsize=fs-2
    )
    ax[0,1].set_ylabel('days (thousands)', fontsize=fs-4)

    # plot duration change
    ax[1,1].plot(
        pid_changes['Date_Reported_As_Of'],
        (pid_changes['Latest_Schedule_Changes'] /
        (
            project_duration - pid_changes['Latest_Schedule_Changes']
        ).replace(0,1))*100,
        'ko-'
    )
    ax[1,1].axhline(0, color='gray', linestyle='-', alpha=0.4)
    ax[1,1].set_title('Percentage duration change', fontsize=fs-2)
    ax[1,1].set_ylabel('percent change', fontsize=fs-4)
    
    ax[1,1].set_xlabel('project change date', fontsize=fs-4)
    
    for a in ax.flat:
        a.grid(':', alpha=0.4)
        set_date_axis(a, years, years_fmt)
    
    plt.tight_layout()
    plt.show()
