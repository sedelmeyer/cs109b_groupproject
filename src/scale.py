"""
This module contains functions for scaling features of an X features design matrix

FUNCTIONS

    scale_features()
        Scales a dataframe's features based on the values of a training dataframe
        and returns the resulting scaled dataframe. Accepts various sklearn scalers
        and allows you to specify features you do not want affected by scaling by
        using the exclude_scale_cols parameter.

    sigmoid()
        Efficient numpy-based sigmoid transformation of a dataframe, array, or matrix

    log_plus_one()
        Adds 1 to the input data and then applies Log transformation to those values

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def scale_features(train_df, val_df, exclude_scale_cols=[], scaler=RobustScaler,
                   scale_before_func=None, scale_after_func=None,
                   reapply_scaler=False, **kwargs):
    """Scales val_df features based on train_df and returns scaled dataframe
    
    Accepts various sklearn scalers and allows you to specify features you do not
    want affected by scaling by using the exclude_scale_cols parameter.
    
    :param train_df: The training data
    :param val_df: Your test/validation data
    :param exclude_scale_cols: Optional list containing names of columns we
                               do not wish to scale, default=[]
    :param scaler: The sklearn scaler method used to fit the data (i.e. StandardScaler,
                    MinMaxScaler, RobustScaler, etc.), default=RobustScaler
    :param scale_before_func: Optional function (i.e. np.log, np.sigmoid, or custom
                              function) to be applied to train and val dfs prior to the
                              scaler fitting and scaling val_df, default=None
    :param scale_after_func: Optional function (i.e. np.log, np.sigmoid, or custom
                             function) to be applied to val_df after the scaler has
                             scaled the datafrme
    :param reapply_scaler: Boolean, if set to True, the scaler is fitted a second time
                           after the scale_after_func is applied (useful if using
                           MinMaxScaler and you wish to maintain a 0 to 1 scale after
                           applying a secondary transformation to the data), default
                           is reapply_scaler=False
    :param kwargs: Any additional arguments are passed as parameters to the selected
                   scaler (for instance feature_range=(-1,1) would be an appropriate
                   argument if scaler is set to MinMaxScaler)
    :return: a feature-scaled version of the val_df dataframe, and a list of fitted
             sklearn scaler objects that were used to scale values (for later use in
             case original values need to be restored), list will either be of length
             1 or 2 depending on whether reapply_scaler was set to True
    """
    # create list of columns to ensure proper ordering of columns for output df
    col_list = list(train_df)
    
    # create list of non-binary column names for scaling
    scaled_columns = train_df.columns.difference(exclude_scale_cols)
    
    # apply initial scaling if specified
    if scale_before_func:
        train_df = scale_before_func(train_df.copy()[scaled_columns])
        val_df = scale_before_func(val_df.copy()[scaled_columns])
        
    # initialize list for storing fitted scaler objects
    Scaler = []
    
    # create Scaler instance fitted on non-binary train data
    Scaler.append(
        scaler(**kwargs).fit(train_df[scaled_columns])
    )
    
    # scale val_df and convert to dataframe with column names
    scaled_train_df = pd.DataFrame(
        Scaler[0].transform(train_df[scaled_columns]),
        columns=scaled_columns,
    )

    scaled_val_df = pd.DataFrame(
        Scaler[0].transform(val_df[scaled_columns]),
        columns=scaled_columns,
    )
    
    # apply initial scaling if specified
    if scale_after_func:
        scaled_train_df = scale_after_func(scaled_train_df.copy())
        scaled_val_df = scale_after_func(scaled_val_df.copy())

    # create StandardScaler instance fitted on non-binary train data
    if reapply_scaler:
        Scaler.append(
            scaler(**kwargs).fit(scaled_train_df[scaled_columns])
        )
        
        scaled_val_df = pd.DataFrame(
            Scaler[1].transform(scaled_val_df[scaled_columns]),
            columns=scaled_columns,
        )
    
    # merge scaled columns with unscaled columns
    scaled_df = pd.concat(
        [
            val_df.drop(scaled_columns, axis=1).reset_index(drop=True),
            scaled_val_df.copy()
        ],
        axis=1,
    )[col_list]
    
    # Return full scaled val dataframe and fitted Scaler object list
    return scaled_val_df, Scaler


def sigmoid(x):
    """Efficient numpy-based sigmoid transformation of a dataframe, array, or matrix
    
    :param x: data to undergo transformation (datatypes accepted include,
              pandas DataFrames and Series, numpy matrices and arrays, or single
              int or float values x)
    :return: The transformed dataframe, series, array, or value depending on
             the type of original input x object
    """
    return 1/(1 + np.exp(-x)) 


def log_plus_one(x):
    """Adds 1 to the input data and then applies Log transformation to those values
    
    :param x: data to undergo transformation (datatypes accepted include,
              pandas DataFrames and Series, numpy matrices and arrays, or single
              int or float values x)
    :return: The transformed dataframe, series, array, or value depending on
             the type of original input x object
    """
    return np.log(x + 1)


def encode_categories(data, colname, one_hot=True, drop_cat=None,
                      cat_list=None, drop_original_col=False):
    """Encodes categorical variable column and appends values to dataframe

    This function offers the option to either one-hot-encode or LabelEncode
    the values by setting one_hot to either True or False

    A comprehensive list of categories need to be specified for this function
    to work

    :param data: The pd.dataframe object containing the column you wish to
                 encode
    :param colname: string indicating name of column you wish to encode
    :param one_hot: boolean indicating whether you with to one-hot-encode
                    the categories. If False, the values are simply encoded to
                    a set of consecutive integers. (default)
    :param drop_cat: None or category value you wish to drop from your
                     one-hot-encoded variable columns. If None and
                     one_hot=True, no variable columns are dropped. If
                     one_hot=False, any category value passed drop_cat
                     will ensure that value is sorted to the last place
                     position in the resulting encoded integer values
                     (default drop_cat=None)
    :param cat_list: None or list specifying the full set of category values
                     contained in your target column. The benefit of
                     providing your own list is that it allows you to provide
                     a custom ordering of categories to the encoder. If None,
                     the categories will default to alphabetical order.
                     (default cat_list=None)
    :param drop_original_col: Boolean indicating whether the original
                              category column specified by colname will be
                              dropped from the resulting dataframe

    :return: pd.DataFrame of the original input dataframe with the additional
             encoded category column(s) appended to it.
    """
    # copy dataframe to prevent overwrite if not desired
    data_copy = data.copy()

    if not cat_list:
        cat_list = sorted(list(map(str, set(data_copy[colname]))))

    if drop_cat:
        # create ordered list with drop_cat at end of list
        cat_list_ordered = cat_list.copy()
        cat_list_ordered.remove(drop_cat)
        cat_list_ordered.append(drop_cat)

        # removed drop_cat from original cat_list
        cat_list.remove()

    if one_hot:
        # one-hot-encode categorical predictors and sort columns with cat_list
        cat_dummies_df = pd.get_dummies(data_copy['Category'])[cat_list]
        # append columns to original dataframe
        data_copy[cat_list] = cat_dummies_df

    else:
        # create dictionary for encoding categories to numerical values
        cat_map_dict = {
            cat: i for i, cat in enumerate(cat_list_ordered)
        }
        # generate encoded labels in one single 'coded' column
        data_copy['{}_Code'.format(colname)] = data_copy[colname].map(
            cat_map_dict
        ).fillna(data_copy[colname])

    if drop_original_col:
        # drop original category column if specified
        data_copy = data_copy.drop(columns=colname)

    return data_copy
