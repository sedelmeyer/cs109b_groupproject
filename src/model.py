"""
This module contains functions for generating fitted models and summarizing the results

FUNCTIONS

    generate_model_dict()
        Fits the specified model type and generates a dictionary of the results.
        This function is compatible with sklearn, keras, pygam, and statsmodels models.
        Statsmodels models used with this function must be called using the
        statsmodels.formulas.api interface.

    print_model_results()
        Summarizes model results that are stored in a generate_model_dict() output
        model dictionary

"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def generate_model_dict(model, model_descr, X_train, X_test, y_train, y_test,
                        multioutput=True, verbose=False, predictions=True,
                        scores=True, model_api='sklearn', sm_formula=None,
                        y_stored=True, **kwargs):
    """Fits the specified model type and generates a dictionary of results
    
    This function works for fitting and generating predictions for 
    sklearn, keras, and statsmodels models. PyGam models typically also
    work by specifying the 'sklearn' model_api. For statsmodels models, only those
    that depend on the statsmodels.formula.api work.
    
    :param model: the uninitialized sklearn, pygam, or statsmodels regression
                  model object, or a previously compiled keras model
    :param model_descr: a brief string describing the model (cannot exceed 80
                        characters)  
    :param X_train, X_test, y_train, y_test: the datasets on which to fit and
                                             evaluate the model
    :param multioutput: Boolean, if True and sklearn model_api, will attempt
                        fitting a single multioutput model, if False or 'statsmodel'
                        model_api fits separate models for each output
    :param verbose: if True, prints resulting fitted model object (default=False)
    :param predictions: if True the dict stores model.predict() predictions for
                        both the X_train and X_test input dataframes
    :param scores: if True, metrics scores are calculated and stored in the
                   resulting dict for both the train and test predictions
    :param model_api: specifies the api-type required for the input model, options
                      include 'sklearn', 'keras', or 'statsmodels' (default='sklearn')
    :param sm_formula: statsmodels formula defining model (include only endogenous
                       variables, such as 'x1 + x2 + x3' instead of 'y ~ x1 + x2 + 
                       x3'), default is None
    :param y_stored: boolean, determines whether the true y values are stored in the
                     resulting dictionary. It is convenient to keep these stored
                     alongside the predictions for easier evaluation later (default
                     is y_stored=True)
    :param **kwargs: are optional arguments that pass directly to the model object
                     at time of initialization, or in the case of the 'keras' model
                     api, they pass to the keras.mdoel.fit() method

    :return: returns a dictionary object containing the resulting fitted model
             object, resulting predictions, and train and test scores (if specified
             as True)
    """
    # check description len
    max_model_descr = 80
    len_model_descr = len(model_descr) 
    if len_model_descr > max_model_descr:
        raise ValueError(
            'Model description is currently {}, but cannot exceed {} characters'\
            ''.format(len_model_descr, max_model_descr)
        )
    # check that valid model_api was entered
    if model_api not in ['sklearn', 'keras', 'statsmodels']:
        raise ValueError(
            "model_api only accepts 'sklearn', 'keras', or 'statsmodels', "\
            "but you have entered: {}".format(model_api)
        )
    
    # initialize fit model list
    FitModel = []
    
    # initialize formula to store when statsmodels
    formulas = []
    
    # store exogen
    y_variables = list(y_train.columns)
    
    # Fit model with parameters specified by kwargs
    if model_api=='sklearn' and multioutput:
        FitModel.append(model(**kwargs).fit(X_train, y_train))
    
    if model_api=='sklearn' and not multioutput:
        for col in y_variables:
            FitModel.append(model(**kwargs).fit(X_train, y_train[col]))
            
    # Note that the **kwargs are passed to the .fit() method in the keras api
    # Keras models must be defined and compiled prior to passing to this function
    if model_api=='keras':
        FitModel.append(model.fit(X_train, y_train, **kwargs))

    # statsmodel fit using statsmodels.formula.api, so need to record
    # resulting formulas for use while fitting and in final dict
    if model_api=='statsmodels':
        for i, y in enumerate(y_variables):
            formulas.append(y + ' ~ {}'.format(sm_formula))
            FitModel.append(
                model(
                    formula=formulas[i],
                    data=X_train.join(y_train[y])
                ).fit()
            )

    # generate and save predictions on both train and test data
    if model_api=='statsmodels' or not multioutput:
        train_pred = np.hstack([
            np.array(model.predict(X_train)).reshape(-1,1)
            for model in FitModel
        ])
            
        test_pred = np.hstack([
            np.array(model.predict(X_test)).reshape(-1,1)
            for model in FitModel
        ])
        
    else:
        train_pred = FitModel[0].predict(X_train)
        test_pred = FitModel[0].predict(X_test)
    
    # store fitted model, predictions and scores to dict 
    model_dict = {
        'description': model_descr
    }
    
    model_dict['model'] = FitModel
    model_dict['y_variables'] = y_variables
    model_dict['formulas'] = formulas
    
    if y_stored:
        model_dict['y_values'] = {
            'train': y_train,
            'test': y_test,
        }
        
    if predictions:
        model_dict['predictions'] = {
            'train': train_pred,
            'test': test_pred,
        }
    
    if scores:
        model_dict['score'] = {
            'train': r2_score(y_train, train_pred, multioutput='raw_values'),
            'test': r2_score(y_test, test_pred, multioutput='raw_values'),
        }
    
    if verbose:
        print("\t{}".format(FitModel))

    return model_dict


def print_model_results(model_dict, score='both'):
    """
    Prints a model results summary from the model dictionary generated
    using the generate_model_dict() function
    
    :param model_dict: dict, output dictionary from the generate_model_dict()
                       function
    :param accuracy: None, 'both', 'test', or 'train' parameters accepted,
              identifies which results to print for this particular metric
              
    :return: nothing is returned, this function just prints summary output
    """
    train_opt = ['train', 'both']
    test_opt = ['test', 'both']
    
    print(
        '\nMODEL SUMMARY:\n{}\n\n\nThe fitted model object(s):\n'.format(
            model_dict['description']
        )
    )
    
    for model in model_dict['model']:
          print('\t{}\n'.format(model))
    
    if ('formulas' in model_dict.keys()) and len(model_dict['formulas']):
        print("\nThe formula for each fitted model object:\n")
        for formula in model_dict['formulas']:
            print('\t{}\n'.format(formula))
    
    if score:
        print('\nThis model resulted in the following R-squared scores:\n')
        for i, var in enumerate(model_dict['y_variables']):
            print('\t{}\n'.format(var))
            if score in train_opt:
                print('\t\tTraining\t{:.4f}'.format(model_dict['score']['train'][i]))
            if score in test_opt:
                print('\t\tTest\t\t{:.4f}'.format(model_dict['score']['test'][i]))
            print()
        print('\n')
