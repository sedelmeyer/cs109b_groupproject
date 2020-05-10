"""
This module contains functions for generating and analyzing trees and
tree ensemble models and visualizing the model results

PARAMETERS

    depths = list(range(1,21))
        sets default depths for comparison in cross validation

    cv = 5
        sets cross-validation kfold parameter

FUNCTIONS

    generate_adaboost_staged_scores()
        Generates adaboost staged scores in order to find ideal number of
        iterations

    plot_adaboost_staged_scores()
        Plots the adaboost staged scores for each y variable's predictions
        and iteration

    calc_meanstd_logistic()

    calc_meanstd_regression()

    define_train_and_test()
        Return x and y data for train and test sets

    expand_attributes()
        helper function to expand attributes when dummies or multiple
        columns are used

    plot_me()
        plot the best depth finder for decision tree model

    calculate()
        returns the results of using a set of attributes on the data

    calc_models()
        iterates over all combinations of attributes to return lists of
        resulting models    

"""


import itertools

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .model import generate_model_dict


# Calculate train and test scores for model inputs and outputs

depths = list(range(1, 21))
cv = 5


def generate_adaboost_staged_scores(model_dict, X_train, X_test, y_train, y_test):
    """Generates adaboost staged scores in order to find ideal number of iterations
    
    :return: tuple of 2D np.arrays for adaboost staged scores at each iteration and
             each response variable, one array for training scores and one for test 
    """
    staged_scores_train = np.hstack(
        [
            np.array(
                list(
                    model.staged_score(
                        X_train.reset_index(drop=True),
                        y_train.reset_index(drop=True).iloc[:, i]
                    )
                )
            ).reshape(-1,1) for i, model in enumerate(model_dict['model'])
        ]
    )
    
    staged_scores_test = np.hstack(
        [
            np.array(
                list(
                    model.staged_score(
                        X_test.reset_index(drop=True),
                        y_test.reset_index(drop=True).iloc[:, i]
                    )
                )
            ).reshape(-1,1) for i, model in enumerate(model_dict['model'])
        ]
    )
    
    return staged_scores_train, staged_scores_test


def plot_adaboost_staged_scores(model_dict, X_train, X_test, y_train, y_test):
    """Plots the adaboost staged scores for each y variable's predictions and iteration
    
    """
    # generate staged_scores
    training_scores, test_scores = generate_adaboost_staged_scores(
        model_dict, X_train, X_test, y_train, y_test
    )
    
    max_depth = model_dict['model'][0].base_estimator.max_depth
    learning_rate = model_dict['model'][0].learning_rate
    y_vars = [var.replace('_', ' ') for var in model_dict['y_variables']]

    # create list of iteration numbers for plotting
    iteration_numbers = np.arange(model_dict['model'][0].n_estimators) + 1

    # plot figure
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.title(
        "Number of iterations' effect on the AdaBoost Regessor's\nperformance "\
        "with max depth {} and learning rate {}".format(
            max_depth,
            learning_rate,
        ),
        fontsize=18,
    )

    ax.plot(
        iteration_numbers, training_scores[:, 0],
        color='k', linestyle='--', linewidth=2,
        label='{}, training'.format(y_vars[0]))

    ax.plot(
        iteration_numbers, test_scores[:, 0],
        color='k', linestyle='-', linewidth=2,
        label='{}, TEST'.format(y_vars[0]))

    ax.plot(
        iteration_numbers, training_scores[:, 1],
        color='silver', linestyle='--', linewidth=2,
        label='{}, training'.format(y_vars[1]))

    ax.plot(
        iteration_numbers, test_scores[:, 1],
        color='silver', linestyle='-', linewidth=2,
        label='{}, TEST'.format(y_vars[1]))

    ax.tick_params(labelsize=12)
    ax.set_ylabel("$R^2$ score", fontsize=16)
    ax.set_xlabel("number of adaboost iterations", fontsize=16)
    ax.set_xticks(iteration_numbers)
    ax.grid(':', alpha=0.4)
    ax.legend(fontsize=12, edgecolor='k')

    plt.tight_layout()
    plt.show()


def calc_meanstd_logistic(X_tr, y_tr, X_te, y_te, depths:list=depths, cv:int=cv):
    cvmeans = []
    cvstds = []
    train_scores = []
    test_scores = []
    models = []
    
    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=109)
        model.fit(X_tr, y_tr) # train model
        
        # cross validation
        cvmeans.append(np.mean(cross_val_score(model, X_tr, y_tr, cv=cv, scoring='accuracy' )))
        cvstds.append( np.std (cross_val_score(model, X_te, y_te, cv=cv, scoring='accuracy' )))

        models.append(model)
        
        # use AUC scoring
        train_scores.append( roc_auc_score(y_tr, model.predict(X_tr)))
        test_scores.append(  roc_auc_score(y_te, model.predict(X_te)))
 
    # make the lists np.arrays
    cvmeans = np.array(cvmeans)
    cvstds = np.array(cvstds)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    
    return cvmeans, cvstds, train_scores, test_scores, models


def calc_meanstd_regression(X_tr, y_tr, X_te, y_te, depths:list=depths, cv:int=cv):
    cvmeans = []
    cvstds = []
    train_scores = []
    test_scores = []
    models = []
    
    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=109)
        model.fit(X_tr, y_tr) # train model
        
        # cross validation
        cvmeans.append(np.mean(cross_val_score(model, X_tr, y_tr, cv=cv, scoring='r2')))
        cvstds.append( np.std (cross_val_score(model, X_te, y_te, cv=cv, scoring='r2')))

        models.append(model)
        
        # use R2 scoring
        train_scores.append( model.score(X_tr, y_tr) )  # append train score - picks accuracy or r2 automatically
        test_scores.append(  model.score(X_te, y_te) ) # append cv test score - picks accuracy or r2 automatically
       
    # make the lists np.arrays
    cvmeans = np.array(cvmeans)
    cvstds = np.array(cvstds)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    
    return cvmeans, cvstds, train_scores, test_scores, models


def define_train_and_test(data_train, data_test, attributes,
                          response, logistic) -> (pd.DataFrame, pd.DataFrame):
    """Return x and y data for train and test sets
    """
    X_tr = data_train[attributes]
    y_tr = data_train[response]

    X_te = data_test[attributes]
    y_te = data_test[response]
    
    if logistic:
        y_tr = (y_tr>0)*1
        y_te = (y_te>0)*1
    
    return X_tr, X_te, y_tr, y_te


def expand_attributes(attrs, categories):
    """helper function to expand attributes when dummies or multuple colmuns are used
    """
    # update the attributes to use dummies if 'category' is included
    if 'Category' in attrs:
        attrs.remove('Category')
        attrs += categories
       
    if 'umap_attributes_2D_embed' in attrs:
        attrs.remove('umap_attributes_2D_embed')
        attrs += ['umap_attributes_2D_embed_1', 'umap_attributes_2D_embed_2']
        
    # ensure that only 1 embedding is selected
    count = 0
    embedding = None
    for i in ['umap_descr_2D_embed', 'ae_descr_embed', 'pca_descr_embed']:
        if i in attrs:
            count += 1
            embedding = i
    if count > 1:
        print("ERROR")
        print("Only one of the three embeddings is allowed.")
        return
    
    if embedding in attrs:
        attrs.remove(embedding)
        attrs += [f'{embedding}_1', f'{embedding}_2']
        
    return attrs


def plot_me(result):
    """plot the best depth finder for decision tree model
    
    relies on 'result' dictionary from 'calculate' function
    """    
    depths = list(range(1, 21))
    cv = 5
    
    responses = result.get('responses')
    full_attributes = result.get('full_attributes')
    attributes = result.get('attributes')
    score_type = result.get('scoring')
    model_type = result.get('model_type')
    train_scores = result.get('train_scores')
    test_scores = result.get('test_scores')
    x = result.get('depths')
    
    print(f"Model Optmized for: {result.get('responses')}")
    
    fig, ax = plt.subplots(ncols = len(responses), figsize=(15,6))
    
    for i, (a, response) in enumerate(zip(np.ravel(ax), responses)):

        best_depth = result.get('best_depth')
        best_score = test_scores[best_depth-1]

        a.set_xlabel("Maximum Tree Depth")

        attrs_title = '\n'.join(attributes)
        title = f"Model: {model_type}\nResp: {response}\nAttrs: {attrs_title}"

        a.set_title(
            f"{title}\nBest test {score_type.capitalize()} score: {best_score} at depth {best_depth}",
            fontsize=10
        )
        a.set_ylabel(f"{score_type.capitalize()} Score")
        a.set_xticks(depths)

        # Plot model train scores
        a.plot(
            x, train_scores, 'b-', marker='o',
            label=f"Model Train {score_type.capitalize()} Score"
        )

        # Plot model test scores
        a.plot(
            x, test_scores, 'o-', marker='.',
            label=f"Model Test {score_type.capitalize()} Score"
        )

        if i == len(responses)-1:
            a.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
            

def calculate(data_train, data_test, categories, attributes:list, 
              responses_list:list, logistic=True):
    """returns the results of using a set of attributes on the data
    """
    if logistic:
        model_type = 'Logistic'
        score_type = 'auc'
        calc = calc_meanstd_logistic
    else:
        model_type = 'Regression'
        score_type = 'r2'
        calc = calc_meanstd_regression
        
    # remove multi-output responses, if not using logistic regression
    responses = [] 
    for r in responses_list:
        if type(r) == str:
            r = [r]
        if len(r) > 1 and not logistic:
            continue
        responses.append(r)
    
    results = []
    model_dict = []
    # update the attributes to use dummies if 'category' is included
    attrs = expand_attributes(attributes.copy(), categories)
    
    for i, response in enumerate(responses):

        X_tr, X_te, y_tr, y_te = define_train_and_test(
            data_train, data_test, attrs,
            ['Budget_Change_Ratio', 'Schedule_Change_Ratio'], logistic=logistic
        )
        
        cvmeans, cvstds, train_scores, test_scores, models = calc(
            X_tr, y_tr[response], X_te, y_te[response]
        )

        best_model = models[test_scores.argmax()]
        best_score = test_scores[test_scores.argmax()]
        best_depth = test_scores.argmax()+1
        
        desc = f"{model_type} Tree. Depth: {best_depth}"

        results.append(
            {
                'desc':desc,
                'model_type':model_type,
                'attributes':attributes,
                'full_attributes':attrs,
                'responses':response,
                'Budget_Change_Ratio': 1 if 'Budget_Change_Ratio' in response and len(response) == 1 else 0,
                'Schedule_Change_Ratio': 1 if 'Schedule_Change_Ratio' in response and len(response) == 1 else 0,
                'Budget_and_Schedule_Change': 1 if len(response) == 2 else 0,
                'scoring': score_type,
                'best_depth':best_depth,
                'train_score':train_scores[best_depth-1],
                'train_scores':train_scores,
                'test_score':test_scores[best_depth-1],
                'test_scores':test_scores,
                'best_model':best_model,
                'depths':depths
            }
        )
        
        model_dict.append(
            generate_model_dict(
                model=DecisionTreeClassifier if logistic else DecisionTreeRegressor, 
                model_descr=desc, 
                X_train=X_tr, 
                X_test=X_te, 
                y_train=y_tr, 
                y_test=y_te, 
                multioutput=logistic,
                verbose=False,
                predictions=True,
                scores=True,
                model_api='sklearn',
                sm_formulas=None,
                y_stored=True,
                max_depth=best_depth, 
                random_state=109))
    
    return results, model_dict


def calc_models(data_train, data_test, categories, 
                nondescr_attrbutes, descr_attributes,
                responses_list, logistic=True):
    """iterates over all combinations of attributes to return lists of resulting models
    """
    results_all = []
    model_dicts = []
    
    print(f"Using {'LOGISTIC' if logistic else 'REGRESSION'} models")
    for i in tqdm(range(1, len(nondescr_attrbutes))):
        alist = list(itertools.combinations(nondescr_attrbutes, i))
        for a in tqdm(alist, leave=False):
            a = list(a)
            results, model_dict = calculate(
                data_train, data_test, categories, attributes=a, 
                responses_list=responses_list, logistic=logistic
            )
            results_all += results
            model_dicts += model_dict
            for d_emb in tqdm(descr_attributes, leave=False):
                results, model_dict = calculate(
                    data_train, data_test, categories, attributes=a + [d_emb],
                    responses_list=responses_list, logistic=logistic
                )
                results_all += results
                model_dicts += model_dict
                
    return results_all, model_dicts