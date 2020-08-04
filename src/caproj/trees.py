"""
caproj.trees
~~~~~~~~~~~~

This module contains functions for generating and analyzing trees and
tree ensemble models and visualizing the model results

**Module variables:**

.. autosummary::

   depths
   cv

**Module functions:**

.. autosummary::

   generate_adaboost_staged_scores
   plot_adaboost_staged_scores
   calc_meanstd_classifier
   calc_meanstd_regression
   plot_tree_depth_finder
   calculate
   calc_models

"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm.notebook import tqdm

from .model import generate_model_dict
from .visualize import save_plot

depths = list(range(1, 21))
"""sets default depths for comparison in cross validation"""

cv = 5
"""sets cross-validation kfold parameter"""


def generate_adaboost_staged_scores(
    model_dict, X_train, X_test, y_train, y_test
):
    """Generates adaboost staged scores in order to find ideal number of iterations

    :param model_dict: Output fitted model dictionary generated using
            :func:`caproj.model.generate_model_dict`
    :type model_dict: dict
    :param X_train: Training data X values
    :type X_train: array-like
    :param X_test: Test data X values
    :type X_test: array-like
    :param y_train: Training data y values
    :type y_train: array-like
    :param y_test: Test data y values
    :type y_test: array-like
    :return: tuple of 2D numpy arrays for adaboost staged scores at each iteration and
             each response variable, one array for training scores and one for test
    :rtype: tuple
    """
    staged_scores_train = np.hstack(
        [
            np.array(
                list(
                    model.staged_score(
                        X_train.reset_index(drop=True),
                        y_train.reset_index(drop=True).iloc[:, i],
                    )
                )
            ).reshape(-1, 1)
            for i, model in enumerate(model_dict["model"])
        ]
    )

    staged_scores_test = np.hstack(
        [
            np.array(
                list(
                    model.staged_score(
                        X_test.reset_index(drop=True),
                        y_test.reset_index(drop=True).iloc[:, i],
                    )
                )
            ).reshape(-1, 1)
            for i, model in enumerate(model_dict["model"])
        ]
    )

    return staged_scores_train, staged_scores_test


def plot_adaboost_staged_scores(
    model_dict, X_train, X_test, y_train, y_test, height=4, savepath=None,
):
    """Plots the adaboost staged scores for each y variable's predictions and iteration

    :param model_dict: Output fitted model dictionary generated using
            :func:`caproj.model.generate_model_dict`
    :type model_dict: dict
    :param X_train: Training data X values
    :type X_train: array-like
    :param X_test: Test data X values
    :type X_test: array-like
    :param y_train: Training data y values
    :type y_train: array-like
    :param y_test: Test data y values
    :type y_test: array-like
    :param height: Height dimension of resulting plot, defaults to 4
    :type height: int, optional
    :param savepath: filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    :type savepath: str or None
    """
    # generate staged_scores
    training_scores, test_scores = generate_adaboost_staged_scores(
        model_dict, X_train, X_test, y_train, y_test
    )

    max_depth = model_dict["model"][0].base_estimator.max_depth
    learning_rate = model_dict["model"][0].learning_rate
    y_vars = [var.replace("_", " ") for var in model_dict["y_variables"]]

    # create list of iteration numbers for plotting
    iteration_numbers = np.arange(model_dict["model"][0].n_estimators) + 1

    # plot figure
    fig, ax = plt.subplots(figsize=(12, height))

    plt.title(
        "Number of iterations' effect on the AdaBoost Regessor's\nperformance "
        "with max depth {} and learning rate {}".format(
            max_depth, learning_rate,
        ),
        fontsize=18,
    )

    ax.plot(
        iteration_numbers,
        training_scores[:, 0],
        color="k",
        linestyle="--",
        linewidth=2,
        label="{}, training".format(y_vars[0]),
    )

    ax.plot(
        iteration_numbers,
        test_scores[:, 0],
        color="k",
        linestyle="-",
        linewidth=2,
        label="{}, TEST".format(y_vars[0]),
    )

    ax.plot(
        iteration_numbers,
        training_scores[:, 1],
        color="silver",
        linestyle="--",
        linewidth=2,
        label="{}, training".format(y_vars[1]),
    )

    ax.plot(
        iteration_numbers,
        test_scores[:, 1],
        color="silver",
        linestyle="-",
        linewidth=2,
        label="{}, TEST".format(y_vars[1]),
    )

    ax.tick_params(labelsize=12)
    ax.set_ylabel("$R^2$ score", fontsize=16)
    ax.set_xlabel("number of adaboost iterations", fontsize=16)
    ax.set_xticks(iteration_numbers)
    ax.grid(":", alpha=0.4)
    ax.legend(fontsize=12, edgecolor="k")

    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def calc_meanstd_classifier(
    X_tr, y_tr, X_te, y_te, depths: list = depths, cv: int = cv
):
    """Fits and generates tree classifier results, iterated for each input depth

    :param X_tr: Training data X values
    :type X_tr: array-like
    :param y_tr: Training data y values
    :type y_tr: array-like
    :param X_te: Test data X values
    :type X_te: array-like
    :param y_te: Test data y values
    :type y_te: array-like
    :param depths: List of depths for each iterated decision tree classifier,
            defaults to depths
    :type depths: list, optional
    :param cv: Number of k-folds used for cross-validation, defaults to cv
    :type cv: int, optional
    :return: Five arrays are returned (1) mean cross-validation scores for each
            iteration, (2) standard deviation of each cross-validation score, (3)
            each training observation's ROC AUC score, (4) each test observation's
            ROC AUC score, (5) each fitted classifier's model object
    :rtype: tuple
    """
    cvmeans = []
    cvstds = []
    train_scores = []
    test_scores = []
    models = []

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=109)
        model.fit(X_tr, y_tr)  # train model

        # cross validation
        cvmeans.append(
            np.mean(
                cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
            )
        )
        cvstds.append(
            np.std(
                cross_val_score(model, X_te, y_te, cv=cv, scoring="accuracy")
            )
        )

        models.append(model)

        # use AUC scoring
        train_scores.append(roc_auc_score(y_tr, model.predict(X_tr)))
        test_scores.append(roc_auc_score(y_te, model.predict(X_te)))

    # make the lists np.arrays
    cvmeans = np.array(cvmeans)
    cvstds = np.array(cvstds)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    return cvmeans, cvstds, train_scores, test_scores, models


def calc_meanstd_regression(
    X_tr, y_tr, X_te, y_te, depths: list = depths, cv: int = cv
):
    """Fits and generates tree regressor results, iterated for each input depth

    :param X_tr: Training data X values
    :type X_tr: array-like
    :param y_tr: Training data y values
    :type y_tr: array-like
    :param X_te: Test data X values
    :type X_te: array-like
    :param y_te: Test data y values
    :type y_te: array-like
    :param depths: List of depths for each iterated decision tree regressor,
            defaults to depths
    :type depths: list, optional
    :param cv: Number of k-folds used for cross-validation, defaults to cv
    :type cv: int, optional
    :return: Five arrays are returned (1) mean cross-validation scores for each
            iteration, (2) standard deviation of each cross-validation score, (3)
            each training observation's :math:`R^2` score, (4) each test observation's
            :math:`R^2` score, (5) each fitted regressors's model object
    :rtype: tuple
    """
    cvmeans = []
    cvstds = []
    train_scores = []
    test_scores = []
    models = []

    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=109)
        model.fit(X_tr, y_tr)  # train model

        # cross validation
        cvmeans.append(
            np.mean(cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2"))
        )
        cvstds.append(
            np.std(cross_val_score(model, X_te, y_te, cv=cv, scoring="r2"))
        )

        models.append(model)

        # use R2 scoring
        train_scores.append(
            model.score(X_tr, y_tr)
        )  # append train score - picks accuracy or r2 automatically
        test_scores.append(
            model.score(X_te, y_te)
        )  # append cv test score - picks accuracy or r2 automatically

    # make the lists np.arrays
    cvmeans = np.array(cvmeans)
    cvstds = np.array(cvstds)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    return cvmeans, cvstds, train_scores, test_scores, models


def plot_tree_depth_finder(result, height=5, savepath=None):
    """plot the best depth finder for decision tree model

    :param result: Dictionary returned from the :func:`calculate` function
    :type result: dict
    :param height: Height dimension of resulting plot, defaults to 4
    :type height: int, optional
    :param savepath: filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    :type savepath: str or None
    """
    depths = list(range(1, 21))

    responses = result.get("responses")
    attributes = result.get("attributes")
    score_type = result.get("scoring")
    model_type = result.get("model_type")
    train_scores = result.get("train_scores")
    test_scores = result.get("test_scores")
    x = result.get("depths")

    # print(f"Model Optmized for: {result.get('responses')}")

    fig, ax = plt.subplots(ncols=len(responses), figsize=(12, height))

    for i, (a, response) in enumerate(zip(np.ravel(ax), responses)):

        best_depth = result.get("best_depth")
        best_score = test_scores[best_depth - 1]

        a.set_xlabel("Maximum tree depth", fontsize=14)

        attrs_title = "\n".join(attributes)
        title = f"Model: {model_type}\nResp: {response}\nAttrs: {attrs_title}"

        a.set_title(
            f"{title}\nBest TEST {score_type.upper()} score: {best_score} "
            f"at depth {best_depth}",
            fontsize=14,
        )
        a.set_ylabel(f"{score_type.upper()} score", fontsize=14)
        a.set_xticks(depths)

        # Plot model train scores
        a.plot(
            x,
            train_scores,
            "k--",
            # marker="o",
            label=f"Training {score_type.upper()} score",
        )

        # Plot model test scores
        a.plot(
            x,
            test_scores,
            "k-",
            marker="o",
            label=f"TEST {score_type.upper()} score",
        )

        if i == len(responses) - 1:
            a.legend(
                fontsize=12,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )
            # a.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

    plt.grid(":", alpha=0.5)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def _define_train_and_test(
    data_train, data_test, attributes, response, classifier
) -> (pd.DataFrame, pd.DataFrame):
    """Return x and y data for train and test sets
    """
    X_tr = data_train[attributes]
    y_tr = data_train[response]

    X_te = data_test[attributes]
    y_te = data_test[response]

    if classifier:
        y_tr = (y_tr > 0) * 1
        y_te = (y_te > 0) * 1

    return X_tr, X_te, y_tr, y_te


def _expand_attributes(attrs, categories):
    """Helper function to expand attributes when dummies or multuple columns are used
    """
    # update the attributes to use dummies if 'category' is included
    if "Category" in attrs:
        attrs.remove("Category")
        attrs += categories

    if "umap_attributes_2D_embed" in attrs:
        attrs.remove("umap_attributes_2D_embed")
        attrs += ["umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2"]

    # ensure that only 1 embedding is selected
    count = 0
    embedding = None
    for i in ["umap_descr_2D_embed", "ae_descr_embed", "pca_descr_embed"]:
        if i in attrs:
            count += 1
            embedding = i
    if count > 1:
        print("ERROR")
        print("Only one of the three embeddings is allowed.")
        return

    if embedding in attrs:
        attrs.remove(embedding)
        attrs += [f"{embedding}_1", f"{embedding}_2"]

    return attrs


def calculate(
    data_train,
    data_test,
    categories,
    attributes: list,
    responses_list: list,
    classifier=True,
):
    """Calculate decision tree results using a particular set of X features

    :param data_train: Training dataset
    :type data_train: array-like
    :param data_test: Test dataset
    :type data_test: array-like
    :param categories: List of project categories as they appear in the data
    :type categories: list
    :param attributes: Column names of feature columns (i.e. each different
            X variable under consideration)
    :type attributes: list
    :param responses_list: Column names of model responses (i.e. each different
            y variable)
    :type responses_list: list
    :param classifier: Indicates whether to use decision tree classifier
            (i.e. ``classifier=True``) or regressor (i.e. ``classifier=False``),
            defaults to True
    :type classifier: bool, optional
    :return: Two lists containing (1) dictionaries of model results and
            (2) fitted model dictionaries, one dictionary for each response
            variable
    :rtype: tuple
    """
    if classifier:
        model_type = "Classifier"
        score_type = "auc"
        calc = calc_meanstd_classifier
    else:
        model_type = "Regression"
        score_type = "r2"
        calc = calc_meanstd_regression

    # remove multi-output responses, if not using classifier regression
    responses = []
    for r in responses_list:
        if type(r) == str:
            r = [r]
        if len(r) > 1 and not classifier:
            continue
        responses.append(r)

    results = []
    model_dict = []
    # update the attributes to use dummies if 'category' is included
    attrs = _expand_attributes(attributes.copy(), categories)

    for i, response in enumerate(responses):

        X_tr, X_te, y_tr, y_te = _define_train_and_test(
            data_train,
            data_test,
            attrs,
            ["Budget_Change_Ratio", "Schedule_Change_Ratio"],
            classifier=classifier,
        )

        cvmeans, cvstds, train_scores, test_scores, models = calc(
            X_tr, y_tr[response], X_te, y_te[response]
        )

        best_model = models[test_scores.argmax()]
        best_depth = test_scores.argmax() + 1

        desc = f"{model_type} Tree. Depth: {best_depth}"

        results.append(
            {
                "desc": desc,
                "model_type": model_type,
                "attributes": attributes,
                "full_attributes": attrs,
                "responses": response,
                "Budget_Change_Ratio": 1
                if "Budget_Change_Ratio" in response and len(response) == 1
                else 0,
                "Schedule_Change_Ratio": 1
                if "Schedule_Change_Ratio" in response and len(response) == 1
                else 0,
                "Budget_and_Schedule_Change": 1 if len(response) == 2 else 0,
                "scoring": score_type,
                "best_depth": best_depth,
                "train_score": train_scores[best_depth - 1],
                "train_scores": train_scores,
                "test_score": test_scores[best_depth - 1],
                "test_scores": test_scores,
                "best_model": best_model,
                "depths": depths,
            }
        )

        model_dict.append(
            generate_model_dict(
                model=DecisionTreeClassifier
                if classifier
                else DecisionTreeRegressor,
                model_descr=desc,
                X_train=X_tr,
                X_test=X_te,
                y_train=y_tr,
                y_test=y_te,
                multioutput=classifier,
                verbose=False,
                predictions=True,
                scores=True,
                model_api="sklearn",
                sm_formulas=None,
                y_stored=True,
                max_depth=best_depth,
                random_state=109,
            )
        )

    return results, model_dict


def calc_models(
    data_train,
    data_test,
    categories,
    nondescr_attrbutes,
    descr_attributes,
    responses_list,
    classifier=True,
):
    """Iterate over all combinations of attributes to return lists of resulting models

    :param data_train: Training dataset
    :type data_train: array-like
    :param data_test: Test dataset
    :type data_test: array-like
    :param categories: List of project categories as they appear in the data
    :type categories: list
    :param nondescr_attrbutes: Column names of all features not consisting of those
            engineered from project descriptions
    :type nondescr_attrbutes: list
    :param descr_attributes: Column names of features engineered using project
            descriptions
    :type descr_attributes: list
    :param responses_list: Column names of model responses (i.e. each different
            y variable)
    :type responses_list: list
    :param classifier: Indicates whether to use decision tree classifier
            (i.e. ``classifier=True``) or regressor (i.e. ``classifier=False``),
            defaults to True
    :type classifier: bool, optional
    :return: Two list objects containing (1) lists of dictionaries of model results and
            (2) lists of fitted model dictionaries for each iterated model
    :rtype: tuple
    """
    results_all = []
    model_dicts = []

    print(f"Using {'CLASSIFIER' if classifier else 'REGRESSION'} models")
    for i in tqdm(range(1, len(nondescr_attrbutes))):
        alist = list(itertools.combinations(nondescr_attrbutes, i))
        for a in tqdm(alist, leave=False):
            a = list(a)
            results, model_dict = calculate(
                data_train,
                data_test,
                categories,
                attributes=a,
                responses_list=responses_list,
                classifier=classifier,
            )
            results_all += results
            model_dicts += model_dict
            for d_emb in tqdm(descr_attributes, leave=False):
                results, model_dict = calculate(
                    data_train,
                    data_test,
                    categories,
                    attributes=a + [d_emb],
                    responses_list=responses_list,
                    classifier=classifier,
                )
                results_all += results
                model_dicts += model_dict

    return results_all, model_dicts


def _flatten(T):
    """Handles attributes list and flattens if required
    """
    if type(T) != tuple:
        return (T,)
    if len(T) == 0:
        return ()
    else:
        return _flatten(T[0]) + _flatten(T[1:])
