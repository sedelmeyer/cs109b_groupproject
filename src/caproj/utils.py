"""
caproj.utils
~~~~~~~~~~~~

This module contains utlility functions for performaing HDBSCAN and UMAP analyses

"""
import logging

import hdbscan
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
)
from umap import UMAP

# imports not installed in environment
# NOTE: code using these libraries is commented out below
# import xgboost as xgb
# import shap

# Unused imports required for commented-out functions
# import contextlib
# from IPython.utils.capture import capture_output
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm


def predict_ensemble(ensemble, X):
    """
    predict_ensemble runs the X data set through
    each classifier in the ensemble list to get predicted
    probabilities.

    Those are then averaged out across all classifiers.
    """
    probs = [r.predict_proba(X)[:, 1] for r in ensemble]
    return np.vstack(probs).mean(axis=0)


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def print_report(
    m, X_valid, y_valid, t=0.5, X_train=None, y_train=None, show_output=True
):
    """
    print_report prints a comprehensive classification report
    on both validation and training set (if provided).
    The metrics returned are AUC, F1, Precision, Recall and
    Confusion Matrix.

    It accepts both single classifiers and ensembles.

    Results are dependent on the probability threshold
    applied to individual predictions.
    """
    #     X_train = X_train.values
    #     X_valid = X_valid.values

    if isinstance(m, list):
        probs_valid = predict_ensemble(m, X_valid)
        y_val_pred = adjusted_classes(probs_valid, t)

        if X_train is not None:
            probs_train = predict_ensemble(m, X_train)
            y_train_pred = adjusted_classes(probs_train, t)
    else:
        probs_valid = m.predict_proba(X_valid)[:, 1]
        y_val_pred = adjusted_classes(probs_valid, t)

        if X_train is not None:
            probs_train = m.predict_proba(X_train)[:, 1]
            y_train_pred = adjusted_classes(probs_train, t)

    res = [
        roc_auc_score(y_valid, probs_valid),
        f1_score(y_valid, y_val_pred),
        confusion_matrix(y_valid, y_val_pred),
    ]
    result = f"AUC valid: {res[0]} \nF1 valid: {res[1]}"

    if X_train is not None:
        res += [
            roc_auc_score(y_train, probs_train),
            f1_score(y_train, y_train_pred),
        ]
        result += f"\nAUC train: {res[3]} \nF1 train: {res[4]}"

    acc_train = m.score(X_train, y_train)
    acc_valid = m.score(X_valid, y_valid)

    if show_output:
        logging.info(f"train acc: {acc_train}")
        logging.info(f"test acc: {acc_valid} ")

        logging.info(result)
        plot_confusion_matrix(
            m, X_valid, y_valid, display_labels=y_valid.unique()
        )
        logging.info(classification_report(y_valid, y_val_pred))
        plt.show()
    return {
        "train": {"AUC": res[3], "F1": res[4], "acc": acc_train},
        "test": {"AUC": res[0], "F1": res[1], "acc": acc_valid},
    }


# def train_output_metrics(
#     X,
#     y,
#     *,
#     test_size=0.3,
#     name,
#     path=None,
#     no_seed=False,
#     fitted=None,
#     show_output=True,
#     early_stopping,
#     model=None,
#     **kwargs,
# ):
#     if no_seed:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, stratify=y
#         )
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42, stratify=y
#         )

#     if not fitted:

#         if not model:
#             model = xgb.XGBClassifier(verbosity=0, **kwargs)
#            # xgb.XGBClassifier(
#            #     n_estimators=100,  subsample=0.8, colsample_bytree=0.5
#            # )
#         eval_set = [(X_train, y_train), (X_test, y_test)]
#         eval_metric = ["error", "logloss", "auc"]

#         # check if multi-label classification
#         if "num_class" in kwargs.keys():
#             eval_metric += ["merror", "mlogloss"]

#         with contextlib.nullcontext() if show_output else capture_output():
#             print(f"extra XGBoost kwargs: {kwargs}")
#             print(X_train.shape, y_train.shape)
#             print(eval_metric)

#             if early_stopping:
#                 fitted = model.fit(
#                     X_train,
#                     y_train,
#                     early_stopping_rounds=5,
#                     eval_metric=eval_metric,
#                     eval_set=eval_set,
#                 )

#             else:
#                 fitted = model.fit(
#                     X_train,
#                     y_train,
#                     eval_metric=["error", "logloss", "auc", "aucpr", "map"],
#                     eval_set=eval_set,
#                 )

#         results = model.evals_result()
#     if show_output:
#         logging.info(f"train size:{1- test_size}, test_size:{test_size}")
#         logging.info(
#             f"train case/control num "
#             f"{y_train.value_counts()[1]}/{y_train.value_counts()[0]}"
#         )
#         logging.info(
#             f"test case/control num  "
#             f"{y_test.value_counts()[1]}/{y_test.value_counts()[0]}"
#         )

#     metrics_dict = print_report(
#         model,
#         X_test,
#         y_test,
#         t=0.5,
#         X_train=X_train,
#         y_train=y_train,
#         show_output=show_output,
#     )

#     if path:
#         dump(fitted, open(path, "wb"))
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     shap_values_non_zero = pd.DataFrame(shap_values, columns=X.columns).loc[
#         :, shap_values.sum(axis=0) != 0
#     ]
#     important_feat = (
#         np.abs(shap_values_non_zero).mean().sort_values(ascending=False)
#     )

#     formatted_name = name.format()
#     important_feat.name = f"{name}_shap_vals"
#     important_feat = important_feat.to_frame()
#     return {
#         "shap_df": shap_values_non_zero,
#         "important_feat": important_feat,
#         "metrics": metrics_dict,
#         "model": model,
#     }


# def train_multi_params(*args, param_dict:dict=None, func=None, **kwargs):
#     if None in [changing_param, func]:
#         raise ValueError
#     else:
#         results = defaultdict(dict)
#         for key in tqdm(param_dict):
#             for val in tqdm(param_dict[key]):
#                 new_dict = kwargs.copy()
#                 new_dict[key] = val
#                 results[key][val] = func(*args,**new_dict)


# def repeat_train(
#     X, y, num_times, different_train_test_split, name, model=None, **kwargs
# ):
#     """[summary]

#     [extended_summary]

#     :param X: [description]
#     :type X: [type]
#     :param y: [description]
#     :type y: [type]
#     :param num_times: [description]
#     :type num_times: [type]
#     :param different_train_test_split: [description]
#     :type different_train_test_split: [type]
#     :param name: [description]
#     :type name: [type]
#     :param model: [description], defaults to None
#     :type model: [type], optional
#     :return: [description]
#     :rtype: [type]
#     """
#     result_list = []
#     print(f"Extra params: {kwargs}")
#     for i in tqdm(range(num_times), f"Training {num_times} models"):
#         result_dict = train_output_metrics(
#             X,
#             y,
#             name="base_model",
#             no_seed=different_train_test_split,
#             early_stopping=True,
#             model=model,
#             **kwargs,
#         )
#         result_list.append(result_dict)
#     auc_list = [
#         metric_dict["metrics"]["test"]["AUC"] for metric_dict in result_list
#     ]
#     auc_df = pd.DataFrame({"AUC": auc_list, "dataset": name})
#     fig = px.violin(auc_df, box=True, y="AUC", color="dataset")
#     return {"AUC": auc_df, "result_list": result_list, "fig": fig}


def draw_umap(
    data,
    n_neighbors=15,
    min_dist=0.1,
    c=None,
    n_components=2,
    metric="euclidean",
    title="",
    plot=True,
    cmap=None,
    use_plotly=False,
    **kwargs,
):
    """[summary]

    [extended_summary]

    :param data: [description]
    :type data: [type]
    :param n_neighbors: [description], defaults to 15
    :type n_neighbors: int, optional
    :param min_dist: [description], defaults to 0.1
    :type min_dist: float, optional
    :param c: [description], defaults to None
    :type c: [type], optional
    :param n_components: [description], defaults to 2
    :type n_components: int, optional
    :param metric: [description], defaults to "euclidean"
    :type metric: str, optional
    :param title: [description], defaults to ""
    :type title: str, optional
    :param plot: [description], defaults to True
    :type plot: bool, optional
    :param cmap: [description], defaults to None
    :type cmap: [type], optional
    :param use_plotly: [description], defaults to False
    :type use_plotly: bool, optional
    :return: [description]
    :rtype: [type]
    """
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
    )
    mapper = fit.fit(data)
    u = fit.transform(data)
    if plot:
        if use_plotly:
            fig = px.scatter(
                x=u[:, 0], y=u[:, 1], color=c, title=title, **kwargs
            )
            fig.update_layout(
                {
                    "plot_bgcolor": "rgba(0, 0, 0, 0)",
                    "paper_bgcolor": "rgba(0, 0, 0, 0)",
                }
            )
            fig.show()
        else:
            fig = plt.figure()
            if n_components == 1:
                ax = fig.add_subplot(111)
                ax.scatter(u[:, 0], range(len(u)), c=c)
            if n_components == 2:
                ax = fig.add_subplot(111)
                scatter = ax.scatter(u[:, 0], u[:, 1], c=c, label=c, cmap=cmap)
            if n_components == 3:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=c, s=100)
            plt.title(title, fontsize=18)
            legend = ax.legend(*scatter.legend_elements())
            ax.add_artist(legend)

    return u, mapper


def cluster_hdbscan(
    clusterable_embedding, min_cluster_size, viz_embedding_list
):
    print(f"min_cluster size: {min_cluster_size}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, prediction_data=True
    ).fit(clusterable_embedding)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,).fit_predict(
        clusterable_embedding
    )
    print(f"found {len(np.unique(labels))} clusters")
    clustered = labels >= 0
    print(f"fraction clustered: {np.sum(clustered)/labels.shape[0]}")
    for embedding in viz_embedding_list:
        plt.scatter(
            embedding[~clustered][:, 0],
            embedding[~clustered][:, 1],
            c=(0.5, 0.5, 0.5),
            s=10,
            alpha=0.5,
        )
        plt.scatter(
            embedding[clustered][:, 0],
            embedding[clustered][:, 1],
            c=labels[clustered],
            s=10,
            cmap="Spectral",
        )
        plt.legend(labels)
        plt.show()

    return labels, clusterer


# def get_cluster_defining_features(X, clustering_label, cluster_setting_name):
#     """Need to provide X as features only and NO clustering label """
#     cluster_names = np.unique(clustering_label)
#     final_dict = {}
#     all_cluster_info_mean_peaks_per_cluster = []

#     for cluster_name in tqdm(cluster_names):
#         final_dict[cluster_name] = {}
#         cluster_one_v_all_labels = (clustering_label == cluster_name).astype(
#             int
#         )
#         # use scale_pos_weight to address imbalance between clustering class vs rest

#         all_OnevsAll_models_result_dict = repeat_train(
#             X,
#             cluster_one_v_all_labels,
#             num_times=1000,
#             different_train_test_split=True,
#             name=cluster_setting_name,
#             scale_pos_weight=(cluster_one_v_all_labels == 0).sum()
#             / (cluster_one_v_all_labels == 1).sum(),
#             show_output=False,
#         )

#         final_dict[cluster_name]["AUC_fig"] = all_OnevsAll_models_result_dict[
#             "fig"
#         ]

#         shap_mean_peaks_per_cluster_all_models = pd.concat(
#             [
#                 result["shap_df"].assign(model_num=i)
#                 for i, result in enumerate(
#                     all_OnevsAll_models_result_dict["result_list"]
#                 )
#                 if result["metrics"]["test"]["AUC"] > 0.9
#             ]
#         ).reset_index()
#         # explainer = shap.TreeExplainer(shap_mean_peaks_per_cluster_list[0][2])

#         final_dict[cluster_name]["shap_summary_plot"] = shap.summary_plot(
#             shap_mean_peaks_per_cluster_all_models.drop(columns=["model_num"])
#             .groupby("index")
#             .mean()
#             .values,
#             X[
#                 shap_mean_peaks_per_cluster_all_models.drop(
#                     columns=["index", "model_num"]
#                 ).columns
#             ],
#         )

#         aggregate_peak_info_mean_peaks_per_cluster = (
#             X[
#                 shap_mean_peaks_per_cluster_all_models.drop(
#                     columns=["index", "model_num"]
#                 ).columns
#             ]
#             .assign(
#                 is_cluster=np.where(
#                     cluster_one_v_all_labels == 1,
#                     "cases_in_cluster",
#                     "cases_not_in_cluster",
#                 )
#             )
#             .groupby("is_cluster")
#             .agg(["mean", "median"])
#             .swaplevel(-2, -1, axis=1)
#         )
#         aggregate_peak_info_mean_peaks_per_cluster = pd.concat(
#             [
#                 aggregate_peak_info_mean_peaks_per_cluster["mean"].T.add_suffix(
#                     "_mean"
#                 ),
#                 aggregate_peak_info_mean_peaks_per_cluster[
#                     "median"
#                 ].T.add_suffix("_median"),
#             ],
#             axis=1,
#         )
#         aggregate_peak_info_mean_peaks_per_cluster

#         cluster_info_mean_peaks_per_cluster = (
#             shap_mean_peaks_per_cluster_all_models.drop_duplicates("model_num")
#             .drop(columns=["index", "model_num"])
#             .count()
#             .to_frame()
#         )
#         cluster_info_mean_peaks_per_cluster.columns = [
#             "num_model_used_this_peak"
#         ]
#        cluster_info_mean_peaks_per_cluster = cluster_info_mean_peaks_per_cluster.
#        merge(
#            aggregate_peak_info_mean_peaks_per_cluster,
#            how="outer",
#            left_index=True,
#            right_index=True,
#        )
#         cluster_info_mean_peaks_per_cluster["cluster"] = cluster_name

#         final_dict[cluster_name][
#             "cluster_info_mean_peaks_per_cluster"
#         ] = cluster_info_mean_peaks_per_cluster
#         all_cluster_info_mean_peaks_per_cluster.append(
#             cluster_info_mean_peaks_per_cluster
#         )

#     final_dict["combined_cluster_info_mean_peaks_per_cluster"] = pd.concat(
#         all_cluster_info_mean_peaks_per_cluster
#     )
#     final_dict["cluster_setting"] = cluster_setting_name
#     return final_dict


# def get_cluster_defining_features(X, clustering_label, cluster_setting_name):
#     """Need to provide X as features only and NO clustering label """
#     cluster_names = np.unique(clustering_label)
#     final_dict = {}
#     all_cluster_info_mean_peaks_per_cluster = []

#     for cluster_name in tqdm(cluster_names):
#         final_dict[cluster_name] = {}
#         cluster_one_v_all_labels = (clustering_label == cluster_name).astype(
#             int
#         )
#         # use scale_pos_weight to address imbalance between clustering class vs rest
#         all_OnevsAll_models_result_dict = repeat_train(
#             X,
#             cluster_one_v_all_labels,
#             num_times=1000,
#             different_train_test_split=True,
#             name=cluster_setting_name,
#             scale_pos_weight=(cluster_one_v_all_labels == 0).sum()
#             / (cluster_one_v_all_labels == 1).sum(),
#             show_output=False,
#         )

#         final_dict[cluster_name]["AUC_fig"] = all_OnevsAll_models_result_dict[
#             "fig"
#         ]

#         shap_mean_peaks_per_cluster_all_models = pd.concat(
#             [
#                 result["shap_feature_mean_per_cluster"].assign(model_num=i)
#                 for i, result in enumerate(
#                     all_OnevsAll_models_result_dict["result_list"]
#                 )
#                 if result["metrics"]["test"]["AUC"] > 0.9
#             ]
#         ).reset_index()
#         # explainer = shap.TreeExplainer(shap_mean_peaks_per_cluster_list[0][2])

#         final_dict[cluster_name]["shap_summary_plot"] = shap.summary_plot(
#             shap_mean_peaks_per_cluster_all_models.drop(columns=["model_num"])
#             .groupby("index")
#             .mean()
#             .values,
#             X[
#                 shap_mean_peaks_per_cluster_all_models.drop(
#                     columns=["index", "model_num"]
#                 ).columns
#             ],
#         )

#         aggregate_peak_info_mean_peaks_per_cluster = (
#             X[
#                 shap_mean_peaks_per_cluster_all_models.drop(
#                     columns=["index", "model_num"]
#                 ).columns
#             ]
#             .assign(
#                 is_cluster=np.where(
#                     cluster_one_v_all_labels == 1,
#                     "cases_in_cluster",
#                     "cases_not_in_cluster",
#                 )
#             )
#             .groupby("is_cluster")
#             .agg(["mean", "median"])
#             .swaplevel(-2, -1, axis=1)
#         )
#         aggregate_peak_info_mean_peaks_per_cluster = pd.concat(
#             [
#                 aggregate_peak_info_mean_peaks_per_cluster["mean"].T.add_suffix(
#                     "_mean"
#                 ),
#                 aggregate_peak_info_mean_peaks_per_cluster[
#                     "median"
#                 ].T.add_suffix("_median"),
#             ],
#             axis=1,
#         )
#         aggregate_peak_info_mean_peaks_per_cluster

#         cluster_info_mean_peaks_per_cluster = (
#             shap_mean_peaks_per_cluster_all_models.drop_duplicates("model_num")
#             .drop(columns=["index", "model_num"])
#             .count()
#             .to_frame()
#         )
#         cluster_info_mean_peaks_per_cluster.columns = [
#             "num_model_used_this_feature"
#         ]
#        cluster_info_mean_peaks_per_cluster = cluster_info_mean_peaks_per_cluster.
#        merge(
#            aggregate_peak_info_mean_peaks_per_cluster,
#            how="outer",
#            left_index=True,
#            right_index=True,
#        )
#         cluster_info_mean_peaks_per_cluster["cluster"] = cluster_name

#         final_dict[cluster_name][
#             "cluster_info_feature_mean_per_cluster"
#         ] = cluster_info_mean_peaks_per_cluster
#         all_cluster_info_mean_peaks_per_cluster.append(
#             cluster_info_mean_peaks_per_cluster
#         )

#     final_dict["combined_cluster_info_feature_mean_per_cluster"] = pd.concat(
#         all_cluster_info_mean_peaks_per_cluster
#     )
#     final_dict["cluster_setting"] = cluster_setting_name
#     return final_dict
