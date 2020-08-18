4. Research question and predictive features
============================================

.. contents:: In this section
  :local:
  :depth: 2
  :backlinks: top


Research question
-----------------

After initial exploration and cleansing of the available data, we now focus our efforts on the following research question:

    *Given the set of New York City Capital Projects change data, can we create a model that can accurately predict 3-year change in forecasted project budget and 3-year change in forecasted project duration using only the data available at the start of the project as our predictors?*

In other words, using historical project data, can we predict how much the forecasted budget and duration of any given capital project run by the City of New York will deviate from it's original budgeted estimates by the end of year-3 for the project?

The significance of a model that can accurately address this question means, given any new project, project managers and city administrators would have another tool at their disposal for objectively identifying potential budget and schedule risk at the start of a new city-run capital project. Such a tool can help to overcome common planning fallacies and optimism biases to help to mitigate cost and and schedule overruns.

Response variables
------------------

Throughout the remainder of this analysis, the specific response variables we will be seeking to predict are:

1. ``Budget_Change_Ratio`` as defined by the total forecasted budget change (in dollars) for a project experienced during the 3-year interval divided by the initial starting budget of the project (i.e. ``Budget_Start``) 

2. ``Schedule_Change_Ratio`` as defined by the total scheduled duration change (in days) for a project experienced during the 3-year interval divided by the initial start scheduled duration of the project (i.e. ``Duration_Start``) 

Therefore, we are seeking to predict 2 response variables with all of the modelling methods investigated in this analysis.

Feature engineering
-------------------

As mentioned in :ref:`the introduction to this analysis<intro>`, there are very few features available at the start of each project and in their original form many of the individual features taken by themselves are either problematic or not that useful. Therefore, in order make the most of the limited number of features available to us, we realized very quickly that creative feature engineering be an important part of any effective predictive model using this data.

Approaches to feature engineering undertaken in the next several sections of this analysis include (1) competing methods for generating latent "reference class" categories for each project using K-means clustering (:ref:`see Section 5<cluster1>`) as well as uniform manifold approximation and projection (UMAP) in combination with the HDBSCAN clustering algorithm (:ref:`see Section 6<cluster2>`), which take into account all categorical and quantitative characteristics of our training projects. Additionally, (2) we use bidirectional encoder representations from transformers (BERT) embeddings of the textual information available for each project, and further encoded those embeddings into a smaller, more usable feature-space through competing methods of dimensionality reduction such as principal compenent analysis (PCA), latent space encoding with a dense autoencoder neural network, and UMAP (:ref:`see Section 7<embed>`).

.. Note::

  Additional resources providing background on each of the methods listed above can be found in each method's respective section of this analysis.

.. _data-dict:

Data dictionary (final feature set)
-----------------------------------

Shown below is a table containing details for all columns available for modeling in our final dataset. The "Column Use" column in this table indicates the use for each variable.

Column Use categories:

* Indentifier:
  
  These variables were used strictly as identifiers for each project in the dataset.

* EDA:

  These variables were used during the EDA stage of this analysis to gain a better understanding of our data set and to help us determine how to structure our predictive models.

* Feature engineering:

  These variables were used as inputs during the feature engineering stage of this analysis, but were not themselves used in our resulting predictive models.

* Model Predictor:

  These variables were used as inputs for our predictive models. Different models used different subsets of these predictors. To learn which predictors were used for which models, please see the details for each individual model shown in Section 7 through Section 10 of this analysis.

* Response Variable:

  These variables were used as the response variables in our predictive models. Therefore, these are the features we are trying to predict with each of our models.

.. csv-table:: Data Dictionary *(Final 3-year interval dataset)*
   :file: ../reference/data_dict_final_features.csv
   :header-rows: 1
   :stub-columns: 1