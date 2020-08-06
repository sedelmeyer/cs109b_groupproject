.. _intro:

1. About this analysis
======================

.. contents:: In this section
  :local:
  :depth: 2
  :backlinks: top

About this project
------------------

This site contains an exploratory analysis, wherein we attempt different approaches to modeling and feature engineering to predict 3-year outcomes for New York City managed capital projects with budgets greater than $25 million using `an openly available dataset hosted by NYC Open Data <datasource_>`_. Due to the limited scope and amount of data available in this dataset, this project can only be thought of as a proof of concept, which will need to replicated under more stringent standards with a larger set of more robust data.

.. _datasource: https://data.cityofnewyork.us/City-Government/Capital-Projects/n7gv-k5yt


.. note::

    **This project's online documentation is being actively developed.** Therefore this site currently contains little more than a starting skeleton with which to document the project and its findings as they are developed.

    * To view the in-depth analysis and findings associated with this project, the best resource to review will be `the Jupyter notebook located here <https://github.com/sedelmeyer/nyc-capital-projects/blob/master/notebooks/11_FINAL_REPORT.ipynb>`_.

Summary of initial findings
---------------------------

Overall, we found promising results that, given a limited amount of project data, moderately accurate predictions could be made as to the 3-year budget change ratio and schedule change ratio for changes made on NYC capital projects.

Modeling methods
^^^^^^^^^^^^^^^^

To acheive this, we explored a number of modeling methods ranging from basic linear regression (as an initial baseline), to smoothing spline generalized additive models (GAMs), to non-parameteric ensemble methods using decision tree regressors and boosting.

Feature engineering
^^^^^^^^^^^^^^^^^^^

In addition, using such a limited dataset, we sought to extract as much predictive information as possible from it by focusing heavily on feature engineering. Methods for feature engineering included competing methods for generating latent "reference class" categories for each project using K-means clustering as well as uniform manifold approximation and projection (UMAP) in combination with the HDBSCAN clustering algorithm. These reference class clustering algorithms took into account all categorical and quantitative characteristics of our training projects. Additionally, we used bidirectional encoder representations from transformers (BERT) embeddings of the textual information available for each project. We further encoded those embeddings into a smaller, more usable, feature-space through competing methods of dimensionality reduction including principal compenent analysis (PCA), latent space encoding with a dense autoencoder neural network, and UMAP.

Predictive results
^^^^^^^^^^^^^^^^^^

Not surprisingly, linear models lacked the expressiveness needed to successfully model our predictions, particularly those for the "budget ratio" change of our test projects. Not only were there significant differences in the predictive performance of models predicting **budget change ratio** versus those predicting **schedule change ratio**, but models also needed to be individually parameterized for optimal predictions on each of those two response variables.

In the end, through the use of ensemble decision tree regressors using boosting, we were able to achieve test :math:`R^2` scores of :math:`0.48` on our budget change ratio predictions and :math:`0.70` on our schedule change ratio predictions.

While there is still a great deal of additional work we can put towards this problem, we feel that this analysis is a good starting point for more principled exploration.
