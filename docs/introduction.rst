.. _intro:

============
Introduction
============

.. contents:: Contents
  :local:
  :depth: 1
  :backlinks: none

About this project
==================

This site contains an exploratory analysis, wherein we attempt different approaches to modeling and feature engineering to predict 3-year outcomes for New York City managed capital projects with budgets greater than $25 million using `an openly available dataset hosted by NYC Open Data`_. Due to the limited scope and amount of data available in this dataset, this project can only be thought of as a proof of concept, which will need to replicated under more stringent standards with a larger set of more robust data.

.. _an openly available dataset hosted by NYC Open Data: https://www1.nyc.gov/site/capitalprojects/dashboard/category.page?category=All%20Capital%20Projects


A note about supporting code and analyses
-----------------------------------------

Research question
=================

After initial exploration and cleansing of the available data, we have focused our efforts on the following research question:

* Given the set of New York City Capital Projects change data, can we create a model that can accurately predict 3-year change in forecasted project budget and 3-year change in forecasted project duration using only the data available at the start of the project as our predictors?

* In other words, using historical project data, can we predict how much the forecasted budget and duration of any given capital project run by the City of New York will deviate from it's original budgeted estimates by the end of year-3 for the project?

The significance of a model that can accurately address this question means, given any new project, project managers and city administrators would have another tool at their disposal for objectively identifying potential budget and schedule risk at the start of a new city-run capital project. Such a tool can help to overcome common planning fallacies and optimism biases to help to mitigate cost and and schedule overruns.

Summary of Findings
===================

Overall, we found promising results that, given a limited amount of project data, moderately accurate predictions could be made as to the 3-year budget change ratio and schedule change ratio for changes made on NYC capital projects. Ultimately, we explored a number of modeling methods ranging from basic linear regression (as an initial baseline), to smoothing spline generalized additive models (GAMs), to non-parameteric ensemble methods using decision tree regressors and boosting. In addition, using our limited dataset, we sought to extract the greatest amount of predictive information as possible from it by focusing heavily on feature engineering. Methods for feature engineering included competing methods for generating latent "reference class" categories for each project using K-means clustering as well as uniform manifold approximation and projection (UMAP) in combination with the HDBSCAN clustering algorithm, which took into account all categorical and quantitative characteristics of our training project. Additionally, we use bidirectional encoder representations from transformers (BERT) embeddings of the textual information available for each project, and further encoded those embeddings into a smaller, more usable feature-space through competing methods of dimensionality reduction such as principal compenent analysis (PCA), latent space encoding with a dense autoencoder neural network, and UMAP.

Not surprisingly, linear models lacked the expressiveness needed to successfully model our predictions, particularly those for budget ratio change of our test projects. Not only were there drastic differences in the predictive performance of models on both budget change ratio and schedule change ratio as as our primary response variables, but models also needed to be individually parameterized for optimal predictions on each of those two response variables.

In the end, through the use of ensemble decision tree regressors using boosting, we were able to achieve test $R^2$ scores of $0.48$ on our budget change ratio predictions and $0.70$ on our schedule change predictions. While there is still a great deal of additional work we can put towards this problem, we feel that this analysis is a good starting point for more principled exploration.
