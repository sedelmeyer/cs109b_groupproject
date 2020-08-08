8. Smoothing spline generalized additive models (GAMs)
======================================================

The unabridged notebook used to generate the findings in this section can be `found here on GitHub <https://github.com/sedelmeyer/nyc-capital-projects/blob/master/notebooks/08_smoothing_spline_models.ipynb>`_.

.. contents:: In this section
  :local:
  :depth: 2
  :backlinks: top

Smoothing spline GAMs with baseline predictors
----------------------------------------------

As a first step toward developing our smoothing spline GAMs, we will train a model using the same baseline set of predictors we used in our baseline linear regression model in section 4.1. above.

Identifying the optimal values :math:`\lambda`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using gridsearch to find optimal value :math:`\lambda` for each term in the smoothing spline GAM model:

* Here we treat each y output independently, partly because PyGam's Linear GAM will not fit a multi-output model, but mostly because each y output behaves differently and we have found that different :math:`\lambda` penalty values are required to optimize outputs for both of our response variables.

* In the coded cell below, we use PyGam's native ``gridsearch`` method to choose our values :math:`\lambda` for each term.
  
  * Because there are so few instances of some project categories, traditional cross-validation using k-splits creates split-instances where some categories are missing from our training splits
  
  * When that occurs, PyGam cannot fit a coefficient to that category and generates an error.

.. code-block::

    GAM gridsearch results for BUDGET_CHANGE_RATIO prediction model:

    LinearGAM                                                                                                 
    ================================= =================================
    Distribution:          NormalDist Effective DoF:            18.7569
    Link Function:       IdentityLink Log Likelihood:         -419.7587
    Number of Samples:            134 AIC:                     879.0312
                                      AICc:                    886.2739
                                      GCV:                      11.5842
                                      Scale:                     8.7077
                                      Pseudo R-Squared:          0.8263
    ===================================================================
    Feature Function  Lambda      Rank    EDoF   P > x       Sig. Code   
    ================= =========== ======= ====== =========== ==========
    s(0)              [0.001]     20      17.4   1.11e-16    ***         
    s(1)              [100000.]   20      1.1    9.77e-01
    f(2)              [215.4435]  11      0.3    2.42e-01
    intercept                     1       0.0    7.88e-15    ***         
    ===================================================================
    Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    WARNING: Fitting splines and a linear function to a feature
             introduces a model identifiability problem which can cause
             p-values to appear significant when they are not.

    WARNING: p-values calculated in this manner behave correctly for
             un-penalized models or models with known smoothing
             parameters, but when smoothing parameters have been
             estimated, the p-values are typically lower than they
             should be, meaning that the tests reject the null too
             readily.


