3. Exploratory data analysis (EDA)
==================================

The notebook used to generate the findings in this section :notebooks:`can be found here on GitHub<00_eda_and_clean_data.ipynb>`.

.. contents:: In this section
  :local:
  :depth: 2
  :backlinks: top

Projects distributed by various attributes
------------------------------------------

To give a better sense of the nature of the data contained in this dataset, it is useful to provide some visual representations of the types of projects available. Shown below are a set of barplots illustrating the distribution of the full set of 355 projects in the cleansed dataset, categorized on several different dimensions.

First, here are the projects distributed among the set of project categories as they were assigned in this initial dataset. This plot demonstrates the far greater proportion of projects designated as "Streets and Roadways" versus all other categories. It is also worth noting the very low count numbers for the several smallest categories. For the purpose of this analysis, we will merge several of these smaller groupings with other like groups to more evenly distribute the projects and to reduce the overall number of categories.

.. figure:: ../../docs/_static/figures/01-projects-by-cat-barplot.jpg
  :align: center
  :width: 100%

  Figure 1: Capital projects by category

This next plot illustrates the managing agency defined for each project. Once again, we can see that the most frequently occuring value appears at a far greater rate than all other agencies. This agency, `the Department of Design and Construction (DDC) <https://www1.nyc.gov/site/ddc/about/about-ddc.page>`_, is considered to be NYC's primary capital construction project manager. Therefore, the relatively large proportion of projects under the DDC's management is not necessarily surprising. Likewise, the second most frequent managing agency, `the Department of Transportation (DOT) <https://www1.nyc.gov/html/dot/html/about/about.shtml>`_, is also not surprising considering the large proportion of "Streets and Roadways" projects contained within the dataset.

.. figure:: ../../docs/_static/figures/02-projects-by-agency-barplot.jpg
  :align: center
  :width: 100%

  Figure 2: Capital projects by managing agency

.. figure:: ../../docs/_static/figures/03-projects-by-borough-barplot.jpg
  :align: center
  :width: 100%

  Figure 3: Capital projects by NYC borough

.. figure:: ../../docs/_static/figures/04-projects-by-changes-barplot.jpg
  :align: center
  :width: 100%

  Figure 4: Capital projects by number of project change records

.. _figure5:

.. figure:: ../../docs/_static/figures/05-projects-by-age-barplot.jpg
  :align: center
  :width: 100%

  Figure 5: Capital projects by age of project at time of analysis

As can be seen in the horizontal barplots above, there were several categorical features available for each project. However, the categories provided were highly imbalanced, and as was the case with NYC borough designations for projects, not all categories were exclusive. Some categories overlapped and there were in some instances duplicative categories based on different naming conventions.

Additionally, in the final plot above, we can easily see illustrated supporting evidence for why 3 years was an ideal interval to select for our predictive analysis.

Distribution of project change data
-----------------------------------

Now for a scatter matrix illustrating the correlative relationships of all quantitative variables in our dataset.

.. figure:: ../../docs/_static/figures/06-features-scatter-matrix.jpg
  :align: center
  :width: 100%

  Figure 6: Distribution of budget and duration change features by project

  (Click on image for more detail.)

As can be seen scatterplots above, many of the quantitative variables are heavily skewed with extreme outliers, particularly for budget-related metrics. There are also a number of variables with week correlation including relationships between starting budgets and schedules, as well as ending budgets and schedules. The variables exhibiting the greatest levels of correlation various change metrics that we created during our initial investigation of how to measure project change over our 3-year interval. Therefore, it would be expected that those features would ehibit high levels of correlation, and not particularly troubling, because those competing metrics will not likely coexist in any model that we build.

.. figure:: ../../docs/_static/figures/07-project-start-hist.jpg
  :align: center
  :width: 100%

  Figure 7: Distribution of projects by originally budgeted project cost and originally scheduled project duration

.. figure:: ../../docs/_static/figures/08-project-change-hist.jpg
  :align: center
  :width: 100%

  Figure 8: Distribution of projects by forecasted changes to project budget and project duration

.. figure:: ../../docs/_static/figures/09-project-change-ratio-hist.jpg
  :align: center
  :width: 100%

  Figure 9: Distribution of projects by ratio of original vs. reforecasted change to project budget and project duration

As was highlighted in our analysis of the scatter matrix above, our core quantitative features related to budget and schedule are heavily shewed with extreme outliers. In addition, there are extreme difference in the scale of the values measured by each metric (i.e. budget is measured in hundreds of millions of dollars, while schedule is measured in thousands of days). These side-by-side histograms illustrate the severity of this problem.

These findings tell us that we will need to take great care in both scaling as well as transforming our quantitative predictors to mitigate these issues, particularly for classes of models where this will pose a major issue. 

Project change trends
---------------------

Now let's looks at the change trends for some specific projects in our dataset.

.. figure:: ../../docs/_static/figures/10-project-603-trend.jpg
  :align: center
  :width: 100%

  Figure 10: Project change trend for project 603

.. figure:: ../../docs/_static/figures/11-project-480-trend.jpg
  :align: center
  :width: 100%

  Figure 11: Project change trend for project 480

.. figure:: ../../docs/_static/figures/12-project-96-trend.jpg
  :align: center
  :width: 100%

  Figure 12: Project change trend for project 96

.. figure:: ../../docs/_static/figures/13-project-482-trend.jpg
  :align: center
  :width: 100%

  Figure 13: Project change trend for project 482

As is illustrated by these time series of individual project changes above, we can see the varying degrees to which project budgets and durations change relative to one another within any singular project.

While the form of the data we will be using will represent a starting snapshot of each project as well as a 3-year snapshot of each project at the end of the prediction interval (effectively removing change-to-change variability over that period), these time trends tell us that budget changes and schedule changes, as they occur over time, appear to exhibit very little correlation to one another. Often times, counterintuitive and opposite changes in schedule duration appear to accompany comparable changes in budget.

It is definitely interesting to see how this seemingly independent behavior between budget changes and schedule changes manifests itself as we continue this analysis and begin fitting models to our data.

Now, armed with the basic underpinnings we have identified during our initial EDA and data cleansing activies, we are ready to define our research question and begin our model engineering process.
