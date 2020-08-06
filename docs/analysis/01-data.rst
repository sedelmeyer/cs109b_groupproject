2. About the data
=================

.. contents:: Contents
  :local:
  :depth: 2
  :backlinks: top

The notebooks used to generate the findings in this section can be `found here on GitHub <https://github.com/sedelmeyer/nyc-capital-projects/blob/master/notebooks/00_eda_and_clean_data.ipynb>`_ and `here on GitHub. <https://github.com/sedelmeyer/nyc-capital-projects/blob/master/notebooks/01_generate_data_splits.ipynb>`_

Data Source
-----------

The focus of this analysis is to determine what methods might be available for predicting project success on large capital projects.

To investigate this question, we perform an in depth analysis and modelling exercise using a dataset containing project-change records for currently active City of New York (NYC) capital projects. The dataset describes all major infrastructure and information technology projects with a budget of $25 million or more that were currently active (in the design, procurement, or construction phase). The version of that dataset used throughout this report contained project data that was reported as of September 1st, 2019.

If you wish to access the latest version of the data available from the NYC website, or care to read more about the collection and availability of the data, it can be `accessed online here <https://data.cityofnewyork.us/City-Government/Capital-Projects/n7gv-k5yt>`_.

Level of observation
--------------------

The raw data provided by the NYC website contained 2,259 records, each of which represent one change (either budgetary or scheduling) issued to an active NYC capital project. Therefore, the original raw data contained 2,259 project change records inclusive of 378 unique activate projects.

Data cleansing
--------------

We encountered a high degree of missingness in this original raw data, and after the process of cleansing the data, populating project information and change information in instances where it could be identified through other project change records, we ultimately ended up with a set of 2,095 valid change records comprising 355 unique projects.

Subsetting for a valid prediction time-interval
-----------------------------------------------

Furthermore, because all projects in this dataset are currently, and if our objective is to make project-outcome predictions of some sort, we ultimately decided to limit our prediction interval to changes that occur during the first 3 years of a project. Therefore, it we are thinking of this problem in terms of outcomes, we would predict the outcome of a project after 3 years given it's initial characteristics at project start time. This 3-year interval was determined by empirical exploration of the data, and the limiting factor that an interval any longer than 3 years would not provide a large enough subset of available projects and any interval shorter than that would likely be of minimal usefulness given the scope and long-term duration of these $25+ million capital projects.

Therefore, after we further subsetted our dataset to account for projects that could statisfy this 3-year interval requirement, we were left with only 149 unique projects.

Record count summary
--------------------

To give a more concise picture of the data and how few records remained after cleansing and transforming the raw data into a usable 3-year-interval format for our analysis, below are listed the numbers of records at each step in this process.

For the *original* cleansed data, containing all available NYC capital projects change records::

   Number of dataset records: 2095
   Number of unique projects in dataset: 355

For the data containing start and end data for all available NYC capital projects for the *entire interval* of changes covered in the *original* data::

   Number of dataset records: 355
   Number of unique projects in dataset: 355

For the *final training* data, containing the training split of 3-year project data used in this analysis::

   Number of dataset records: 134
   Number of unique projects in dataset: 134

For the *final test* data, containing the test split of 3-year project data used in this analysis::

   Number of dataset records: 15
   Number of unique projects in dataset: 15
