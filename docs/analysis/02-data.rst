2. About the data
=================

The original version of the report from which this 

.. contents:: In this section
  :local:
  :depth: 2
  :backlinks: top

The notebooks used to generate the findings in this section can be :notebooks:`found here on GitHub<00_eda_and_clean_data.ipynb>` and :notebooks:`here on GitHub<01_generate_data_splits.ipynb>`.

Data Source
-----------

The focus of this analysis is to determine what methods might be available for predicting project success on large capital projects.

To investigate this question, we perform an in-depth analysis and modelling exercise using a dataset containing project-change records for currently active City of New York (NYC) capital projects. This dataset describes all major infrastructure and information technology projects with a budget of $25 million or more that were active (i.e. in their design, procurement, or construction phases) at the time the report was generated. The version of the dataset used throughout this report contained project data that was reported as of September 1st, 2019.

* If you wish to access the latest version of the data available from the NYC website, or care to read more about the collection and availability of the data, it can be `accessed online here <https://data.cityofnewyork.us/City-Government/Capital-Projects/n7gv-k5yt>`_.

* If you wish to view the September 1st, 2019 version of the data used for this particular analysis, it can be viewed in the ``data/raw/`` :datadir:`sub-directory of the accompanying GitHub repository<raw>`.

Level of observation
--------------------

The original raw data provided by the NYC OpenData website contained 2,259 records, each of which represent one change (either budgetary or scheduling) issued to an active NYC capital project. Therefore, the original raw (i.e. *uncleansed*) data from which this analysis was built contained 2,259 project change records, comprising 378 unique activate projects.

Data cleansing and missingness
------------------------------

We encountered a high degree of missingness in this original raw data. Luckily, much of this missing data could be easily addressed by populating project information and calculating sequential record changes in instances where the required information could be identified through other project change records for the same unique project. After the process of cleansing the data and populating missing data where possible, we ultimately ended up with a set of 2,095 valid change records, comprising 355 unique projects.

Subsetting for a valid prediction time-interval
-----------------------------------------------

Because our objective in this analysis is to make project-outcome predictions of some sort, we needed to take into consideration the fact that all projects in this dataset should be considered "currently active." To accomodate this fact, we decided to limit our prediction interval to changes that occured during the first 3 years of any given project. Therefore, if we are thinking of this problem in terms of outcomes, we would predict the outcome of a project after 3 years given the initial characteristics available at the start of that project. This 3-year interval period was selected by empirical exploration of the data. The limiting factor that we identified was that any interval any longer than 3 years would not provide a large enough subset of available projects for us to perform a comprehensive analysis. In addition, we felt that predictions for any interval shorter than 3 years would likely be of limited practical use given the scope and long-term duration of these $25+ million capital projects.

Therefore, after we further subsetted our dataset to account for projects that could statisfy this 3-year interval requirement, we were left with only 149 unique projects remaining.

* For a visual representation of the distribution of projects by each project's "age" in our original 355-project cleansed dataset, :ref:`please see Figure 5 located in the EDA section of this analysis<figure5>`.

Record count summary
--------------------

To give a more concise picture of the data and how few records remained after cleansing and transforming the raw data into a usable 3-year-interval format for our analysis, below are listed the numbers of records at each step in this process.

The original **raw data**, prior to removing records for which missing data could not be fixed::

   Number of project-change records:    2259
   Number of unique projects:           378

The **cleansed data**, after the remaining records with missing data were removed::

   Number of project-change records:    2095
   Number of unique projects:           355

After subsetting and transforming this cleansed data into a dataset containing **3-year-interval data** for all projects that were 3-years of age or older::

   Number of unique projects:           149

Rather than containing individual project-change records, this 3-year-interval dataset consisted of one record for each unique project. Each of those project records contained data fields describing (1) the project at its start, (2) the project at the end of the 3-year interval, and (3) changes that took place over the course of that interval. 

Lastly, for the purposes of our analysis, we split that 3-year-interval dataset into training and test splits. The resulting **train-test-split** used throughout this analysis consisted of the following number of 3-year-interval project records::

   Number of training projects:        134
   Number of TEST projects:            15

For further reference
---------------------

* For additional information on the data-cleansing activities used to inspect and clean this dataset, please see :notebooks:`the accompanying data-cleansing and EDA notebook<00_eda_and_clean_data.ipynb>`.
* For additional information on the processes and calculations used to generate this 3-year-interval dataset, please see :notebooks:`the accompanying notebook where the interval was generated<01_generate_data_splits.ipynb>`, as well as :src:`the source code for the module<datagen.py>` ``caproj.datagen`` used to generate the interval.
* For an overview of the data fields available in the finished training and TEST interval dataset used for modeling, please refer to :ref:`the data dictionary shown in the "Research question and predictive features" section of this analysis<data-dict>`.