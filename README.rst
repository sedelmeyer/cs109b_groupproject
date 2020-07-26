NYC Capital Projects
====================

This project is an exploratory analysis experimenting with different approaches to modeling and feature engineering to predict budget and schedule outcomes for New York City-managed capital projects with budgets greater than $25 million, using a `data set hosted by NYC Open Data`_.


.. image:: https://github.com/sedelmeyer/nyc-capital-projects/workflows/build/badge.svg?branch=master
    :target: https://github.com/sedelmeyer/nyc-capital-projects/actions

.. image:: https://img.shields.io/badge/License-MIT-black.svg
    :target: https://github.com/sedelmeyer/cc-pydata/blob/master/LICENSE

.. contents:: Contents
  :local:
  :depth: 1
  :backlinks: none

Summary
-------

This analysis is built upon one initially completed as a final project for `CS109B: Advanced Topics In Data Science`_, a course offered by Harvard University's John A. Paulson School of Engineering and Applied Sciences (SEAS). The authors of that original project are:

- `An Hoang <https://github.com/hoangthienan95>`_
- `Mark McDonald <https://github.com/mcdomx>`_
- `Mike Sedelmeyer <https://github.com/sedelmeyer>`_

That original project can be found on GitHub at: https://github.com/mcdomx/cs109b_groupproject. The final report summarizing the methods and findings for that project can be found in the `Jupyter notebook-based report for that project <https://github.com/mcdomx/cs109b_groupproject/blob/master/notebooks/Module-E-final-report-Group71.ipynb>`_.


Research question
^^^^^^^^^^^^^^^^^

After initial exploration and cleansing of the available data, modeling efforts focus on the following research question:

- *Using very limited historical project data, can we predict how much the forecasted budget and duration of any given capital project run by the City of New York will deviate from its original budgeted estimates by the end of year-3 for the project?*


Analysis and findings
---------------------

The analysis and findings associated with this project can be found here:

https://sedelmeyer.github.io/nyc-capital-projects


Source code documentation
-------------------------

Documentation for the python modules built specifically for this analysis (i.e. modules located in the ``./src/`` directory of this project) can be found here:

https://sedelmeyer.github.io/nyc-capital-projects/modules.html

.. _replication:

Replicating this analysis
-------------------------

In order to replicate this analysis and run the Python code available in this analysis locally, follow these steps:

.. contents:: In this section
  :local:
  :backlinks: none

.. todo::

    * Below is a placeholder template containing typical steps required to replicate a PyData project.
    * Content must be added to each section, outlining requirements and explaining how to replicate the analysis locally

0. Ensure system requirements are met
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone this repository locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2. Install the required environment using Pipenv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3. Run the data pipeline and analysis workflows locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4. Explore the associated Jupyter notebooks included with this project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _development:

Adding to this project
----------------------

If you'd like to build off of this project to explore additional methods or to practice your own data science and development skills, below are some important notes regarding the configuration of this project.

.. contents:: In this section
  :local:
  :backlinks: none

.. todo::

    * Below are placeholder sections for explaining important characteristics of this project's configuration.
    * This section should contain all details required for someone else to easily begin adding additional development and analyses to this project.

Project repository directory structure, design, and usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python package configuration and associated workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing
^^^^^^^

Version control and git workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Documentation using Sphinx and reStructuredText
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _issues:

Questions or issues related to this project
-------------------------------------------

.. todo::

    * Add details on the best method for others to reach you regarding questions they might have or issues they identify related to this project.


.. _sources:

Sources and additional resources
--------------------------------

.. todo::

    * Add links to further reading and/or important resources related to this project.

.. _data set hosted by NYC Open Data: https://www1.nyc.gov/site/capitalprojects/dashboard/category.page?category=All%20Capital%20Projects

.. _`CS109b: Advanced Topics In Data Science`: https://harvard-iacs.github.io/2020-CS109B/
