Changelog
=========

Future releases
---------------

.. todo::

    * Convert notebook ``00_FINAL_REPORT.ipnb`` findings to ``.rst`` format for online Sphinx rendering
    * Generate dependency workflow schematic, illustrating DAG of notebook inputs and outputs
    * Fix notebook ``05_umap_hdbscan_features.ipynb`` data generation functionality
    * Improve ``caproj.utils`` API documentation

v2.0.1 (2020-08-01)
-------------------

* Add all author names to ``docs/conf.py`` and ``setup.py``
* Remove unused notebook-generated outputs from each respective notebook
* Commit ``notebook/05_umap_hdbscan_features.ipynb`` output feature data to version control due to notebook errors preventing successful output
* Move ``notebook/11_FINAL_REPORT.ipynb`` decision tree diagram output to ``docs/_static/figures/`` for use in future documentation enhancements

v2.0.0 (2020-08-01)
-------------------

* Add ``pipenv`` package and dependency management
* Convert custom ``src/`` modules to Python package ``caproj``
* Improve user ``README.rst`` documentation
* Improve documentation within ``.ipynb`` notebook files and fix sequential notebook workflow
* Remove deprecated notebooks, files, and directories
* Add ``tox`` automated testing and continuous integration services with GitHub Actions
* Add ``caproj`` API documentation using Sphinx and reStructuredText
* Publish initial project documentation via GitHub Pages

v1.0.0 (2020-05-17)
-------------------

* Initial version of project completed for CS109B
