# Dataset Creation Scripts

This directory contains the scripts to create datasets, their tasks, and aggregate their metadata.

## Creating New Datasets and Tasks

* The `datasets` directory contains a sub-dir per datasets. Each sub-dir contains code to
  preprocess the original data source and the .yaml file with all dataset-specific required metadata.
* The `_template.yaml` contains a template for the metadata file.
* The `dataset_check_utils.py` contains code for checking the dataset (see
  `datasets/anneal/anneal.py` for example usage).
* The `create_openml_datasets_and_tasks.py` contains code to upload datasets and tasks from the
  metadata and results files per sub-dir in `datasets`.

## Creating Metadata Aggregates

* The `metadata` directory contains various automatically or manually curated metadata files.
* The `create_datasets_metadata.py` script creates and aggregates metadata in a .csv and .text table saved in the
  `metadata` directory.
* The `create_openml_suite.py` script creates a suite of datasets and tasks for OpenML.