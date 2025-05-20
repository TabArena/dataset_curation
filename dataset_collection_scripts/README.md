# Dataset Collection Scripts

For some prior benchmarks, we were able to check duplicates via code.
Moreover, for others we had to extract the used datasets from the source code.
This directory contains the scripts for this purpose.

* `openml_suite_benchmark_datasets.py` verifies and checks that we included all uniuqe dataset IDs from TabRepo,
  OpenML-CC18, OpenML-CTR23, and the AutoML Benchmark.
* `pmlb_datasets.py` verifies the uniqueness of names of datasets in PMLB and PMLBMini.
* `pytabkit_datasets.py` maps UCI datasets used in PyTabKit to unique and new UCI DOIs for our curation effort. 