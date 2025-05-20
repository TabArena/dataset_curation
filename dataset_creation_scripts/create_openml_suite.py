from __future__ import annotations

import openml
import pandas as pd

TESTING = False

if TESTING:
    openml.config.start_using_configuration_for_example()
else:
    openml.config.apikey = "39e72709e59c772cf544b839fe994228"  # TODO: use your own
    openml.config.server = "https://api.openml.org/api/v1/xml"
    openml.config.cache_directory = "./openml_cache"


def run_create_all_tasks():
    """We create tasks for all datasets in the created_datasets.json.

    We expect the created_datasets.json to be in the same directory as this script and
    contain a list of tuples with (openml_dataset_id, target_feature,
    is_classification).
    """
    metadata_df = pd.read_csv("metadata/created_tasks.csv")
    task_ids = metadata_df["task_id"].tolist()

    study = openml.study.create_benchmark_suite(
        name="TabArena-v0.1 Suite",
        alias="tabarena-v0.1",
        description="""
This suite contains datasets curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: These datasets shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data.
""",
        task_ids=task_ids,
    )
    study_id = study.publish()
    print(f"Published benchmark suite with id {study_id}.")


if __name__ == "__main__":
    run_create_all_tasks()
    # get the suite via: openml.study.get_suite("tabarena-v0.1")
