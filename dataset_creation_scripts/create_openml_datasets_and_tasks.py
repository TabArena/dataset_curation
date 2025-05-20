from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import openml
import pandas as pd
import yaml
from openml.datasets.functions import create_dataset

TESTING = True

if TESTING:
    openml.config.start_using_configuration_for_example()
else:
    openml.config.apikey = ""  # TODO: use your own
    openml.config.server = "https://api.openml.org/api/v1/xml"
    openml.config.cache_directory = "./openml_cache"

EST_ID_10_REPEATED_3_FOLD_CV_STRATIFIED = 31
EST_ID_10_REPEATED_3_FOLD_CV = 32


def generate_description_from_metadata(dataset_metadata: dict) -> str:
    """Template for dataset description."""
    return f"""
This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is {dataset_metadata["problem_type"]}.

---
#### Dataset Metadata
- **Licence:** {dataset_metadata["licence"]}
- **Original Data Source:** {dataset_metadata["original_dataset_source"]}
- **Reference (please cite)**: {dataset_metadata["reference"]}
- **Dataset Year:** {dataset_metadata["dataset_year"]}
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
{dataset_metadata["curation_comments"]}
"""


def _normalize_text_for_openml(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_encoded = normalized.encode("ascii", "ignore")
    return ascii_encoded.decode("ascii")


def _is_uri(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def create_dataset_from_yaml(
    dataset_dir: Path, alternative_csv_path: None | Path = None
) -> tuple[int, str, bool, str]:
    """Upload a dataset from our curated version of the data."""
    dataset_name = dataset_dir.name

    # Filter hack to skip, to be implemented better
    # if dataset_name not in ["X"]:
    #     return []

    # Read the YAML file
    with (dataset_dir / f"{dataset_name}.yaml").open("r") as file:
        dataset_metadata = yaml.safe_load(file)

    csv_file_dir = dataset_dir if alternative_csv_path is None else alternative_csv_path
    data = pd.read_csv(csv_file_dir / f"{dataset_name}.csv", index_col=False)

    is_classification = dataset_metadata["problem_type"] == "classification"
    target_feature = dataset_metadata["target_feature"]
    categorical_features = dataset_metadata["categorical_features"]

    # Get some basic metadata (only for sanity/debugging)
    try:
        print(
            dataset_name,
            dataset_metadata["problem_type"],
            len(np.unique(data[target_feature])),
            len(data.columns),
            len(data),
            dataset_metadata["dataset_year"],
        )
    except Exception as e:
        print(dataset_name)
        raise e

    missing_cols = [c for c in categorical_features if c not in data.columns]
    assert all(c in data.columns for c in categorical_features), (
        f"Not all categorical features are in the data {missing_cols}"
    )
    assert target_feature in data.columns, "Target feature is not in the data"
    assert target_feature not in categorical_features, (
        "Target feature is wrongly part of the categorical features list"
    )

    if is_classification:
        categorical_features.append(target_feature)

    if categorical_features:
        data[categorical_features] = (
            data[categorical_features].astype(str).astype("category")
        )

    # Fix invalid symbols for OpenML
    #   - This will remove some ASCII symbols.
    #   - It will also sadly butcher some author names (e.g., "é" → "e") :( - cannot fix
    dataset_metadata["reference"] = _normalize_text_for_openml(
        dataset_metadata["reference"]
    )

    new_dataset = create_dataset(
        name=dataset_name,
        description=generate_description_from_metadata(dataset_metadata),
        creator="See original data source.",
        contributor="TabArena Team",
        collection_date=dataset_metadata["dataset_year"],
        language="English",
        licence=dataset_metadata["licence"],
        attributes="auto",
        data=data,
        default_target_attribute=target_feature,
        ignore_attribute=None,
        citation=dataset_metadata["reference"],
        row_id_attribute=None,
        original_data_url=dataset_metadata["original_dataset_source"]
        if _is_uri(dataset_metadata["original_dataset_source"])
        else "See Description",
        paper_url="https://tabarena.ai/paper-tabular-ml-iid-study",
        version_label="tabular-ml-iid-study-0.0.1",
        update_comment=None,
    )
    new_dataset.publish()
    print(f"Published {dataset_name} at {new_dataset.openml_url}")

    return new_dataset.id, target_feature, is_classification, dataset_name


def run_create_all_datasets(alternative_csv_path: None | Path = None) -> None:
    """Upload all datasets to OpenML.

    Parameters
    ----------
    alternative_csv_path : None | Path
        If given, use this path is used to find the .csv files for the dataset instead
        of looking next to the .yaml file.
    """
    path_to_datasets_folder = "./datasets/"
    created_dataset_metadata = []

    for dataset in Path(path_to_datasets_folder).glob("*/"):
        dataset_dir = Path(dataset)
        created_dataset_metadata.append(
            create_dataset_from_yaml(
                dataset_dir=dataset_dir, alternative_csv_path=alternative_csv_path
            )
        )
        print(
            f"Created {len(created_dataset_metadata)} datasets:\n{created_dataset_metadata}"
        )

    with open("metadata/created_datasets.json", "w") as file:
        json.dump(created_dataset_metadata, file)


def run_create_all_tasks():
    """We create tasks for all datasets in the created_datasets.json.

    We expect the created_datasets.json to be in the same directory as this script and
    contain a list of tuples with (openml_dataset_id, target_feature,
    is_classification).
    """
    with open("metadata/created_datasets.json") as file:
        created_dataset_metadata = json.load(file)

    all_created_tasks = []
    for (
        dataset_id,
        target_feature,
        is_classification,
        dataset_name,
    ) in created_dataset_metadata:
        print(dataset_id)
        estimation_procedure_id = (
            EST_ID_10_REPEATED_3_FOLD_CV_STRATIFIED
            if is_classification
            else EST_ID_10_REPEATED_3_FOLD_CV
        )
        task_type = (
            openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
            if is_classification
            else openml.tasks.TaskType.SUPERVISED_REGRESSION
        )
        task = openml.tasks.create_task(
            task_type=task_type,
            dataset_id=dataset_id,
            target_name=target_feature,
            estimation_procedure_id=estimation_procedure_id,
        )
        try:
            task.publish()
            task_id = task.task_id
            print(f"Published Task with Task id: {task_id}")
        except openml.exceptions.OpenMLServerException as e:
            import re

            match = re.search(r"\[(\d+)\]", e.message)

            if not match:
                raise e

            task_id = int(match.group(1))
            print(f"Task already exists as {task_id} - skipping.")

        all_created_tasks.append(
            (dataset_id, task_id, target_feature, is_classification, dataset_name)
        )

    print(all_created_tasks)
    pd.DataFrame(
        all_created_tasks,
        columns=[
            "dataset_id",
            "task_id",
            "target_feature",
            "is_classification",
            "dataset_name",
        ],
    ).to_csv("metadata/created_tasks.csv", index=False)


if __name__ == "__main__":
    run_create_all_datasets()  # or use: alternative_csv_path=Path("./final_csvs")
    run_create_all_tasks()
