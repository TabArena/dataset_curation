from __future__ import annotations

import numpy as np
import openml
import pandas as pd
from dataset_creation_scripts.metadata.dataset_to_metadata_maps import (
    DOMAIN_MAP,
    REF_MAPPING,
    SOURCE_MAP,
    URL_OVERWRITE_MAP,
)
from tqdm import tqdm

if __name__ == "__main__":
    additional_metadata_df = []
    metadata_df = pd.read_csv("metadata/created_tasks.csv")
    tasks = metadata_df["task_id"].tolist()

    for task_id in tqdm(tasks):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        repeats, folds, _ = task.get_split_dimensions()

        n_samples = dataset.qualities["NumberOfInstances"]
        n_features = dataset.qualities["NumberOfFeatures"]
        percentage_cat_features = dataset.qualities["PercentageOfSymbolicFeatures"]
        n_classes = dataset.qualities["NumberOfClasses"]

        if n_classes == 0:
            problem_type = "regression"
        else:
            problem_type = "binary" if n_classes == 2 else "multiclass"

        # From Paper
        if n_samples < 2_500:
            tabarena_repeats = 10
        elif n_samples > 250_000:  # Fallback
            tabarena_repeats = 1
        else:
            tabarena_repeats = 3

        # Bools for constrained methods from paper
        # - 10k training samples (so 2/3 * n_samples)
        can_run_tabpfnv2 = (
            (n_samples <= 15_000) and (n_features <= 500) and (n_classes <= 10)
        )
        # - n_classes != 0 as TabICL only works for classification
        can_run_tabicl = (
            (n_samples <= 150_000) and (n_features <= 500) and (n_classes != 0)
        )

        additional_metadata_df.append(
            [
                task_id,
                dataset.name,
                problem_type,
                n_features,
                n_samples,
                n_classes if n_classes != 0 else np.nan,
                percentage_cat_features,
                folds,
                repeats,
                tabarena_repeats,
                can_run_tabpfnv2,
                can_run_tabicl,
                REF_MAPPING[dataset.name],
                SOURCE_MAP[dataset.name],
                DOMAIN_MAP[dataset.name],
                dataset.collection_date,
                dataset.licence,
                URL_OVERWRITE_MAP.get(dataset.name, dataset.original_data_url),
            ],
        )

    additional_metadata_df = pd.DataFrame(
        additional_metadata_df,
        columns=[
            "task_id",
            "openml_dataset_name",
            "problem_type",
            "num_features",
            "num_instances",
            "num_classes",
            "percentage_cat_features",
            "num_folds",
            "openml_num_repeats",
            "tabarena_num_repeats",
            "can_run_tabpfnv2",
            "can_run_tabicl",
            "reference",
            "data_source",
            "domain",
            "year",
            "licence",
            "original_data_url",
        ],
    )

    # Merge on Task ID
    metadata_df = pd.merge(
        metadata_df, additional_metadata_df, on="task_id", how="left"
    )
    metadata_df.to_csv("metadata/tabarena_dataset_metadata.csv", index=False)

    # Create latex table for paper
    latex_df = metadata_df[
        [
            "dataset_id",
            "task_id",
            "dataset_name",
            "domain",
            "reference",
            "num_instances",
            "num_features",
            "num_classes",
            "percentage_cat_features",
            "can_run_tabpfnv2",
            "can_run_tabicl",
            "original_data_url",
        ]
    ]
    latex_df.loc[:, "reference"] = r"\citep{" + latex_df["reference"] + "}"
    latex_df.columns = [
        "Dataset ID",
        "Task ID",
        "Name",
        "Domain",
        "Ref.",
        "N",
        "d",
        "C",
        r"\% cat",
        "TabPFNv2",
        "TabICL",
        "Source",
    ]
    latex_df.loc[:, "C"] = latex_df["C"].fillna(0)
    latex_df.loc[:, ["Dataset ID", "Task ID", "N", "d", "C"]] = (
        latex_df[["Dataset ID", "Task ID", "N", "d", "C"]].astype(int).astype(str)
    )
    latex_df.loc[:, r"\% cat"] = latex_df[r"\% cat"].round(2).astype(str)
    latex_df.loc[:, "Name"] = latex_df["Name"].str.replace("_", r"\_")
    latex_df.loc[:, "Domain"] = latex_df["Domain"].str.replace("&", r"\&")

    latex_df.loc[:, "TabPFNv2"] = latex_df["TabPFNv2"].replace(
        {True: r"\yessymb", False: r"\nosymb"}
    )
    latex_df.loc[:, "TabICL"] = latex_df["TabICL"].replace(
        {True: r"\yessymb", False: r"\nosymb"}
    )
    latex_df["Subset"] = latex_df["TabPFNv2"] + " | " + latex_df["TabICL"]

    latex_df["Task ID"] = latex_df["Task ID"].apply(
        lambda x: r"\href{https://www.openml.org/t/" + str(x) + "}{" + str(x) + "}"
    )
    latex_df["Dataset ID"] = latex_df["Dataset ID"].apply(
        lambda x: r"\href{https://www.openml.org/d/" + str(x) + "}{" + str(x) + "}"
    )
    latex_df["Dataset (Task) ID"] = (
        latex_df["Dataset ID"] + " (" + latex_df["Task ID"] + ")"
    )

    latex_df["N"] = latex_df["N"].astype("int")
    latex_df = latex_df.sort_values(by=["N"], ascending=[True])
    latex_df = latex_df.reset_index(drop=True)
    latex_df["N"] = latex_df["N"].astype(str)

    latex_df["Name"] = latex_df["Name"].replace(
        {
            r"HR\_Analytics\_Job\_Change\_of\_Data\_Scientists": r"\makecell[l]{HR\_Analytics\_Job\_Change\_ \\ of\_Data\_Scientists}"
        }
    )
    latex_df["Name"] = latex_df[["Name", "Source"]].apply(
        lambda row: r"\href{" + str(row[1]) + "}{" + str(row[0]) + "}", axis=1
    )

    # Replace 0 with -
    latex_df["C"] = latex_df["C"].replace("0", "-")

    latex_df = latex_df[
        ["Dataset (Task) ID", "Name", "Ref.", "N", "d", "C", r"\% cat", "Subset"]
    ]
    latex_df.to_latex("metadata/tabarena_dataset_metadata.tex", index=False)
