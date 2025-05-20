from __future__ import annotations

import numpy as np
import pandas as pd


def two_sample_test_for_data(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    *,
    test_repeats: int = 5,
    use_case_text: str = "default",
    num_cpus: int = 8,
    verbose: bool = True,
skip_feature_importance_shift_detection: bool = False,
):
    """Code to test for distribution shift between two datasets."""
    import tempfile
    from pathlib import Path

    from autogluon.core.utils import generate_train_test_split_combined
    from autogluon.tabular import TabularPredictor
    from scipy.stats import mannwhitneyu

    print(f"\n\n==== Testing if {use_case_text} have a distribution shift")

    # Heuristics
    score_threshold = 0.55
    feature_importance_threshold = 0.05

    train_data = data_a.copy()
    test_data = data_b.copy()

    label_2_sample = "__dist__"
    train_data[label_2_sample] = "train"
    test_data[label_2_sample] = "test"

    data_2_sample = pd.concat([train_data, test_data], ignore_index=True)
    data_2_sample = data_2_sample.sample(frac=1, random_state=0).reset_index(drop=True)

    feature_shifts_per_repeat = []
    threshold_shifts_per_repeat = []
    mann_whitney_shifts_per_repeat = []
    for i in range(test_repeats):
        if verbose:
            print("repeat: ", i)
        train_data_2_sample, test_data_2_sample = generate_train_test_split_combined(
            data_2_sample,
            label=label_2_sample,
            problem_type="binary",
            test_size=0.5,
            random_state=42 + i,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir).rmdir()
            predictor = TabularPredictor(
                label=label_2_sample,
                eval_metric="roc_auc",
                sample_weight="balance_weight",
                path=temp_dir,
            )

            predictor: TabularPredictor = predictor.fit(
                train_data_2_sample,
                hyperparameters={
                    "GBM": {},
                },
                verbosity=0,
                fit_weighted_ensemble=False,
                num_gpus=0,
                num_cpus=num_cpus,
            )

            predictor.leaderboard(test_data_2_sample, display=verbose)
            test_score = predictor.evaluate(
                data=test_data_2_sample, auxiliary_metrics=False
            )[predictor.eval_metric.name]
            if not skip_feature_importance_shift_detection:
                feature_importance = predictor.feature_importance(
                    test_data_2_sample, silent=True
                )
            proba = predictor.predict_proba(test_data_2_sample)

        dist_shift = test_score >= score_threshold

        if not skip_feature_importance_shift_detection:
            feature_importance_significant = feature_importance[
                feature_importance["importance"] >= feature_importance_threshold
            ]
            feature_importance_significant = list(feature_importance_significant.index)
        else:
            feature_importance_significant = []

        proba_class_a = proba[test_data_2_sample[label_2_sample] == "train"]["train"]
        proba_class_b = proba[test_data_2_sample[label_2_sample] == "test"]["train"]
        res = mannwhitneyu(x=proba_class_b, y=proba_class_a)
        alpha = 0.05
        dist_shift_test = res.pvalue <= alpha

        if verbose:
            print(f"\tDistribution Shift Detected based on Threshold: {dist_shift}")
            print(f"\tDistribution Shifted Features: {feature_importance_significant}")
            print(
                "\tDistribution Shift Detected based on Mann Whitney U Test: ",
                dist_shift_test,
            )

        feature_shifts_per_repeat += feature_importance_significant
        threshold_shifts_per_repeat.append(dist_shift)
        mann_whitney_shifts_per_repeat.append(dist_shift_test)

    print(f"\n*****Overall for: {use_case_text}")
    print(
        f"\tDistribution Shift Detected based on Threshold: {np.mean(threshold_shifts_per_repeat)}"
    )
    print(f"\tDistribution Shifted Features: {list(set(feature_shifts_per_repeat))}")
    print(
        "\tDistribution Shift Detected based on Mann Whitney U Test: ",
        np.mean(mann_whitney_shifts_per_repeat),
    )


def run_all_checks(
    *,
    data: pd.DataFrame,
    classification: bool,
    target_feature: str,
    n_shift_repeats: int = 5,
    custom_data_split_for_shift: None | int = None,
skip_feature_importance_shift_detection: bool = False,
) -> None:
    """Run common data checks on a given dataset to spot anomalies."""
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        None,
    ):
        # Initial check
        print("\n#### First Two Rows")
        print(data.head(2))
        print("\n#### Pandas Describe per Feature")
        res = data.describe(include="all").T
        res["UNIQUE"] = data.nunique()
        print(res)

        # Check class count
        if classification:
            print("\n#### Target Feature Distribution:")
            print(data[target_feature].value_counts())

    # Check Duplicates
    train_data_dups = sum(data.duplicated()) / len(data)
    train_data_dups_wo_labels = sum(
        data.drop(columns=[target_feature]).duplicated()
    ) / len(data)
    duplicate_report = f"""\n#### Duplicate Report:
    {"Duplicates:":<40} {train_data_dups:.4f}
    {"Duplicates without target feature:":<40} {train_data_dups_wo_labels:.4f}
    """
    print(duplicate_report)

    # Distribution Shift Check (assumes original order of data)
    data_split = len(data) // 2 if custom_data_split_for_shift is None else custom_data_split_for_shift
    two_sample_test_for_data(
        data.iloc[:data_split],
        data.iloc[data_split:],
        test_repeats=n_shift_repeats,
        use_case_text="dataset halves based on index",
        skip_feature_importance_shift_detection=skip_feature_importance_shift_detection,
    )

    # Distribution Shift Check (assumes original order of data)
    two_sample_test_for_data(
        data.drop(columns=[target_feature]).iloc[:data_split],
        data.drop(columns=[target_feature]).iloc[data_split:],
        test_repeats=n_shift_repeats,
        use_case_text="dataset halves based on index without target features",
        skip_feature_importance_shift_detection=skip_feature_importance_shift_detection,
    )
