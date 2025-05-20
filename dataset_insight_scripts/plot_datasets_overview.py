from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from tueplots import bundles

df = pd.read_csv("../dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv")

df["license_grouped"] = df["licence"].replace(
    {
        "CC0: Public Domain": "Public",
        "Public Domain": "Public",
        "ODC Public Domain Dedication and Licence (PDDL)": "Public",
        "CC BY-NC-SA": "Other",
        "CC BY-SA": "Other",
        "Database Contents License (DbCL) v1.0": "Other",
        "MIT License": "Other",
    }
)
df["problem_type"] = df["problem_type"].replace(
    {
        "binary": "Binary",
        "multiclass": "Multiclass",
        "regression": "Regression",
    }
)

# Set the visual style
sns.set_context("paper")
sns.set_style("whitegrid")


with plt.rc_context(bundles.neurips2024(rel_width=0.5, nrows=2)) as c:
    ### Distribution Plot
    problem_type_dist = df["problem_type"].value_counts()
    problem_type_dist = problem_type_dist.reset_index().rename(
        columns={"problem_type": "value"}
    )
    problem_type_dist["type"] = "Task"

    license_type_dist = df["license_grouped"].value_counts()
    license_type_dist = license_type_dist.reset_index().rename(
        columns={"license_grouped": "value"}
    )
    license_type_dist["type"] = "License"

    data_source_dist = df["data_source"].value_counts()
    data_source_dist = data_source_dist.reset_index().rename(
        columns={"data_source": "value"}
    )
    data_source_dist["type"] = "Source"

    dataset_age = 2025 - df["year"]
    mask_5 = dataset_age <= 5
    mask_5_10 = (dataset_age > 5) & (dataset_age <= 15)
    mask_15 = dataset_age > 15
    dataset_age[dataset_age <= 5] = "0-5 Years"
    dataset_age[mask_5_10] = "6-15 Years"
    dataset_age[mask_15] = "16+ Years"
    age_dist = dataset_age.value_counts()
    age_dist = age_dist.reset_index().rename(columns={"year": "value"})
    age_dist["type"] = "Age"

    plot_df = pd.concat(
        [problem_type_dist, license_type_dist, data_source_dist, age_dist], axis=0
    )

    def custom_barplot(data, **kwargs):
        current_type = data["type"].iloc[0]

        # Set a custom order only for a specific facet
        if current_type == "Age":
            order = [
                "0-5 Years",
                "6-15 Years",
                "16+ Years",
            ]  # Replace with your specific order
            sns.barplot(data=data, order=order, **kwargs)
        else:
            sns.barplot(data=data, **kwargs)

    g = sns.FacetGrid(plot_df, col="type", sharex=False, col_wrap=2, height=1.8)
    g.map_dataframe(
        custom_barplot, x="value", y="count", color=(0.4, 0.7, 1.0), alpha=0.9
    )
    g.set_axis_labels("", r"Number of Datasets")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig("dataset_types_overview.pdf")
    plt.show(dpi=400)

with plt.rc_context(bundles.neurips2024(rel_width=0.8)):
    ### Size/Scatter Plot
    plot_df = df.copy()

    cmap = sns.color_palette("flare", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    ax = sns.scatterplot(
        x="num_instances",
        y="num_features",
        hue="percentage_cat_features",
        data=plot_df,
        legend=False,
        hue_norm=sm.norm,
        palette=cmap,
        s=100,
    )
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r"Percentage (\%) of Categorical Features")
    plt.xlabel("Number of Samples")
    plt.ylabel("Number of Features")

    # Apply log scaling
    ax.set_xscale("log")
    ax.set_yscale("log")
    subs = [1, 5, 15]
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.savefig("dataset_size_overview.pdf")
    plt.show(dpi=400)
