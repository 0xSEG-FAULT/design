import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from constants import *
from sklearn.decomposition import PCA
from scipy.stats import zscore


def identify_outliers(df, columns, fixed_thresholds=False):
    if fixed_thresholds:
        outlier_counts = pd.DataFrame(index=columns)
        outlier_min_values = pd.DataFrame(index=columns)
        for col in columns:
            i = 0
            for threshold in fixed_thresholds[col]:
                i += 1
                outlier_mask = df[col] > threshold
                outlier_counts.loc[col, "Count_" + str(i)] = outlier_mask.sum()
                outlier_min_values.loc[col, "Threshold_" + str(i)] = threshold
    else:
        z_thresholds = list(range(3, 11))
        outlier_counts = pd.DataFrame(index=columns, columns=z_thresholds)
        outlier_min_values = pd.DataFrame(index=columns, columns=z_thresholds)
        for col in columns:
            z_scores = np.abs(zscore(df[col], nan_policy="omit"))
            mean = df[col].mean()
            std_dev = df[col].std()
            for outlier_z_threshold in z_thresholds:
                outlier_mask = z_scores > outlier_z_threshold
                outlier_counts.loc[col, outlier_z_threshold] = outlier_mask.sum()
                outlier_min_values.loc[col, outlier_z_threshold] = (
                    mean + std_dev * outlier_z_threshold
                ).round(1)
    return outlier_counts, outlier_min_values


def generate_data_plots(features, output_dir):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=features, fill=True, color="#4C72B0", linewidth=1.5)
    plt.title("Feature Distributions", fontsize=16, weight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png")
    plt.close()

    corr = features.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm_r")
    plt.savefig(f"{output_dir}/feature_correlation.png")
    plt.close()

    for col in features.columns:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(
            data=features[col], fill=True, alpha=0.7, color="#4C72B0", linewidth=2
        )
        plt.title(f"Distribution of {col}", fontsize=16, weight="bold")
        plt.xlabel(col, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.grid(alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distribution_{col}.png")
        plt.close()

    print(f"Plots saved to: {output_dir}")


def env_data_exploration(features, output_dir, target=None, target_columns=[]):
    if target:
        features[target.name] = target

    print("Data statistics:")
    stats = features.describe().T
    stats["missing_values"] = features.isnull().sum()
    stats["skewness"] = features.skew()
    stats["kurtosis"] = features.kurtosis()
    print(stats)

    corr = features.corr()

    outlier_counts, outlier_min_values = identify_outliers(features, target_columns)
    print("\nOutliers counts:")
    print(outlier_counts)
    print("\nOutliers min values:")
    print(outlier_min_values)
    outlier_info = (
        outlier_counts.astype(str) + " (>" + outlier_min_values.astype(str) + ")"
    )

    # fixed_outlier_counts, fixed_thresholds = identify_outliers(features, target_columns,
    #     fixed_thresholds =  {COL_GLUCOSE:[500,600,800,1000],
    #                         COL_LAC:[600,700,800],
    #                         COL_ETH:[1200,1500,2000],
    #                         COL_CAF:[170,200,230],
    #                         COL_ACT:[100]})
    # print("\nFixed Outliers counts:")
    # print(fixed_outlier_counts)
    # print("\nFixed Outliers values:")
    # print(fixed_thresholds)
    # fixed_outlier_info = fixed_outlier_counts.astype(str) + " (>" + fixed_thresholds.astype(str) + ")"

    # null_temp_count = features[COL_TEMP].isnull().sum()
    # print(f"Samples with missing temperature: {null_temp_count}")
    zero_counts = (
        features[target_columns] == 0
    ).sum(axis=1)
    counts = zero_counts.value_counts().sort_index()
    mixture_counts = {
        "5 substances": counts.get(0, 0),
        "4 substances": counts.get(1, 0),
        "3 substances": counts.get(2, 0),
        "2 substances": counts.get(3, 0),
        "1 substance": counts.get(4, 0),
    }
    mixture_df = pd.DataFrame.from_dict(
        mixture_counts, orient="index", columns=["Sample Count"]
    )

    if PLOTTING_ENABLED:
        generate_data_plots(features, output_dir)

    if PLOTTING_ENABLED:
        generate_data_plots(features, output_dir)

    with pd.ExcelWriter(f"{output_dir}/data_exploration.xlsx") as writer:
        stats.to_excel(writer, sheet_name="Data Statistics", index=True)
        mixture_df.to_excel(writer, sheet_name="Mixture Counts", index=True)
        corr.to_excel(writer, sheet_name="Data Correlation", index=True)
        outlier_counts.to_excel(writer, sheet_name="Outlier Counts", index=True)
        outlier_min_values.to_excel(writer, sheet_name="Outlier Min Values", index=True)
        outlier_info.to_excel(writer, sheet_name="Outlier Summary", index=True)
        # fixed_outlier_counts.to_excel(writer, sheet_name='Fixed Outlier Counts', index=True)
        # fixed_thresholds.to_excel(writer, sheet_name='Fixed Outlier Values', index=True)
        # fixed_outlier_info.to_excel(writer, sheet_name='Fixed Outlier Summary', index=True)
    print(f"Data saved to {output_dir}/data_exploration.xlsx")


def spectral_data_analysis(
    features, output_dir, feature_use_fraction=1, peak_regions=None
):
    if not PLOTTING_ENABLED:
        return
    features_fraction = features.iloc[:, :: int(np.round(1 / feature_use_fraction))]

    plt.figure(figsize=(12, 8))
    sns.heatmap(features_fraction, cmap="viridis", xticklabels=100)
    plt.xlabel("Wavelength", fontsize=14)
    plt.ylabel("Sample Index", fontsize=14)
    plt.savefig(f"{output_dir}/spectra_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    for i in range(min(len(features_fraction), 50)):  # plot up to 50 spectra
        plt.plot(
            features_fraction.columns.astype(float),
            features_fraction.iloc[i],
            alpha=0.3,
        )
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Absorption", fontsize=14)
    plt.title("Spectral Data", fontsize=16, weight="bold")
    plt.savefig(f"{output_dir}/spectra_plot.png")
    plt.close()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_fraction)
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.title("PCA of Spectral Data", fontsize=16, weight="bold")
    plt.savefig(f"{output_dir}/spectra_pca_2.png")
    plt.close()

    if peak_regions:
        for i in range(len(peak_regions)):
            col_index_start = features.columns.get_loc(peak_regions[i][0])
            col_index_end = features.columns.get_loc(peak_regions[i][1] - 1)
            peak_features = features.iloc[:, col_index_start:col_index_end]
            plt.figure(figsize=(12, 8))
            sns.boxplot(peak_features, showfliers=True, showmeans=True, width=0.75)
            xticks = range(0, len(peak_features.columns), 20)
            plt.xticks(ticks=xticks, labels=[peak_features.columns[i] for i in xticks])
            plt.title("Feature Distribution", fontsize=16, weight="bold")
            plt.xlabel("Features", fontsize=14)
            plt.ylabel("Values", fontsize=14)
            plt.savefig(f"{output_dir}/spectra_peak_box_{i}.png")
            plt.close()

    print(f"Spectra plots saved to: {output_dir}")
