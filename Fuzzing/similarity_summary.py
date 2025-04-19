import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Path to the directory containing summary CSVs
summary_dir = "similarities"

# Collect all summary CSV files
summary_files = glob(os.path.join(summary_dir, "folder_*_summary.csv"))

if not summary_files:
    print(f"No summary CSV files found in '{summary_dir}' directory.")
else:
    # Read and concatenate into a DataFrame
    df_list = [pd.read_csv(f) for f in summary_files]
    df_summary = pd.concat(df_list, ignore_index=True)

    # Convert folder to numeric and sort
    df_summary['folder'] = df_summary['folder'].astype(int)
    df_summary.sort_values('folder', inplace=True)

    # Compute overall statistics
    overall_mean = df_summary['mean_similarity'].mean()
    overall_median = df_summary['mean_similarity'].median()

    # Print overall mean and median
    print(f"Overall Mean of Mean Similarities: {overall_mean:.4f}")
    print(f"Overall Median of Mean Similarities: {overall_median:.4f}")

    # Plot mean similarity per folder
    plt.figure(figsize=(12, 6))
    plt.plot(df_summary['folder'], df_summary['mean_similarity'], marker='o', linestyle='-')
    plt.title('Mean Semantic Similarity per Folder')
    plt.xlabel('Folder')
    plt.ylabel('Mean Similarity')
    plt.grid(True)

    # Annotate overall mean and median
    plt.axhline(overall_mean, linestyle='--', label=f'Overall Mean: {overall_mean:.4f}')
    plt.axhline(overall_median, linestyle=':', label=f'Overall Median: {overall_median:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()