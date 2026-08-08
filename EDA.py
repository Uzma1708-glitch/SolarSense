import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

# Folder to save charts
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def save_chart(filename):
    """Save chart and display it."""
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, filename))
    plt.show()        # Display graph
    plt.close()


def perform_eda():

    # Load dataset
    data = load_data()

    print("=" * 80)
    print("SOLARSENSE - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    print("\nFirst 5 Rows")
    print(data.head())

    print("\nDataset Shape")
    print(data.shape)

    print("\nColumns")
    print(data.columns.tolist())

    print("\nData Types")
    print(data.dtypes)

    print("\nMissing Values")
    print(data.isnull().sum())

    print("\nStatistical Summary")
    print(data.describe())

    numeric_cols = [
        "DC_POWER",
        "AC_POWER",
        "DAILY_YIELD",
        "TOTAL_YIELD"
    ]

    # ==================================================
    # 1. Histograms
    # ==================================================
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_chart(f"{col.lower()}_histogram.png")

    # ==================================================
    # 2. Boxplots
    # ==================================================
    for col in numeric_cols:
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=data[col])
        plt.title(f"{col} Boxplot")
        save_chart(f"{col.lower()}_boxplot.png")

    # ==================================================
    # 3. Correlation Heatmap
    # ==================================================
    plt.figure(figsize=(8, 6))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_chart("correlation_heatmap.png")

    # ==================================================
    # 4. Scatter Plot
    # ==================================================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="DC_POWER",
        y="AC_POWER",
        data=data
    )
    plt.title("DC Power vs AC Power")
    save_chart("dc_vs_ac_scatter.png")

    # ==================================================
    # 5. Line Plot
    # ==================================================
    data["DATE_TIME"] = pd.to_datetime(
        data["DATE_TIME"],
        dayfirst=True
    )

    line_data = data.sort_values("DATE_TIME").head(500)

    plt.figure(figsize=(12, 5))
    plt.plot(
        line_data["DATE_TIME"],
        line_data["AC_POWER"],
        color="green"
    )
    plt.title("AC Power Over Time")
    plt.xlabel("Date Time")
    plt.ylabel("AC Power")
    plt.xticks(rotation=45)
    save_chart("ac_power_line.png")

    print("\nEDA Completed Successfully!")
    print("Charts are displayed and saved in:")
    print(CHARTS_DIR)


if __name__ == "__main__":
    perform_eda()