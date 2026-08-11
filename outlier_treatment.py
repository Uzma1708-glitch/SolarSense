"""
outlier_treatment.py
--------------------------------------------------------------------
Handles outlier detection and treatment using statistical methods (IQR).
Calculates outlier statistics for all numeric features and supports
interactive treatments:
  - Keep (no action)
  - Remove (drop rows containing outliers)
  - Cap/Winsorize (clip values to IQR bounds)
Saves updated visualizations to static/charts.
--------------------------------------------------------------------
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

INK, MUTED, GRID = "#1e293b", "#64748b", "#cbd5e1"
AMBER, TEAL, ROSE = "#f59e0b", "#10b981", "#f43f5e"


def get_iqr_params(df: pd.DataFrame, col: str, k: float = 1.5) -> dict:
    """
    Calculate IQR parameters for a specific column.
    For power features (DC_POWER, AC_POWER), we restrict the calculation
    to daylight hours (06:00-18:00) to prevent night-time zeros from biasing bounds.
    """
    series = df[col]

    # Check if we should use daylight filtering
    is_power_col = col in ["DC_POWER", "AC_POWER"]
    if is_power_col and "HOUR" in df.columns:
        daylight = (df["HOUR"] >= 6) & (df["HOUR"] <= 18)
        base_series = series[daylight]
    else:
        base_series = series

    q1 = float(base_series.quantile(0.25))
    q3 = float(base_series.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    # Count outliers
    if is_power_col and "HOUR" in df.columns:
        daylight = (df["HOUR"] >= 6) & (df["HOUR"] <= 18)
        # Outliers can only occur during daylight; night zeros are normal
        outliers = (df[daylight][col] < lo) | (df[daylight][col] > hi)
        n_outliers = int(outliers.sum())
    else:
        outliers = (series < lo) | (series > hi)
        n_outliers = int(outliers.sum())

    total_rows = len(df)
    pct_outliers = round((n_outliers / total_rows) * 100, 2)

    return {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lo, 2),
        "upper_bound": round(hi, 2),
        "count": n_outliers,
        "percentage": pct_outliers
    }


def calculate_all_outliers(df: pd.DataFrame) -> dict:
    """Calculate IQR details for all numerical columns."""
    cols = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
    report = {}
    for col in cols:
        if col in df.columns:
            report[col] = get_iqr_params(df, col)
    return report


def apply_outlier_treatment(df: pd.DataFrame, col: str, treatment: str = "keep") -> tuple:
    """
    Apply selected outlier treatment (keep, remove, cap) on a single column.
    Returns:
        - Treated DataFrame
        - Count of rows affected
    """
    df = df.copy()
    if treatment == "keep" or col not in df.columns:
        return df, 0

    params = get_iqr_params(df, col)
    lo = params["lower_bound"]
    hi = params["upper_bound"]

    # Identify outliers
    is_power_col = col in ["DC_POWER", "AC_POWER"]
    if is_power_col and "HOUR" in df.columns:
        daylight = (df["HOUR"] >= 6) & (df["HOUR"] <= 18)
        outlier_mask = daylight & ((df[col] < lo) | (df[col] > hi))
    else:
        outlier_mask = (df[col] < lo) | (df[col] > hi)

    n_affected = int(outlier_mask.sum())

    if treatment == "remove":
        # Remove outlier rows
        df = df[~outlier_mask]
    elif treatment == "cap":
        # Cap outlier values
        df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(lower=lo, upper=hi)

    return df, n_affected


def _style_ax(ax, fig):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.title.set_color("#1e293b")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.6, alpha=0.7)


def generate_outlier_plots(df: pd.DataFrame):
    """Generate and save boxplots and scatter plots showing anomalies."""
    daylight = df[(df["HOUR"] >= 6) & (df["HOUR"] <= 18)] if "HOUR" in df.columns else df

    # 1. Boxplots for power columns
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, col, color in zip(axes, ["DC_POWER", "AC_POWER"], [AMBER, TEAL]):
        bp = ax.boxplot(daylight[col], vert=True, patch_artist=True, widths=0.5,
                        flierprops=dict(marker="o", markerfacecolor=ROSE, markersize=3, alpha=0.5, markeredgewidth=0))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.35)
        bp["medians"][0].set_color("#0f172a")
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="semibold")
        ax.set_xticks([])
        _style_ax(ax, fig)

    fig.suptitle("Spread Analysis (Outliers shown in Rose)", color="#1e293b", y=1.03, fontsize=12, fontweight="bold")
    path = os.path.join(CHARTS_DIR, "outlier_boxplots.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. Scatter plot with flagged anomalies
    # Flag outliers based on IQR for demonstration plot
    params_dc = get_iqr_params(df, "DC_POWER")
    params_ac = get_iqr_params(df, "AC_POWER")

    dc_lo, dc_hi = params_dc["lower_bound"], params_dc["upper_bound"]
    ac_lo, ac_hi = params_ac["lower_bound"], params_ac["upper_bound"]

    daylight_mask = (df["HOUR"] >= 6) & (df["HOUR"] <= 18) if "HOUR" in df.columns else pd.Series(True, index=df.index)

    is_anomaly = daylight_mask & (
            (df["DC_POWER"] < dc_lo) | (df["DC_POWER"] > dc_hi) |
            (df["AC_POWER"] < ac_lo) | (df["AC_POWER"] > ac_hi)
    )

    # Midday zero power check (another domain anomaly)
    midday_zero = daylight_mask & (df["HOUR"] >= 10) & (df["HOUR"] <= 15) & (df["DC_POWER"] == 0)
    is_anomaly = is_anomaly | midday_zero

    sample = df[df["DC_POWER"] > 0]
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 4))
    is_anomaly_sample = is_anomaly.loc[sample.index]
    normal = sample[~is_anomaly_sample]
    anomaly = sample[is_anomaly_sample]

    ax.scatter(normal["DC_POWER"], normal["AC_POWER"], s=6, color=TEAL, alpha=0.35, linewidths=0,
               label="Normal Operations")
    ax.scatter(anomaly["DC_POWER"], anomaly["AC_POWER"], s=16, color=ROSE, alpha=0.8, linewidths=0,
               label="Flagged Anomalies")

    ax.set_xlabel("DC Power (kW)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Flagged Outliers & Outages on Conversion Curve", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
    _style_ax(ax, fig)
    ax.grid(True, color="#e2e8f0", linewidth=0.6, alpha=0.7)

    path = os.path.join(CHARTS_DIR, "anomaly_scatter.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from load_data import load_clean

    df = load_clean()
    r = calculate_all_outliers(df)
    print("Outlier Report:")
    for k, v in r.items():
        print(f"Column: {k} -> {v['count']} outliers ({v['percentage']}%)")
    generate_outlier_plots(df)
