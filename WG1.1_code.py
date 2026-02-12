# Opdracht: initial data visualisation + statistiek per dataset
# Bestandsnaam: datasets.csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def main():
    # -----------------------------
    # 1) Read CSV as DataFrame
    # -----------------------------
    df = pd.read_csv(r"C:\Users\sblok\Downloads\datasets.csv")

    # Basic sanity checks
    required_cols = {"dataset", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found columns: {list(df.columns)}")

    # Ensure numeric, als er iets niet naar een getal kan maak er NaN van
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["dataset", "x", "y"]) #rijen weggooien waar NaN in voorkomt

    print("\n--- Head of dataframe ---")
    print(df.head()) #checken of het goed gaat

    # -----------------------------
    # 2) Number + names of datasets
    # -----------------------------
    dataset_names = sorted(df["dataset"].unique()) #datasets sorteren en ze een naam geven
    print("\n--- Datasets ---")
    print(f"Number of datasets: {len(dataset_names)}")
    print("Dataset names:", dataset_names)

    # -----------------------------
    # 3) Statistics per dataset
    #    count, mean, variance, std dev
    # -----------------------------
    print("\n--- Statistics per dataset (x and y) ---")

    grouped = df.groupby("dataset") # elke dataset met een naam groeperen

    stats_table = grouped[["x", "y"]].agg(["count", "mean", "var", "std"]) # voor elke dataset de statistiek doen
    # Make it prettier to read:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    print(stats_table)

    # -----------------------------
    # 4) Correlation + Covariance matrix per dataset
    # -----------------------------
    print("\n--- Correlation (x,y) per dataset ---")
    corr_per_ds = {}
    for name, g in grouped: # voor elke dataset correlatie bepalen
        corr = g["x"].corr(g["y"])
        corr_per_ds[name] = corr
        print(f"{name}: correlation = {corr:.5f}")

    print("\n--- Covariance matrix per dataset ---")
    for name, g in grouped: #voor elke dataset covariance bepalen
        cov = g[["x", "y"]].cov() #covariance matrix maken
        print(f"\n{name} covariance matrix:")
        print(cov)

    # -----------------------------
    # 5) Linear regression per dataset (scipy.stats.linregress)
    #    slope, intercept, r-value
    # -----------------------------
    print("\n--- Linear regression per dataset (y = slope*x + intercept) ---")
    reg_results = {}
    for name, g in grouped: #voor elke dataset lineaire regressie doen
        lr = stats.linregress(g["x"], g["y"])
        reg_results[name] = lr
        print(
            f"{name}: slope={lr.slope:.5f}, intercept={lr.intercept:.5f}, r-value={lr.rvalue:.5f}, p-value={lr.pvalue:.5g}"
        )

    # -----------------------------
    # 6) Observations (dingen geprint om er wat over te zeggen)
    # -----------------------------
    print("\n--- Quick observations (auto-generated hints) ---")
    # Kijk of gemiddelden/varianties opvallend gelijk zijn
    means_x = grouped["x"].mean()
    means_y = grouped["y"].mean()
    vars_x = grouped["x"].var()
    vars_y = grouped["y"].var()

    # Spreiding van gemiddelden/varianties (klein = sterk gelijkend)
    print(f"Spread of mean(x): {means_x.max() - means_x.min():.5f}")
    print(f"Spread of mean(y): {means_y.max() - means_y.min():.5f}")
    print(f"Spread of var(x):  {vars_x.max() - vars_x.min():.5f}")
    print(f"Spread of var(y):  {vars_y.max() - vars_y.min():.5f}")

    # Correlaties op een rij
    corr_series = pd.Series(corr_per_ds).sort_values()
    print("\nCorrelations sorted:")
    print(corr_series)

    print(
        "\nTip: als gemiddelden/varianties erg op elkaar lijken maar de scatterplots er totaal anders uitzien, "
        "noem dat expliciet in je observaties (klassiek voorbeeld: datasets met vergelijkbare samenvattende statistiek "
        "maar andere patronen/outliers/non-lineariteit)."
    )

    # -----------------------------
    # 7) Violin plots: x per dataset, y per dataset
    # -----------------------------
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="dataset", y="x", inner="quartile", cut=0)
    plt.title("Violin plot of x per dataset")
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="dataset", y="y", inner="quartile", cut=0)
    plt.title("Violin plot of y per dataset")
    plt.tight_layout()

    # -----------------------------
    # 8) Scatterplots for all datasets (FacetGrid + map_dataframe)
    # -----------------------------
    g = sns.FacetGrid(df, col="dataset", col_wrap=4, height=3, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="x", y="y")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle("Scatterplots per dataset", y=1.02)
    plt.tight_layout()

    # -----------------------------
    # 9) Scatterplots + regression line for all datasets (lmplot)
    # -----------------------------
    # lmplot maakt zelf een figure; daarom geen plt.figure() hier.
    sns.lmplot(
    data=df,
    x="x",
    y="y",
    col="dataset",
    col_wrap=4,
    height=3,
    ci=None,
    facet_kws={"sharex": False, "sharey": False},
)
    plt.suptitle("Scatterplots with regression line per dataset", y=1.02)
    plt.tight_layout()

    # -----------------------------
    # Show all figures
    # -----------------------------
    plt.show()


if __name__ == "__main__":
    main()