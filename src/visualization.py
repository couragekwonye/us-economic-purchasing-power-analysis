from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def _finish(fig, ax, figure_dir: str | Path, filename: str, title: str, xlabel: str, ylabel: str):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ax.set_title(title, loc="left", weight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = figure_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path


def plot_real_gdp_growth(annual_change: pd.Series, figure_dir="figures"):
    data = annual_change.dropna().rename("Real GDP growth (%)").reset_index()
    data.columns = ["Year", "Real GDP growth (%)"]
    data = data.query("Year >= 2008")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=data, x="Year", y="Real GDP growth (%)", marker="o", ax=ax)
    return _finish(fig, ax, figure_dir, "01_real_gdp_growth.png", "U.S. Real GDP Growth", "Year", "Q4-to-Q4 growth (%)")


def plot_private_employment(us_priv: pd.DataFrame, figure_dir="figures"):
    data = (
        us_priv[["Date", "Private Employees", "Projected Private Employees"]]
        .query('Date >= "2016-01-01"')
        .melt(id_vars="Date", value_vars=["Private Employees", "Projected Private Employees"], var_name="Series", value_name="Employees (thousands)")
        .dropna(subset=["Employees (thousands)"])
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x="Date", y="Employees (thousands)", hue="Series", ax=ax)
    return _finish(fig, ax, figure_dir, "02_private_employment_actual_vs_trend.png", "Private Payroll Employment vs. Extrapolated Pre-COVID Trend", "Date", "Employees (thousands)")


def plot_cpi_projection(cpi_projection: pd.DataFrame, figure_dir="figures"):
    data = (
        cpi_projection[["Date", "CPI", "Projected CPI"]]
        .query('Date >= "2017-01-01"')
        .melt(id_vars="Date", value_vars=["CPI", "Projected CPI"], var_name="Series", value_name="Index")
        .dropna(subset=["Index"])
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x="Date", y="Index", hue="Series", ax=ax)
    return _finish(fig, ax, figure_dir, "03_cpi_actual_vs_trend.png", "Consumer Price Index vs. Extrapolated Pre-COVID Trend", "Date", "CPI index")


def plot_top_purchasing_power(married_dataset: pd.DataFrame, constant_dollar_year=2024, figure_dir="figures"):
    col = f"real_after_tax_purchasing_power_{constant_dollar_year}_dollars"
    latest_year = int(married_dataset["Year"].max())
    latest = married_dataset.query("Year == @latest_year").copy()
    top = latest.nlargest(15, col).sort_values(col)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
    data=top,
    x=col,
    y="State",
    hue="State",
    palette="deep",
    legend=False,
    ax=ax
)
    path = _finish(fig, ax, figure_dir, "04_top_state_purchasing_power.png", f"Highest Estimated After-Tax Purchasing Power by State — {latest_year}", f"Real after-tax purchasing power ({constant_dollar_year} dollars)", "State")
    return path, latest


def plot_top_tax_burden(latest_married: pd.DataFrame, figure_dir="figures"):
    data = latest_married.copy()
    data["Effective total tax rate (%)"] = data["effective_total_tax_rate_approx"] * 100
    top = data.nlargest(15, "Effective total tax rate (%)").sort_values("Effective total tax rate (%)")
    latest_year = int(data["Year"].max())
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
    data=top,
        x="Effective total tax rate (%)",
        y="State",
        hue="State",
        palette="deep",
        legend=False,
        ax=ax
)
    return _finish(fig, ax, figure_dir, "05_highest_estimated_tax_burden.png", f"Highest Estimated Total Tax Burdens — {latest_year}", "Effective total tax rate (%)", "State")


def plot_purchasing_power_trend(married_dataset: pd.DataFrame, latest_married: pd.DataFrame, constant_dollar_year=2024, figure_dir="figures"):
    col = f"real_after_tax_purchasing_power_{constant_dollar_year}_dollars"
    leading_states = latest_married.nlargest(5, col)["State"].tolist()
    data = married_dataset[married_dataset["State"].isin(leading_states)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x="Year", y=col, hue="State", marker="o", ax=ax)
    return _finish(fig, ax, figure_dir, "06_purchasing_power_trend_top_states.png", "Real After-Tax Purchasing Power Over Time", "Year", f"Real after-tax purchasing power ({constant_dollar_year} dollars)")


def plot_income_vs_purchasing_power(latest_married: pd.DataFrame, constant_dollar_year=2024, figure_dir="figures"):
    col = f"real_after_tax_purchasing_power_{constant_dollar_year}_dollars"
    data = latest_married.dropna(subset=["MHI", col]).copy()
    latest_year = int(data["Year"].max())
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(data=data, x="MHI", y=col, ax=ax)
    sns.regplot(data=data, x="MHI", y=col, scatter=False, ci=95, ax=ax)
    return _finish(fig, ax, figure_dir, "07_income_vs_purchasing_power.png", f"Median Household Income vs. Estimated Purchasing Power — {latest_year}", "Median household income (current dollars)", f"Real after-tax purchasing power ({constant_dollar_year} dollars)")
