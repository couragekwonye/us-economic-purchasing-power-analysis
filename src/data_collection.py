\
"""External data acquisition helpers.

Network-bound work lives here so the analysis notebook can remain focused on
questions, results, and interpretation.
"""

from __future__ import annotations

import os
import time
from io import BytesIO, StringIO
from typing import Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fredapi import Fred


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_fred_client(api_key: str | None = None) -> Fred:
    """Return an authenticated FRED client.

    The API key is read from ``FRED_API_KEY`` when it is not passed directly.
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. Copy .env.example to .env or export the "
            "environment variable before running the live-data pipeline."
        )
    return Fred(api_key=key)


def safe_get_series(fred: Fred, code: str, retries: int = 3, delay: int = 20) -> pd.Series:
    """Retrieve a FRED series with bounded retries."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fred.get_series(code)
        except Exception as exc:  # network/API errors vary by backend
            last_error = exc
            if attempt < retries:
                print(f"FRED request failed for {code}; retrying in {delay}s ({attempt}/{retries})...")
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def fred_frame(
    fred: Fred,
    code: str,
    value_name: str,
    observation_end: str | None = None,
) -> pd.DataFrame:
    """Fetch a FRED series and return a two-column Date/value DataFrame."""
    kwargs = {"observation_end": observation_end} if observation_end else {}
    series = fred.get_series(code, **kwargs)
    return series.rename(value_name).to_frame().reset_index().rename(columns={"index": "Date"})


def validate_fred_codes(fred: Fred, fred_codes: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return valid and invalid FRED series identifiers."""
    valid: list[str] = []
    invalid: list[str] = []
    for code in fred_codes:
        try:
            series = fred.get_series(code)
            (valid if series is not None and len(series) else invalid).append(code)
        except Exception:
            invalid.append(code)
    return valid, invalid


def scrape_federal_brackets(format: str = "df"):
    """Scrape historical federal income-tax brackets from Tax Foundation."""
    url = "https://taxfoundation.org/data/all/federal/historical-income-tax-rates-brackets/"
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    raw = tables[0].copy()
    raw = raw.drop(raw.columns[[2, 5, 8, 11]], axis=1)
    raw.columns = [
        "Year",
        "Married Filing Jointly (Rates/Brackets)",
        "Married Filing Jointly Bracket",
        "Married Filing Separately (Rates/Brackets)",
        "Married Filing Separately Bracket",
        "Single Filer (Rates/Brackets)",
        "Single Filer Bracket",
        "Head of Household (Rates/Brackets)",
        "Head of Household Bracket",
        "Notes:",
    ]
    raw = raw.drop(raw.columns[[3, 5, 7, -1]], axis=1)
    raw.columns = ["year", "rate", "married_joint", "married_separate", "single", "head"]
    raw["year"] = pd.to_numeric(raw["year"].astype(str).str.strip().replace("1940(A)", "1940"), errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["year", "rate"])

    long = pd.melt(
        raw,
        id_vars=["year", "rate"],
        value_vars=["married_joint", "married_separate", "single", "head"],
        var_name="Filing_Status",
        value_name="Bracket",
    )
    long["Bracket"] = long["Bracket"].astype(str).str.replace(r"[\$,]", "", regex=True)
    long["Income_Threshold"] = pd.to_numeric(long["Bracket"], errors="coerce")
    long["Income_Threshold"] = (
        long.groupby(["year", "Filing_Status"])["Income_Threshold"]
        .shift(-1)
        .fillna(float("inf"))
    )
    long = long.drop_duplicates(subset=["year", "rate", "Filing_Status"])

    fed_df = long.pivot(
        index=["year", "rate"],
        columns="Filing_Status",
        values="Income_Threshold",
    ).reset_index()
    fed_df.columns.name = None
    fed_df["rate"] = (
        fed_df["rate"].astype(str).str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("No income tax", "0", regex=False)
    )
    fed_df["rate"] = pd.to_numeric(fed_df["rate"], errors="coerce") / 100
    fed_df = fed_df.dropna(subset=["rate"])

    if format == "df":
        return fed_df
    return fed_df.to_dict(orient="records")


def historical_standard_deductions() -> list[dict]:
    """Scrape federal standard deductions and personal-exemption amounts."""
    status_mapping = {
        "Single": "single",
        "Married Filing Jointly": "married_joint",
        "Married Filing Separately": "married_separate",
        "Head of Household": "head",
        "Personal Exemption": "personal",
        "Additional Amount for Unmarried Seniors": "unmarried_senior",
        "Additional Amount for Married Seniors": "married_senior",
    }

    years = list(scrape_federal_brackets("df").query("year >= 2015")["year"].dropna().astype(int).unique())
    all_deductions: list[pd.DataFrame] = []

    for year in years:
        url = f"https://taxfoundation.org/data/all/federal/{year}-tax-brackets/"
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"Skipping {year}: server returned {response.status_code}")
            continue

        try:
            tables = pd.read_html(StringIO(response.text))
            table_index = 3 if year == 2017 else 1
            frame = tables[table_index].iloc[:, 0:2].copy()
            frame["year"] = year
            all_deductions.append(frame)
        except (ValueError, IndexError):
            print(f"Could not find the deduction table for {year}.")

    if not all_deductions:
        raise RuntimeError("No federal deduction tables could be collected.")

    excluded = {
        "Source: Author’s Calculations.",
        "Source: IRS.",
        "Source: Internal Revenue Service",
        "Source: Internal Revenue Source",
        "Source: Author’s calculations.",
    }
    df = pd.concat(all_deductions, ignore_index=True).dropna()
    df = df[~df["Filing Status"].isin(excluded)]
    df["Deduction Amount"] = pd.to_numeric(
        df["Deduction Amount"].astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce",
    )
    df["Filing Status"] = df["Filing Status"].map(status_mapping)
    df = df.dropna(subset=["Filing Status", "Deduction Amount"])
    wide = (
        df.pivot(index="year", columns="Filing Status", values="Deduction Amount")
        .fillna(0)
        .reset_index()
    )
    wide.columns.name = None
    return wide.to_dict(orient="records")


def ss_wage_limits() -> pd.DataFrame:
    """Scrape Social Security taxable wage-base history from SSA.

    Selenium is used because the source page may require a rendered browser.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://www.ssa.gov/oact/cola/cbb.html")
        time.sleep(5)
        tables = pd.read_html(StringIO(driver.page_source))
    finally:
        driver.quit()

    df = tables[1].iloc[:, 0:2].copy()
    df.columns = ["year", "wage_cap"]
    df = df.iloc[2:-1].reset_index(drop=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("Year", "", regex=False).str.replace("Amount", "", regex=False)
    df["wage_cap"] = pd.to_numeric(df["wage_cap"].str.replace(r"[\$,]", "", regex=True), errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna().astype({"year": int})

    historical = {year: 3000 for year in range(1937, 1951)}
    historical.update({year: 3600 for year in range(1951, 1955)})
    historical.update({year: 4200 for year in range(1955, 1959)})
    historical.update({year: 4800 for year in range(1959, 1966)})
    historical.update({year: 6600 for year in range(1966, 1968)})
    historical.update({year: 7800 for year in range(1968, 1972)})
    historical_df = (
        pd.DataFrame.from_dict(historical, orient="index", columns=["wage_cap"])
        .rename_axis("year")
        .reset_index()
    )

    return (
        pd.concat([df, historical_df], ignore_index=True)
        .drop_duplicates("year", keep="first")
        .sort_values("year")
        .reset_index(drop=True)
    )


def _extract_office_viewer_src(url: str) -> str | None:
    parsed = urlparse(url)
    if "view.officeapps.live.com" in parsed.netloc:
        src = parse_qs(parsed.query).get("src")
        if src:
            return unquote(src[0])
    return None


def _find_first_excel_link_from_html(base_url: str, html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.lower().endswith((".xlsx", ".xls")):
            return urljoin(base_url, href)
    return None


def load_state_income_tax(url: str) -> dict[str, pd.DataFrame]:
    """Load a state-income-tax workbook from a direct or HTML-wrapped URL."""
    office_src = _extract_office_viewer_src(url)
    if office_src:
        url = office_src

    try:
        head = requests.head(url, headers=BROWSER_HEADERS, allow_redirects=True, timeout=15)
    except requests.RequestException:
        head = None

    if head is not None:
        content_type = head.headers.get("Content-Type", "").lower()
        final_url = head.url or url
        if final_url.lower().endswith((".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
            response = requests.get(final_url, headers=BROWSER_HEADERS, timeout=30)
            response.raise_for_status()
            return pd.read_excel(BytesIO(response.content), sheet_name=None, engine="openpyxl")

    response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if response.content[:2] == b"PK" or "spreadsheet" in content_type or "excel" in content_type:
        return pd.read_excel(BytesIO(response.content), sheet_name=None, engine="openpyxl")

    excel_link = _find_first_excel_link_from_html(response.url, response.text)
    if excel_link:
        excel_response = requests.get(excel_link, headers=BROWSER_HEADERS, timeout=30)
        excel_response.raise_for_status()
        return pd.read_excel(BytesIO(excel_response.content), sheet_name=None, engine="openpyxl")

    raise ValueError("Could not find an Excel workbook at the provided URL.")


def save_workbook_sheets(dfs: dict[str, pd.DataFrame], path: str) -> None:
    """Write a mapping of sheet name -> DataFrame to an Excel workbook."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
