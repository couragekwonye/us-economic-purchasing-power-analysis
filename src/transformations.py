\
"""Data cleaning, feature engineering, projections, and panel construction."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from sklearn.linear_model import LinearRegression

from .data_collection import safe_get_series
from .tax_engine import TaxEngine


STATE_MAPPING = {
    "Ala.": "AL", "Alaska": "AK", "Ariz.": "AZ", "Ark.": "AR", "Calif.": "CA",
    "Colo.": "CO", "Conn.": "CT", "Del.": "DE", "Fla.": "FL", "Ga.": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Ill.": "IL", "Ind.": "IN", "Iowa": "IA",
    "Kans.": "KS", "Ky.": "KY", "La.": "LA", "Maine": "ME", "Md.": "MD",
    "Mass.": "MA", "Mich.": "MI", "Minn.": "MN", "Miss.": "MS", "Mo.": "MO",
    "Mont.": "MT", "Nebr.": "NE", "Nev.": "NV", "N.H.": "NH", "N.J.": "NJ",
    "N.M.": "NM", "N.Y.": "NY", "N.C.": "NC", "N.D.": "ND", "Ohio": "OH",
    "Okla.": "OK", "Ore.": "OR", "Pa.": "PA", "R.I.": "RI", "S.C.": "SC",
    "S.D.": "SD", "Tenn.": "TN", "Tex.": "TX", "Utah": "UT", "Vt.": "VT",
    "Va.": "VA", "Wash.": "WA", "W.Va.": "WV", "Wis.": "WI", "Wyo.": "WY",
    "D.C.": "DC",
}


def prep_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a Date-indexable frame to annual last observations."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out.set_index("Date").resample("YE").last()


def linear_projection(
    data: pd.DataFrame,
    y_var: str,
    period_start: str,
    period_end: str,
    data_type: str = "float",
    projection_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fit a linear trend to the final 36 observations in a training window and extrapolate.

    This is a descriptive counterfactual benchmark, not a causal model.
    """
    if "Date" not in data.columns:
        raise KeyError("data must contain a Date column")
    if y_var not in data.columns:
        raise KeyError(f"data must contain {y_var!r}")

    train = data.query("Date >= @period_start and Date <= @period_end").tail(36).copy()
    if len(train) < 2:
        raise ValueError("Projection window has too few observations.")

    train["Period"] = np.arange(len(train))
    model = LinearRegression().fit(train[["Period"]], train[y_var])

    if projection_end is None:
        projection_end = pd.Timestamp(date.today().replace(day=1))
    projection_end = pd.Timestamp(projection_end).to_period("M").to_timestamp()
    last_date = pd.Timestamp(train["Date"].iloc[-1]).to_period("M").to_timestamp()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), projection_end, freq="MS")

    if len(future_dates) == 0:
        projected = pd.DataFrame(columns=["Date", f"Projected {y_var}"])
    else:
        future_period = np.arange(len(train), len(train) + len(future_dates))
        predictions = model.predict(future_period.reshape(-1, 1))
        if data_type == "int":
            predictions = np.rint(predictions).astype(int)
        else:
            predictions = np.round(predictions.astype(float), 2)
        projected = pd.DataFrame({"Date": future_dates, f"Projected {y_var}": predictions})

    return projected.merge(data, on="Date", how="outer").sort_values("Date").reset_index(drop=True)


def clean_state_tax_workbook(
    source_path: str | Path,
    cleaned_path: str | Path,
    notes_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Clean the multi-year state-tax workbook and optionally export footnotes."""
    source_path = Path(source_path)
    cleaned_path = Path(cleaned_path)
    notes_dir = Path(notes_dir) if notes_dir else cleaned_path.parent / "state_tax_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    wb = load_workbook(source_path)
    footnote_pattern = re.compile(r"^\([a-zA-Z]{1,2}\)")

    for year_sheet in wb.sheetnames:
        sheet = wb[year_sheet]
        sheet.delete_rows(idx=1)

        split_row = None
        last_val = "start"
        for i, row in enumerate(sheet.iter_rows(min_col=1, max_col=1, values_only=True), start=1):
            current_val = row[0]
            if current_val and footnote_pattern.match(str(current_val)) and last_val is None:
                split_row = i
                break
            last_val = current_val
        if not split_row:
            continue

        notes_wb = Workbook()
        notes_sheet = notes_wb.active
        for row in sheet.iter_rows(min_row=split_row, values_only=True):
            notes_sheet.append(row)
        notes_wb.save(notes_dir / f"{year_sheet}_state_income_tax_notes.xlsx")

        sheet.delete_rows(idx=split_row - 1, amount=(sheet.max_row - split_row + 2))
        columns_to_delete = [12, 9, 7, 5, 3] if year_sheet in {"2017", "2018"} else [6, 3]
        for col in columns_to_delete:
            sheet.delete_cols(col)

        current_state = None
        rows_to_delete: list[int] = []
        for row_idx in range(2, sheet.max_row + 1):
            state_cell = sheet.cell(row=row_idx, column=1)
            if state_cell.value:
                val = str(state_cell.value).strip()
                is_orphan_note = val.startswith("(") or (val.endswith(")") and val[0].islower())
                if is_orphan_note:
                    sheet.cell(row=row_idx, column=11, value=state_cell.value)
                    state_cell.value = None

            other_cells = [sheet.cell(row=row_idx, column=c).value for c in range(2, 11)]
            is_row_empty = all(v is None or v == "" for v in other_cells)
            if state_cell.value in (None, ""):
                if is_row_empty:
                    rows_to_delete.append(row_idx)
                else:
                    state_cell.value = current_state
            else:
                val = str(state_cell.value)
                if "(" in val:
                    left, right = val.split("(", 1)
                    state_cell.value = left.strip()
                    sheet.cell(row=row_idx, column=11, value="(" + right)
                current_state = state_cell.value

        for row_idx in sorted(rows_to_delete, reverse=True):
            sheet.delete_rows(row_idx)

        sheet.insert_cols(1)
        sheet.cell(row=1, column=1, value="Year")
        for row_idx in range(2, sheet.max_row + 1):
            sheet.cell(row=row_idx, column=1).value = year_sheet

        headers = [
            "Year", "State", "Single Filer Rates", "Single Filer Brackets",
            "Married Filer Jointly Rate", "Married Filer Brackets",
            "Single Standard Deduction", "Couple Standard Deduction",
            "Personal Exemption: Single", "Personal Exemption: Couple",
            "Personal Exemption: Dependent", "Notes",
        ]
        for col_idx, header in enumerate(headers, start=1):
            sheet.cell(row=1, column=col_idx, value=header)

    wb.save(cleaned_path)
    return pd.read_excel(cleaned_path, sheet_name=None)


def build_state_tax_master(cleaned_sheets: dict[str, pd.DataFrame], output_csv: str | Path) -> pd.DataFrame:
    """Normalize cleaned state-tax sheets into one long analytical table."""
    pieces: list[pd.DataFrame] = []
    for _, df in cleaned_sheets.items():
        single = df[["Year", "State", "Single Filer Rates", "Single Filer Brackets", "Single Standard Deduction"]].copy()
        single["Filing_Status"] = "single"
        single = single.rename(columns={
            "Single Filer Rates": "Rate",
            "Single Filer Brackets": "Bracket",
            "Single Standard Deduction": "Standard_Deduction",
        })

        married = df[["Year", "State", "Married Filer Jointly Rate", "Married Filer Brackets", "Couple Standard Deduction"]].copy()
        married["Filing_Status"] = "married_joint"
        married = married.rename(columns={
            "Married Filer Jointly Rate": "Rate",
            "Married Filer Brackets": "Bracket",
            "Couple Standard Deduction": "Standard_Deduction",
        })
        pieces.extend([single, married])

    master = pd.concat(pieces, ignore_index=True)
    for col in ["State", "Filing_Status"]:
        master[col] = master[col].astype(str).str.strip()
    master["State"] = master["State"].map(STATE_MAPPING)
    master["Year"] = pd.to_numeric(master["Year"], errors="raise").astype(int)
    master["Bracket"] = pd.to_numeric(master["Bracket"], errors="coerce").fillna(0)
    master["Standard_Deduction"] = master["Standard_Deduction"].replace("n.a.", "0")
    master["Standard_Deduction"] = pd.to_numeric(master["Standard_Deduction"], errors="coerce")
    master["Standard_Deduction"] = master.groupby(["State", "Filing_Status"])["Standard_Deduction"].ffill().fillna(0)

    rate_strings = master["Rate"].fillna("0").replace("none", "0").astype(str)
    extracted = pd.to_numeric(rate_strings.str.extract(r"(\d+\.?\d*)")[0], errors="coerce").fillna(0)
    is_capital = rate_strings.str.contains("capital", case=False, na=False)
    is_interest = rate_strings.str.contains("interest", case=False, na=False)
    master["Capital_Gains_Rate"] = np.where(is_capital, extracted, 0.0)
    master["Interest_Dividend_Rate"] = np.where(is_interest, extracted, 0.0)
    master["Rate"] = pd.to_numeric(np.where(is_capital | is_interest, "0", master["Rate"]), errors="coerce").fillna(0)

    master = master.dropna(subset=["State"]).sort_values(["Year", "State", "Filing_Status", "Bracket"]).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_csv, index=False)
    return master


def annual_pce_price_index(fred) -> pd.DataFrame:
    """Return annual-average PCE price index for cross-year deflation."""
    pcepi = safe_get_series(fred, "PCEPI").rename("PCEPI").to_frame().reset_index().rename(columns={"index": "Date"})
    pcepi["Date"] = pd.to_datetime(pcepi["Date"])
    pcepi["Year"] = pcepi["Date"].dt.year
    return pcepi.groupby("Year", as_index=False)["PCEPI"].mean()


def build_purchasing_power_panel(
    fred,
    fred_codes: list[str],
    states: list[str],
    mhi_codes: list[str],
    tax_engine: TaxEngine,
    target_filing_status: str = "married_joint",
    dependent_count: int = 0,
    constant_dollar_year: int = 2024,
) -> pd.DataFrame:
    """Build the state-year after-tax purchasing-power analytical panel."""
    rpp_list: list[pd.DataFrame] = []
    mhi_list: list[pd.DataFrame] = []

    for code, state in zip(fred_codes, states):
        frame = safe_get_series(fred, code).to_frame(name="RPP").rename_axis("Date")
        frame["State"] = state
        frame["Year"] = frame.index.year
        rpp_list.append(frame)

    for code, state in zip(mhi_codes, states):
        frame = safe_get_series(fred, code).to_frame(name="MHI").rename_axis("Date")
        frame["State"] = state
        frame["Year"] = frame.index.year
        mhi_list.append(frame)

    df = (
        pd.concat(rpp_list).reset_index()
        .merge(pd.concat(mhi_list).reset_index(), on=["State", "Year", "Date"], how="inner")
    )
    df = df[df["Year"].isin(tax_engine.supported_years)].copy()
    if df.empty:
        raise ValueError("No overlapping years across RPP, MHI, federal, payroll, and state-tax sources.")

    breakdowns = df.apply(
        lambda row: tax_engine.calculate_net_income(
            income=row["MHI"],
            year=int(row["Year"]),
            filing_status=target_filing_status,
            state=row["State"],
            dependent_count=dependent_count,
            return_breakdown=True,
        ),
        axis=1,
    ).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), breakdowns.reset_index(drop=True)], axis=1)

    df["rpp_adjusted_after_tax_income_current_dollars"] = df["net_income_approx"] * (100.0 / df["RPP"])
    pcepi = annual_pce_price_index(fred)
    df = df.merge(pcepi, on="Year", how="left")
    if constant_dollar_year not in set(pcepi["Year"]):
        raise KeyError(f"PCEPI does not contain requested constant-dollar year {constant_dollar_year}.")
    base = float(pcepi.loc[pcepi["Year"] == constant_dollar_year, "PCEPI"].iloc[0])
    real_col = f"real_after_tax_purchasing_power_{constant_dollar_year}_dollars"
    df[real_col] = df["rpp_adjusted_after_tax_income_current_dollars"] * (base / df["PCEPI"])

    df["tax_scenario"] = target_filing_status
    df["dependent_count_assumption"] = dependent_count
    df["state_tax_method"] = "approximate statutory brackets + standard deduction; special rules omitted"

    validate_panel(df, tax_engine, target_filing_status, real_col)
    return df.sort_values(["State", "Year"]).reset_index(drop=True)


def validate_panel(df: pd.DataFrame, tax_engine: TaxEngine, filing_status: str, real_col: str) -> None:
    """Fail fast on structural or tax-rule coverage problems."""
    required = [
        "MHI", "RPP", "federal_income_tax", "payroll_tax", "state_income_tax_approx",
        "net_income_approx", real_col,
    ]
    if df[required].isna().any().any():
        missing = df.loc[df[required].isna().any(axis=1), ["State", "Year"] + required]
        raise ValueError(f"Pipeline produced missing required values. Sample:\n{missing.head()}")
    if not (df["net_income_approx"] <= df["MHI"]).all():
        raise AssertionError("At least one row has net income greater than gross median household income.")
    if not (df["RPP"] > 0).all():
        raise AssertionError("RPP must be positive.")
    if df.duplicated(["State", "Year"]).any():
        raise AssertionError("Duplicate State-Year rows detected.")

    missing_rules = [
        (state, int(year))
        for state, year in df[["State", "Year"]].itertuples(index=False, name=None)
        if (state, int(year), filing_status) not in tax_engine.state_tax_lookup
    ]
    if missing_rules:
        raise AssertionError(f"Missing {filing_status} state-tax rules: {missing_rules[:10]}")
