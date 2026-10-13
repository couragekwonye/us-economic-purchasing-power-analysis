\
"""Federal, payroll, and approximate state income-tax calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


FILING_STATUSES = ["single", "married_joint", "married_separate", "head"]

PERSONAL_EXEMPTION_PHASEOUT = {
    2015: {"single": 258250, "married_joint": 309900, "married_separate": 154950, "head": 284050},
    2016: {"single": 259400, "married_joint": 311300, "married_separate": 155650, "head": 285350},
    2017: {"single": 261500, "married_joint": 313800, "married_separate": 156900, "head": 287650},
}

ADDITIONAL_MEDICARE_THRESHOLDS = {
    "single": 200_000,
    "head": 200_000,
    "married_joint": 250_000,
    "married_separate": 125_000,
}


@dataclass
class TaxEngine:
    """Validated lookup-based tax calculator used by the analytical pipeline.

    State income tax is intentionally labeled an approximation. The normalized
    state source models statutory brackets and standard deductions but does not
    fully encode every state-specific credit, exemption, local tax, or special rule.
    """

    federal_brackets_df: pd.DataFrame
    federal_deductions: list[dict]
    payroll_df: pd.DataFrame
    state_tax_df: pd.DataFrame
    payroll_wage_limits: dict[int, float] = field(init=False)
    fed_tax_lookup: dict = field(init=False)
    state_tax_lookup: dict = field(init=False)
    state_tax_min_year: int = field(init=False)
    state_tax_max_year: int = field(init=False)

    def __post_init__(self) -> None:
        self.federal_brackets_df = self.federal_brackets_df.copy()
        self.state_tax_df = self.state_tax_df.copy()
        self.payroll_df = self.payroll_df.copy()

        self.state_tax_df.columns = self.state_tax_df.columns.str.strip()
        for col in ["Standard_Deduction", "Bracket", "Rate"]:
            self.state_tax_df[col] = pd.to_numeric(self.state_tax_df[col], errors="coerce").fillna(0)
        self.state_tax_df["Year"] = pd.to_numeric(self.state_tax_df["Year"], errors="raise").astype(int)

        payroll = self.payroll_df.copy()
        payroll["year"] = pd.to_numeric(payroll["year"], errors="raise").astype(int)
        payroll["wage_cap"] = pd.to_numeric(payroll["wage_cap"], errors="raise")
        self.payroll_wage_limits = (
            payroll.drop_duplicates("year", keep="first")
            .set_index("year")["wage_cap"]
            .astype(float)
            .to_dict()
        )

        self.fed_tax_lookup = {}
        for year, group in self.federal_brackets_df.groupby("year"):
            year = int(year)
            for status in FILING_STATUSES:
                if status not in group.columns:
                    continue
                status_df = group[["rate", status]].copy().rename(columns={status: "Bracket"})
                status_df["Bracket"] = pd.to_numeric(status_df["Bracket"], errors="coerce")
                status_df["rate"] = pd.to_numeric(status_df["rate"], errors="coerce")
                status_df = status_df.dropna(subset=["Bracket", "rate"]).sort_values("Bracket")
                self.fed_tax_lookup[(year, status)] = status_df.to_dict("records")

        self.state_tax_lookup = {}
        grouped = self.state_tax_df.groupby(["State", "Year", "Filing_Status"], observed=True)
        for (state, year, status), group in grouped:
            group = group.sort_values("Bracket").copy()
            group["Ceiling"] = group["Bracket"].shift(-1).fillna(float("inf"))
            self.state_tax_lookup[(str(state), int(year), str(status))] = {
                "standard_deduction": float(group["Standard_Deduction"].iloc[0]),
                "brackets": group[["Rate", "Ceiling"]]
                    .rename(columns={"Ceiling": "Bracket"})
                    .to_dict("records"),
            }

        self.state_tax_min_year = int(self.state_tax_df["Year"].min())
        self.state_tax_max_year = int(self.state_tax_df["Year"].max())

    @property
    def supported_years(self) -> set[int]:
        deduction_years = {int(row["year"]) for row in self.federal_deductions}
        bracket_years = {int(y) for y in self.federal_brackets_df["year"].dropna().unique()}
        state_years = set(range(self.state_tax_min_year, self.state_tax_max_year + 1))
        return deduction_years & bracket_years & set(self.payroll_wage_limits) & state_years

    def _deduction_record(self, year: int) -> dict:
        record = next((row for row in self.federal_deductions if int(row["year"]) == int(year)), None)
        if record is None:
            raise KeyError(f"No federal deduction data available for tax year {year}.")
        return record

    def standard_deduction(self, year: int, filing_status: str, is_senior: bool = False) -> float:
        """Return the standard deduction without substituting another tax year."""
        record = self._deduction_record(year)
        if filing_status not in FILING_STATUSES:
            raise ValueError(f"Unsupported filing_status: {filing_status}")
        if filing_status not in record or pd.isna(record[filing_status]):
            raise KeyError(f"No standard deduction for {year}, {filing_status}.")

        deduction = float(record[filing_status])
        if is_senior:
            senior_key = "married_senior" if filing_status == "married_joint" else "unmarried_senior"
            deduction += float(record.get(senior_key, 0) or 0)
        return deduction

    def federal_personal_exemption(
        self,
        income: float,
        year: int,
        filing_status: str,
        dependent_count: int = 0,
    ) -> float:
        """Approximate personal exemptions for 2015-2017; suspended from 2018 onward."""
        year = int(year)
        if year >= 2018:
            return 0.0

        record = self._deduction_record(year)
        base = float(record.get("personal", 0) or 0)
        taxpayer_count = 2 if filing_status == "married_joint" else 1
        exemption = base * (taxpayer_count + dependent_count)

        threshold = PERSONAL_EXEMPTION_PHASEOUT.get(year, {}).get(filing_status)
        if threshold is not None and income > threshold and exemption > 0:
            step = 1_250 if filing_status == "married_separate" else 2_500
            reduction_pct = min(1.0, 0.02 * math.ceil((income - threshold) / step))
            exemption *= 1 - reduction_pct
        return exemption

    def calculate_income_tax(
        self,
        income: float,
        year: int,
        filing_status: str,
        is_senior: bool = False,
        dependent_count: int = 0,
    ) -> dict:
        if income < 0:
            raise ValueError("income must be non-negative")
        if filing_status not in FILING_STATUSES:
            raise ValueError(f"Unsupported filing_status: {filing_status}")

        deduction = self.standard_deduction(year, filing_status, is_senior)
        exemption = self.federal_personal_exemption(income, year, filing_status, dependent_count)
        taxable_income = max(0.0, float(income) - deduction - exemption)

        brackets = self.fed_tax_lookup.get((int(year), filing_status))
        if not brackets:
            raise KeyError(f"No federal brackets for {year}, {filing_status}.")

        tax_owed = 0.0
        previous_threshold = 0.0
        for row in brackets:
            ceiling = float(row["Bracket"])
            rate = float(row["rate"])
            amount_in_bracket = min(taxable_income, ceiling) - previous_threshold
            if amount_in_bracket > 0:
                tax_owed += amount_in_bracket * rate
            if taxable_income <= ceiling:
                break
            previous_threshold = ceiling

        return {
            "tax": tax_owed,
            "taxable_income": taxable_income,
            "standard_deduction": deduction,
            "personal_exemption": exemption,
            "effective_tax_rate": tax_owed / income if income > 0 else 0.0,
        }

    def calculate_payroll_tax(self, gross_income: float, year: int, filing_status: str) -> dict:
        if gross_income < 0:
            raise ValueError("gross_income must be non-negative")
        if filing_status not in ADDITIONAL_MEDICARE_THRESHOLDS:
            raise ValueError(f"Unsupported filing_status: {filing_status}")
        if int(year) not in self.payroll_wage_limits:
            raise KeyError(f"No Social Security wage-base data for {year}.")

        wage_base_limit = float(self.payroll_wage_limits[int(year)])
        ss_tax = min(float(gross_income), wage_base_limit) * 0.062
        medicare_tax = float(gross_income) * 0.0145
        additional_medicare = 0.0
        if int(year) >= 2013:
            threshold = ADDITIONAL_MEDICARE_THRESHOLDS[filing_status]
            additional_medicare = max(0.0, float(gross_income) - threshold) * 0.009

        total_tax = ss_tax + medicare_tax + additional_medicare
        return {
            "tax": total_tax,
            "social_security_tax": ss_tax,
            "medicare_tax": medicare_tax,
            "additional_medicare_tax": additional_medicare,
            "effective_tax_rate": total_tax / gross_income if gross_income > 0 else 0.0,
        }

    def calculate_state_income_tax(self, income: float, year: int, filing_status: str, state: str) -> dict:
        """Approximate state tax from normalized brackets + standard deduction."""
        rules = self.state_tax_lookup.get((str(state), int(year), filing_status))
        if rules is None:
            raise KeyError(
                f"No state-tax rules for state={state}, year={year}, filing_status={filing_status}. "
                "Missing rules are never interpreted as a zero-tax state."
            )

        taxable_income = max(0.0, float(income) - float(rules["standard_deduction"]))
        tax_owed = 0.0
        previous_ceiling = 0.0
        for row in rules["brackets"]:
            ceiling = float(row["Bracket"])
            rate = float(row["Rate"])
            amount_in_bracket = min(taxable_income, ceiling) - previous_ceiling
            if amount_in_bracket > 0:
                tax_owed += amount_in_bracket * rate
            if taxable_income <= ceiling:
                break
            previous_ceiling = ceiling

        return {
            "tax": tax_owed,
            "taxable_income": taxable_income,
            "standard_deduction": float(rules["standard_deduction"]),
            "effective_tax_rate": tax_owed / income if income > 0 else 0.0,
            "method": "statutory_brackets_plus_standard_deduction_approximation",
        }

    def calculate_net_income(
        self,
        income: float,
        year: int,
        filing_status: str,
        state: str,
        is_senior: bool = False,
        dependent_count: int = 0,
        return_breakdown: bool = False,
    ):
        federal = self.calculate_income_tax(income, year, filing_status, is_senior, dependent_count)
        payroll = self.calculate_payroll_tax(income, year, filing_status)
        state_tax = self.calculate_state_income_tax(income, year, filing_status, state)

        total_tax = federal["tax"] + payroll["tax"] + state_tax["tax"]
        net_income = float(income) - total_tax
        if not return_breakdown:
            return net_income

        return {
            "gross_income": float(income),
            "federal_income_tax": federal["tax"],
            "payroll_tax": payroll["tax"],
            "state_income_tax_approx": state_tax["tax"],
            "total_tax_approx": total_tax,
            "net_income_approx": net_income,
            "effective_total_tax_rate_approx": total_tax / income if income > 0 else 0.0,
        }
