# U.S. State Economic Purchasing Power & Tax Data Pipeline

A Python data pipeline and analytical case study that combines state income, regional price levels, tax rules, inflation, and macroeconomic data to estimate **hypothetical after-tax purchasing power at state median household income**.

> **Methodological note:** median household income is not the same concept as a tax filing unit. State tax estimates are approximations based on normalized statutory brackets and standard deductions. The project therefore treats the output as a comparable analytical scenario rather than observed household tax liability.

## What this repository demonstrates

- Multi-source data acquisition from FRED and public tax sources
- Web scraping and workbook normalization
- Reusable federal, payroll, and state tax calculation logic
- State-year panel construction and feature engineering
- Cross-state Regional Price Parity adjustment
- National inflation adjustment to constant dollars
- Counterfactual linear-trend analysis
- Static Seaborn visualizations suitable for GitHub
- Explicit validation and methodological limitations

## Repository structure

```text
.
├── README.md
├── analysis.ipynb
├── requirements.txt
├── .env.example
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── figures/
└── src/
    ├── data_collection.py
    ├── tax_engine.py
    ├── transformations.py
    └── visualization.py
```

## Architecture

```text
Public APIs / web / Excel sources
              ↓
       data_collection.py
              ↓
       raw source files
              ↓
       transformations.py
              ↓
       processed tax data
              ↓
          tax_engine.py
              ↓
state-year purchasing-power panel
              ↓
        visualization.py
              ↓
      analysis.ipynb + PNGs
```

## Setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set `FRED_API_KEY` in your environment. `.env.example` documents the expected variable name.
4. Open and run `analysis.ipynb` from the repository root.

The notebook saves its two final analytical panels to `data/processed/` and its visualizations to `figures/`.

## Important limitations

- State tax liability is approximate and does not fully model every state-specific credit, exemption, local tax, or special rule contained in source notes.
- Median household income is used as a standardized scenario input; it is not equivalent to taxable income for a representative filing unit.
- RPP adjusts geographic price-level differences within a year; the project uses PCEPI separately for comparisons across years.
- Pre-COVID linear trend projections are descriptive counterfactuals, not causal estimates.
