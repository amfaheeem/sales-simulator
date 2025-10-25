### Overview

This project models the **financial and ownership evolution** of a new pizza restaurant in Vancouver’s Lower Mainland using a **Python simulation** built around JSON-configurable parameters.
It integrates:

* monthly **sales and expense simulation**,
* **ROI-based diminishing partnership logic**, and
* **interactive dashboards** (Plotly) for visualization.

The model helps evaluate business viability, test multiple growth hypotheses, and project **founder vs. investor returns** over time.

---

## 1. Core Simulation Logic

### 🔹 Simulation Engine

Implemented in `simulate_pizza_business.py`.
It:

1. Reads inputs from `config.json`.
2. Generates **monthly** revenue, cost, and cash flow.
3. Aggregates results into **annual financials**.
4. Computes **business ROI** and applies **diminishing partnership equity transfers**.

Outputs:

* `results_monthly_<scenario>.csv`
* `results_annual_<scenario>.csv`
* `ownership_<scenario>.csv`
* `dashboard_<scenario>.html` (interactive dashboard)

---

## 2. Key Input Structure (`config.json`)

### 🏗 Project Setup

```json
"project": {
  "initial_investment": 180000,
  "base_month": "May",
  "currency": "CAD",
  "base_month_revenue": 50000,
  "year_over_year_sales_growth": 0.10,
  "lead_months_before_open": 6,
  "rent_free_months": 4,
  "founder_equity_initial": 0.25,
  "investor_equity_initial": 0.75
}
```

* **Initial investment** defines ROI base.
* **Lead months** = setup time before operation; rent during non-free months is added to capital expenses.

---

## 3. Sales Model

Each sales scenario defines 12 monthly multipliers (relative to base-month revenue).

```json
"sales_models": {
  "Scenario_1_Optimistic": [1.0, 1.1, 1.25, 1.4, 1.3, 1.2, 1.1, 1.3, 1.5, 1.6, 1.4, 1.2],
  "Scenario_2_Pessimistic": [1.0, 0.9, 0.85, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.95, 1.0, 1.05]
}
```

**Monthly revenue** = `base_month_revenue × multiplier × (1 + yoy_growth)^year`.

---

## 4. Expense Structure

### 🏢 Fixed & Semi-Fixed Costs

```json
"expenses": {
  "rent": {"base": 6000, "annual_increase_rate": 0.05},
  "equipment_lease": {"monthly": 800},
  "insurance": {"monthly": 500},
  "accounting": {"monthly": 300},
  "marketing": {"initial_cost": 5000, "monthly": 500},
  "utilities": {
    "tiers": [
      {"sales_min": 0, "sales_max": 40000, "cost": 1600},
      {"sales_min": 40000, "sales_max": 80000, "cost": 1800},
      {"sales_min": 80000, "sales_max": 999999, "cost": 2000}
    ]
  },
  "capital_expenses": {"renovation": 20000, "equipment": 15000, "licenses": 5000}
}
```

Rent increases yearly, utilities and labour scale with sales.

---

## 5. Labour Model

```json
"labour": {
  "roles": {
    "entry": {"hour_rate": 18, "min_count": 1},
    "experienced": {"hour_rate": 30, "min_count": 1},
    "manager": {"hour_rate": 40, "min_count": 1}
  },
  "scaling_rules": {
    "hours_per_day": 12,
    "days_per_month": 30,
    "sales_per_additional_worker": 30000
  }
}
```

Labour cost = (base staff × hours × rate) + (extra entry-level workers based on sales volume).

---

## 6. Menu Composition & Margins

```json
"menu": {
  "sales_mix": {"traditional": 0.66, "seafood": 0.34},
  "traditional": {
    "cheese": {"price": 20, "cost": 6},
    "chicken": {"price": 24, "cost": 8},
    "beef": {"price": 26, "cost": 10}
  },
  "seafood": {
    "shrimp": {"price": 40, "cost": 15},
    "lobster": {"price": 75, "cost": 30},
    "salmon": {"price": 50, "cost": 20}
  }
}
```

COGS is derived dynamically:

* Weighted margin = (Traditional margin × 0.66) + (Seafood margin × 0.34).

---

## 7. ROI and Ownership Model

### ROI Definition

```
ROI = Net Profit / Total Initial Investment
```

### Founder’s Diminishing Partnership

* Founder begins with **25% equity** (10% capital + 15% promote).
* Investors start with **75%**.
* Founder can buy more equity annually based on ROI.

### Unlock Formula

Let:

* `ROI_min = 27%`, `ROI_max = 108%`
* `unlock_step_1 = 15%`, `unlock_step_2 = 7.5%`

Then:

```
unlock_step_1 = 0 if ROI ≤ ROI_min
unlock_step_1 = 15% × (ROI - ROI_min) / (ROI_max - ROI_min) if ROI in range
unlock_step_1 = 15% if ROI ≥ ROI_max

unlock_step_2 = 7.5% × (ROI - ROI_max) / (ROI_max - ROI_min) if ROI > ROI_max
```

The total equity unlocked = step1 + step2 (capped by remaining investor equity).

---

## 8. Outputs

### CSVs

* **Monthly results**: Revenue, expenses, gross margin, net profit, cumulative ROI.
* **Annual results**: Aggregated profit/loss and annual ROI.
* **Ownership results**: Yearly equity transfer and resulting founder/investor shares.

### Interactive Dashboard (`dashboard_<scenario>.html`)

Includes:

* Revenue vs. COGS vs. OPEX over time.
* Net profit and cumulative ROI.
* Annual ownership transitions.
* Multi-year P&L summary.

---

## 9. Assumptions

* Taxes are ignored (pre-tax ROI).
* Founder buyouts happen at **original equity value**, not market value.
* Marketing initial cost and pre-open rent are treated as **capital**.
* All values in CAD.

---

## 10. Next Development Steps

* Implement **reverse solving** (given target ROI → required sales).
* Add **Monte Carlo scenarios** for stochastic demand.
* Integrate with **Google Sheets** API for live parameter input.
* Add **IRR and cash-on-cash return** for investor view.
* Optional: extend visualization via **Streamlit dashboard**.

---