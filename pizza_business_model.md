# Pizza Business Financial Simulation Model

## 1. Overview

This document describes a self-contained Python simulation model designed to project the financial performance of a new pizza restaurant venture. The model reads all business parameters from a single `config.json` file, runs a monthly simulation over a multi-year period, and generates detailed financial reports and an interactive dashboard.

The core of the model is driven by a **daily sales projection**, which is then influenced by monthly seasonality and year-over-year growth assumptions. It calculates key financial metrics, including revenue, Cost of Goods Sold (COGS), Operating Expenses (OPEX), net profit, cash flow, and Return on Investment (ROI). It also includes a dynamic model for founder equity unlocks based on annual performance.

## 2. How to Run the Simulation

The simulation is executed from the command line. You must provide the path to the configuration file and the name of the sales scenario you wish to model.

```bash
python3 simulate_pizza_business.py --config config.json --scenario Scenario_3_Middle_Ground --outdir ./output
```

### Arguments:
- `--config`: (Required) Path to the JSON configuration file. Default: `config.json`.
- `--scenario`: (Required) The key of the desired sales model from the `sales_models` section of the config.
- `--years`: (Optional) The number of years to simulate. Default: `5`.
- `--outdir`: (Optional) The directory where output files will be saved. Default: `.` (the current directory).

## 3. Configuration (`config.json`)

All inputs for the simulation are defined in this JSON file.

### 3.1 `project`
Defines high-level project settings.

```json
"project": {
  "name": "OceanCrust Pizza",
  "currency": "CAD",
  "initial_investment": 180000,
  "base_month": "May",
  "year_over_year_sales_growth": 0.10,
  "daily_sales_projection": 60
}
```
- `daily_sales_projection`: **The most important input.** This is the projected number of pizzas sold per day for the base month of the first year.
- `year_over_year_sales_growth`: The annual growth rate applied to the `daily_sales_projection`.
- `base_month`: The first month of operations. This is used to align the monthly sales multipliers.
- `initial_investment`: A reference value for the initial investment (used in some ROI calculations).

### 3.2 `menu`
A flat list of all menu items. The model calculates revenue and COGS based on this list.

```json
"menu": {
  "items": [
    { "name": "Cheese", "cost": 5.00, "price": 19.00, "ratio": 0.198 },
    { "name": "Chicken", "cost": 9.00, "price": 28.00, "ratio": 0.20 }
  ]
}
```
- `items`: An array of pizza objects.
- `cost`: The cost to produce one unit.
- `price`: The sale price of one unit.
- `ratio`: The percentage of total daily sales that this item represents. **The sum of all ratios should equal 1.0.**

### 3.3 `sales_models`
Contains different scenarios for monthly sales seasonality.

```json
"sales_models": {
  "Scenario_3_Middle_Ground": [1.00, 1.30, 1.69, 2.11, 2.11, 2.11, 1.69, 1.69, 1.35, 1.35, 1.76, 1.76]
}
```
Each scenario is a list of 12 multipliers. For each month, the `daily_sales_projection` is multiplied by the corresponding value in the list to determine the actual daily sales for that month.

### 3.4 `expenses`
Defines all operational and capital expenses.

- **`rent`**: `base` monthly cost and `annual_increase_rate`.
- **`equipment_lease`**: Fixed `monthly` cost.
- **`utilities`**: A tiered system where the monthly `cost` is determined by which `sales_min`/`sales_max` bracket the total monthly revenue falls into.
- **`insurance`, `accounting`, `meals_and_entertainment`, `office_expense`**: Fixed monthly costs.
- **`marketing`**:
    - `initial_cost`: A one-time cost treated as a capital expense. It is added to the principal for ROI calculations but is **not** deducted from the first month's operational cash flow.
    - `monthly`: A fixed operational expense for every month.
- **`capital_expenses`**: A list of one-time startup costs that form the bulk of the initial investment principal.

### 3.5 `labour`
Defines staffing costs and work rules.

```json
"labour": {
  "roles": {
    "entry": { "hour_rate": 19, "min_count": 1 }
  },
  "scaling_rules": {
    "hours_per_day": 10,
    "days_per_month": 28
  }
}
```
- `roles`: Defines different job roles, their `hour_rate`, and the minimum number of staff (`min_count`) for that role.
- `scaling_rules`:
    - `hours_per_day`, `days_per_month`: Used to calculate total monthly work hours.

**Note:** The parameter `sales_per_additional_worker` is obsolete and no longer used by the simulation.

### 3.6 `investment_model`
Defines the rules for the annual founder equity unlock.

- `roi_range`: The `min` and `max` Annual ROI thresholds for the linear equity unlock.
- `unlock_step_1`: The maximum percentage of equity that can be unlocked in Step 1.
- `unlock_step_2`: A bonus equity percentage unlocked if the `unlock_step_2_threshold` is met.

## 4. Core Simulation Logic

### 4.1 Revenue and COGS Calculation
For each month in the simulation:
1.  **Determine Daily Sales**: The `daily_sales_projection` is adjusted by the month's seasonality multiplier and the compounded year-over-year growth rate.
    `current_daily_sales = daily_sales_projection * monthly_multiplier * yoy_growth_factor`
2.  **Calculate Daily Financials**: The model calculates the total revenue and COGS for a single day by distributing the `current_daily_sales` across the `menu` items according to their `ratio`.
3.  **Scale to Monthly**: The daily revenue and COGS are multiplied by `days_per_month` to get the final monthly figures.

### 4.2 Operating Expenses (OPEX)
- **Rent**: Increases annually based on the `annual_increase_rate`.
- **Utilities**: The cost tier is selected based on the calculated monthly revenue.
- **Labour**: The model calculates the cost for the base staff defined in `min_count`. If the `current_daily_sales` for the month exceeds 100, one additional "entry" level worker is added for that month.

### 4.3 Capital and ROI
- **Total Capital**: This is the denominator for ROI calculations. It is the sum of all `capital_expenses`, any rent paid before opening, and the `initial_cost` from marketing.
- **Annual ROI**: Calculated at the end of each year as `Annual Net Profit / Total Capital`. This value is used for the equity unlock.
- **Cumulative ROI**: Calculated at the end of each month as `Cumulative Cash / Total Capital`.

### 4.4 Ownership & Equity Unlock
At the end of each year, the simulation performs a two-step equity transfer from investors to the founder based on the `Annual_ROI`. This process is detailed in the `ownership` report. The initial split is assumed to be 25% for the founder and 75% for investors.

## 5. Output Files

The simulation generates four files in the specified output directory.

### 5.1 `results_monthly_<scenario>.csv`
Provides a detailed month-by-month financial breakdown. All currency values are rounded to the nearest integer.
- **Key Columns**:
    - `NumberOfPizzas`: Total pizzas sold in the month.
    - `TotalCapital`: The total initial invested capital (constant value).
    - `Revenue`, `COGS`, `GrossProfit`
    - `OPEX_Total`, `NetProfit`, `CashFlow`, `CashCumulative`
    - `ROI_Cumulative`

### 5.2 `results_annual_<scenario>.csv`
A summary of financials aggregated by year. All currency values are rounded to the nearest integer.

### 5.3 `ownership_<scenario>.csv`
Details the annual equity unlock, showing the ROI for the year, the amount of equity transferred, and the start/end equity split.

### 5.4 `dashboard_<scenario>.html`
An interactive Plotly dashboard containing four charts:
1.  Monthly Revenue / COGS / OPEX
2.  Monthly Net Profit & Cumulative ROI
3.  Annual Equity Unlock & Ownership
4.  Annual Financials

Large numbers on the chart axes and in the hover tooltips are formatted with "k" for thousands and "M" for millions (e.g., "$50k", "$1.2M").