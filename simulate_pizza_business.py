#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_pizza_business.py

A self-contained Python simulation for a pizza restaurant venture.
- Reads inputs from a JSON config (see example schema below).
- Computes monthly revenue, COGS, OPEX, net profit, cash flow, ROI.
- Applies rent escalation, utilities tiers by sales, labour scaling by sales.
- Supports multiple sales scenarios (monthly multipliers).
- Computes annual ROI and applies ROI-based equity unlock (two-step).
- Generates interactive Plotly charts and CSV outputs.

Run:
    python simulate_pizza_business.py --config config.json --scenario Scenario_3_Middle_Ground

Outputs (in the same folder by default):
    - results_monthly_<scenario>.csv
    - results_annual_<scenario>.csv
    - ownership_<scenario>.csv
    - dashboard_<scenario>.html
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------
# Helpers
# ------------------------------

MONTHS = ["May","June","July","August","September","October","November","December","January","February","March","April"]

def month_index_map(base_month: str) -> Dict[str, int]:
    """Return a mapping from month name to 0..11 index, rotated so base_month is index 0."""
    base_month = base_month.strip()
    if base_month not in MONTHS:
        # Fallback to May if unknown
        base_month = "May"
    start = MONTHS.index(base_month)
    ordered = MONTHS[start:] + MONTHS[:start]
    return {m: i for i, m in enumerate(ordered)}


def get_base_anchor_revenue(cfg: Dict[str, Any]) -> float:
    """
    Base revenue for month 0.
    You can specify it as project.base_month_revenue in the config.
    If not provided, we use a safe default = 50,000 (CAD).
    """
    return float(cfg.get("project", {}).get("base_month_revenue", 50000.0))


def annualize_rent(month_idx: int, base_rent: float, annual_increase_rate: float, base_month_idx: int = 0) -> float:
    """
    Return rent for a given month index with annual step-ups.
    Example: if base_month_idx=0 is May, then Month 12 (next May) jumps by annual increase rate.
    """
    years_elapsed = (month_idx - base_month_idx) // 12
    return base_rent * ((1.0 + annual_increase_rate) ** max(0, years_elapsed))


def utilities_cost_for_sales(utilities_cfg: Dict[str, Any], monthly_sales: float) -> float:
    """Pick utilities tier cost by monthly sales level. If out of range, pick nearest tier."""
    tiers = utilities_cfg.get("tiers", [])
    if not tiers:
        return 0.0
    # Try to find a matching tier
    for t in tiers:
        lo = t.get("sales_min", 0.0)
        hi = t.get("sales_max", float("inf"))
        cost = float(t.get("cost", 0.0))
        if monthly_sales >= lo and monthly_sales < hi:
            return cost
    # Fallback: choose the max tier's cost if above all ranges
    return float(max(tiers, key=lambda x: x.get("sales_max", 0.0)).get("cost", 0.0))


def labour_cost_for_sales(labour_cfg: Dict[str, Any], monthly_sales: float) -> float:
    """
    Compute monthly labour cost:
    - Start with minimum staffing (entry, experienced, manager).
    - Add additional entry-level headcount as sales grow: +1 per 'sales_per_additional_worker' of monthly sales.
      (You can refine later to add experienced roles too if desired.)
    - Total hours = hours_per_day * days_per_month per worker.
    """
    roles = labour_cfg.get("roles", {})
    rules = labour_cfg.get("scaling_rules", {})
    hours_per_day = float(rules.get("hours_per_day", 12))
    days_per_month = float(rules.get("days_per_month", 30))
    sales_per_worker = float(rules.get("sales_per_additional_worker", 30000))

    # Base staffing
    total_monthly_cost = 0.0
    for role_name, role in roles.items():
        rate = float(role.get("hour_rate", 0.0))
        min_count = int(role.get("min_count", 0))
        hours = hours_per_day * days_per_month
        total_monthly_cost += rate * hours * min_count

    # Additional scaling: we'll add entry-level workers first for simplicity
    extra_workers = 0
    if sales_per_worker > 0:
        extra_workers = int(max(0, math.floor(monthly_sales / sales_per_worker)))
    entry_role = roles.get("entry", {"hour_rate": 18})
    entry_rate = float(entry_role.get("hour_rate", 18))
    hours = hours_per_day * days_per_month
    total_monthly_cost += extra_workers * entry_rate * hours

    return total_monthly_cost


def cogs_from_menu_and_mix(menu_cfg: Dict[str, Any], monthly_revenue: float) -> Tuple[float, float, float]:
    """
    Estimate COGS based on menu-level costs and prices and the sales mix.
    Approach: compute weighted average gross margin from traditional vs seafood item baskets,
    then apply to monthly revenue.
    Return: (cogs, gross_profit, gross_margin_pct)
    """
    def avg_margin_one(category: Dict[str, Any]) -> float:
        # average margin across items (simple average of item margins)
        margins = []
        for _, item in category.items():
            price = float(item.get("price", 0))
            cost = float(item.get("cost", 0))
            if price <= 0:
                continue
            margins.append((price - cost) / price)
        return np.mean(margins) if margins else 0.6  # default if missing

    sales_mix = menu_cfg.get("sales_mix", {"traditional": 0.66, "seafood": 0.34})
    trad = menu_cfg.get("traditional", {})
    sea = menu_cfg.get("seafood", {})

    trad_margin = avg_margin_one(trad)
    sea_margin  = avg_margin_one(sea)

    blended_margin = trad_margin * sales_mix.get("traditional", 0.66) + sea_margin * sales_mix.get("seafood", 0.34)
    gross_profit = monthly_revenue * blended_margin
    cogs = monthly_revenue - gross_profit
    gross_margin_pct = blended_margin
    return cogs, gross_profit, gross_margin_pct


def build_monthly_revenue(base_anchor: float, multipliers: List[float], yoy_growth: float, years: int = 1) -> List[float]:
    """
    Build monthly revenue array for N years using provided multipliers (length 12) and optional YoY growth.
    Example: if years=5, you'll get 60 months of revenue, where each block of 12 months
    is increased by (1 + yoy_growth) ** year_index.
    """
    if len(multipliers) != 12:
        raise ValueError("Each sales scenario must provide exactly 12 monthly multipliers.")
    months = []
    for y in range(years):
        growth_factor = (1.0 + yoy_growth) ** y
        months.extend([base_anchor * m * growth_factor for m in multipliers])
    return months


def simulate_one_scenario(cfg: Dict[str, Any], scenario_name: str, years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate monthly financials for the requested scenario over N years (default 5).
    Returns:
      - monthly_df: per-month detail (revenue, cogs, gp, opex lines, net, cash, ROI cumulative)
      - annual_df: aggregated per year
      - ownership_df: annual ownership & equity unlocks
    """
    project = cfg.get("project", {})
    expenses_cfg = cfg.get("expenses", {})
    menu_cfg = cfg.get("menu", {})
    labour_cfg = cfg.get("labour", {})
    invest_cfg = cfg.get("investment_model", {})
    sales_models = cfg.get("sales_models", {})

    # Base anchors
    currency = project.get("currency", "CAD")
    base_month = project.get("base_month", "May")
    yoy_growth = float(project.get("year_over_year_sales_growth", 0.10))
    base_anchor_revenue = get_base_anchor_revenue(cfg)

    # Sales multipliers for scenario
    if scenario_name not in sales_models:
        raise ValueError(f"Scenario '{scenario_name}' not found in sales_models.")
    multipliers = list(map(float, sales_models[scenario_name]))

    # Build monthly revenue series (years * 12 months)
    monthly_revenue = build_monthly_revenue(base_anchor_revenue, multipliers, yoy_growth, years=years)

    # Rent & lead months logic
    rent_cfg = expenses_cfg.get("rent", {"base": 0.0, "annual_increase_rate": 0.0})
    base_rent = float(rent_cfg.get("base", 0.0))
    rent_rate = float(rent_cfg.get("annual_increase_rate", 0.0))
    lead_months = int(project.get("lead_months_before_open", 6))
    rent_free_months = int(project.get("rent_free_months", 4))
    rent_paid_during_lead = max(0, lead_months - rent_free_months)

    # Capital expenses: sum all keys
    cap_cfg = expenses_cfg.get("capital_expenses", {})
    total_capital = sum(float(v) for v in cap_cfg.values())

    # Add pre-open rent (paid months) to capital base
    if rent_paid_during_lead > 0:
        # Use base_rent for pre-open months
        total_capital += base_rent * rent_paid_during_lead

    # OPEX fixed lines
    equip_lease = float(expenses_cfg.get("equipment_lease", {}).get("monthly", 0.0))
    insurance = float(expenses_cfg.get("insurance", {}).get("monthly", 0.0))
    accounting = float(expenses_cfg.get("accounting", {}).get("monthly", 0.0))

    # Marketing
    mkt = expenses_cfg.get("marketing", {})
    mkt_initial = float(mkt.get("initial_cost", 0.0))
    mkt_monthly = float(mkt.get("monthly", 0.0))

    meals = float(expenses_cfg.get("meals_and_entertainment", {}).get("monthly_base", 0.0))
    office = float(expenses_cfg.get("office_expense", {}).get("monthly_base", 0.0))

    utilities_cfg = expenses_cfg.get("utilities", {"tiers": []})

    # Monthly simulation
    rows = []
    month_map = month_index_map(base_month)
    start_month_idx = month_map[base_month]  # normally 0
    n_months = years * 12

    # Track cumulative cash and invested capital (capital + any pre-open rent)
    cash = 0.0
    cumulative_invested = total_capital  # initial investment basis
    # Treat marketing initial as part of capital base (so it affects ROI denominator)
    if mkt_initial > 0:
        cumulative_invested += mkt_initial

    # For the first operational month, subtract marketing initial cost from cash (startup outflow)
    marketing_initial_paid = False

    for t in range(n_months):
        year_num = (t // 12) + 1
        month_in_year_idx = t % 12
        month_name = MONTHS[(start_month_idx + month_in_year_idx) % 12]

        revenue = monthly_revenue[t]
        # COGS via menu + mix
        cogs, gp, gm_pct = cogs_from_menu_and_mix(menu_cfg, revenue)

        # OPEX
        this_rent = annualize_rent(t, base_rent, rent_rate, base_month_idx=0)
        this_equip = equip_lease
        this_ins = insurance
        this_acc = accounting
        this_mkt = mkt_monthly
        this_meals = meals
        this_office = office
        this_util = utilities_cost_for_sales(utilities_cfg, revenue)
        this_labour = labour_cost_for_sales(labour_cfg, revenue)

        opex = this_rent + this_equip + this_ins + this_acc + this_mkt + this_meals + this_office + this_util + this_labour

        # Net profit (operating)
        net = gp - opex

        # Cash flow: add net; subtract marketing initial once in first month
        cf = net
        if not marketing_initial_paid and mkt_initial > 0:
            cf -= mkt_initial
            marketing_initial_paid = True

        cash += cf

        # ROI cumulative uses the capital base including startup items
        roi_cum = (cash) / max(1e-9, cumulative_invested)

        rows.append({
            "Year": year_num,
            "Month": month_name,
            "t": t+1,
            "Revenue": revenue,
            "COGS": cogs,
            "GrossProfit": gp,
            "GrossMarginPct": gm_pct,
            "Rent": this_rent,
            "EquipmentLease": this_equip,
            "Insurance": this_ins,
            "Accounting": this_acc,
            "MarketingMonthly": this_mkt,
            "MarketingInitialPaid": (mkt_initial if marketing_initial_paid and t == 0 else 0.0),
            "MealsOffice": this_meals + this_office,
            "Utilities": this_util,
            "Labour": this_labour,
            "OPEX_Total": opex,
            "NetProfit": net,
            "CashFlow": cf,
            "CashCumulative": cash,
            "ROI_Cumulative": roi_cum
        })

    monthly_df = pd.DataFrame(rows)

    # Annual aggregation
    annual = monthly_df.groupby("Year").agg({
        "Revenue": "sum",
        "COGS": "sum",
        "GrossProfit": "sum",
        "OPEX_Total": "sum",
        "NetProfit": "sum",
        "CashFlow": "sum"
    }).reset_index()
    # Annual ROI (business) = Annual NetProfit / initial_investment (capital base)
    initial_investment = float(project.get("initial_investment", 180000.0))
    annual["Annual_ROI"] = annual["NetProfit"] / max(1e-9, initial_investment)

    # Ownership & equity unlock per year
    roi_min = float(invest_cfg.get("roi_range", {}).get("min", 0.30))
    roi_max = float(invest_cfg.get("roi_range", {}).get("max", 1.10))
    unlock1_max = float(invest_cfg.get("unlock_step_1", 0.15))
    unlock2_max = float(invest_cfg.get("unlock_step_2", 0.075))
    unlock2_threshold = float(invest_cfg.get("unlock_step_2_threshold", roi_max))

    founder_eq = float(project.get("founder_equity_initial", 0.25))
    investor_eq = float(project.get("investor_equity_initial", 0.75))

    own_rows = []
    for _, row in annual.iterrows():
        y = int(row["Year"])
        roi = float(row["Annual_ROI"])

        # step 1 unlock fraction in [0, unlock1_max]
        if roi <= roi_min:
            unlock1 = 0.0
        elif roi >= roi_max:
            unlock1 = unlock1_max
        else:
            unlock1 = unlock1_max * (roi - roi_min) / max(1e-9, (roi_max - roi_min))

        # step 2 bonus unlock if ROI > threshold
        unlock2 = 0.0
        if roi > unlock2_threshold:
            unlock2 = unlock2_max * (roi - roi_max) / max(1e-9, (roi_max - roi_min))
            unlock2 = max(0.0, min(unlock2, unlock2_max))

        total_unlock = unlock1 + unlock2
        total_unlock = min(total_unlock, investor_eq)  # cap by what's available

        own_rows.append({
            "Year": y,
            "Annual_ROI": roi,
            "Unlock_Step1": unlock1,
            "Unlock_Step2": unlock2,
            "Unlock_Total": total_unlock,
            "Founder_Equity_Start": founder_eq,
            "Investor_Equity_Start": investor_eq,
            "Founder_Equity_End": founder_eq + total_unlock,
            "Investor_Equity_End": investor_eq - total_unlock
        })

        # Apply transfer for next year's start
        founder_eq += total_unlock
        investor_eq -= total_unlock

        if investor_eq <= 0:
            investor_eq = 0.0
            break

    ownership_df = pd.DataFrame(own_rows)

    # Attach scenario metadata
    monthly_df["Scenario"] = scenario_name
    annual["Scenario"] = scenario_name
    ownership_df["Scenario"] = scenario_name

    return monthly_df, annual, ownership_df


def build_dashboard(monthly_df: pd.DataFrame, annual_df: pd.DataFrame, ownership_df: pd.DataFrame, scenario_name: str, out_html: Path):
    """Create a compact interactive Plotly dashboard and write to HTML."""
    # --- Figure 1: Monthly Revenue vs COGS vs OPEX ---
    fig1 = make_subplots(specs=[[{"secondary_y": False}]], rows=1, cols=1)
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["Revenue"], name="Revenue"))
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["COGS"], name="COGS"))
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["OPEX_Total"], name="OPEX"))
    fig1.update_layout(title=f"Monthly Revenue / COGS / OPEX — {scenario_name}", xaxis_title="Month", yaxis_title="CAD")

    # --- Figure 2: Monthly Net Profit & Cumulative ROI ---
    fig2 = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
    fig2.add_trace(go.Bar(x=monthly_df["t"], y=monthly_df["NetProfit"], name="Net Profit"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["ROI_Cumulative"], name="Cumulative ROI"), secondary_y=True)
    fig2.update_layout(title=f"Monthly Net Profit & Cumulative ROI — {scenario_name}")
    fig2.update_xaxes(title="Month")
    fig2.update_yaxes(title_text="Net Profit (CAD)", secondary_y=False)
    fig2.update_yaxes(title_text="Cumulative ROI", secondary_y=True)

    # --- Figure 3: Annual Ownership Transfer ---
    if not ownership_df.empty:
        fig3 = make_subplots(specs=[[{"secondary_y": False}]])
        fig3.add_trace(go.Bar(x=ownership_df["Year"], y=ownership_df["Unlock_Total"], name="Equity Unlocked (pp)"))
        fig3.add_trace(go.Scatter(x=ownership_df["Year"], y=ownership_df["Founder_Equity_End"], name="Founder Equity (End)"))
        fig3.add_trace(go.Scatter(x=ownership_df["Year"], y=ownership_df["Investor_Equity_End"], name="Investor Equity (End)"))
        fig3.update_layout(title=f"Annual Equity Unlock & Ownership — {scenario_name}", xaxis_title="Year", yaxis_title="Equity (0..1)")
    else:
        fig3 = go.Figure()
        fig3.update_layout(title=f"Annual Equity Unlock & Ownership — {scenario_name} (No unlocks)")

    # --- Figure 4: Annual P&L ---
    fig4 = make_subplots(specs=[[{"secondary_y": False}]])
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["Revenue"], name="Revenue"))
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["GrossProfit"], name="Gross Profit"))
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["NetProfit"], name="Net Profit"))
    fig4.update_layout(barmode="group", title=f"Annual Financials — {scenario_name}", xaxis_title="Year", yaxis_title="CAD")

    # Assemble simple HTML
    html_parts = []
    for fig in [fig1, fig2, fig3, fig4]:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html = f"""
    <html>
    <head><meta charset='utf-8'><title>Dashboard — {scenario_name}</title></head>
    <body>
        <h1>Dashboard — {scenario_name}</h1>
        <p>Interactive charts for the selected scenario. Hover to inspect values; click legend to toggle series.</p>
        {''.join(html_parts)}
    </body>
    </html>
    """
    out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Simulate pizza business scenarios.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file.")
    parser.add_argument("--scenario", type=str, default="Scenario_3_Middle_Ground", help="Scenario key from 'sales_models' in config.")
    parser.add_argument("--years", type=int, default=5, help="Number of years to simulate (default 5).")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Defaults for lead months & rent-free months if not present (as per your request)
    proj = cfg.setdefault("project", {})
    proj.setdefault("lead_months_before_open", 6)
    proj.setdefault("rent_free_months", 4)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    monthly_df, annual_df, ownership_df = simulate_one_scenario(cfg, args.scenario, years=args.years)

    # Write CSVs
    monthly_csv = outdir / f"results_monthly_{args.scenario}.csv"
    annual_csv = outdir / f"results_annual_{args.scenario}.csv"
    owner_csv  = outdir / f"ownership_{args.scenario}.csv"
    monthly_df.to_csv(monthly_csv, index=False)
    annual_df.to_csv(annual_csv, index=False)
    ownership_df.to_csv(owner_csv, index=False)

    # Build dashboard
    html_path = outdir / f"dashboard_{args.scenario}.html"
    build_dashboard(monthly_df, annual_df, ownership_df, args.scenario, html_path)

    # Print summary pointers
    print(f"✅ Wrote: {monthly_csv}")
    print(f"✅ Wrote: {annual_csv}")
    print(f"✅ Wrote: {owner_csv}")
    print(f"✅ Dashboard: {html_path}")


if __name__ == "__main__":
    main()
