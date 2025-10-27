import streamlit as st
import pandas as pd
import json
from pathlib import Path

# The simulation script is in the same directory, so we can import it directly.
from simulate_pizza_business import simulate_one_scenario

# --- Page Configuration ---
st.set_page_config(
    page_title="Pizza Business ROI Simulation",
    layout="wide"
)

# --- Load Default Config ---
# We use the config for the basic structure and default values.
# Caching ensures we only read the file once.
@st.cache_data
def load_config():
    config_path = Path("config.json")
    if not config_path.exists():
        st.error(f"Error: Configuration file not found at {config_path}")
        st.stop()
    with open(config_path, "r") as f:
        return json.load(f)

config = load_config()

# --- Sidebar for Inputs ---
st.sidebar.header("Simulation Inputs")

# Project Settings
st.sidebar.subheader("Project Settings")
daily_sales = st.sidebar.number_input(
    "Daily Sales Projection (Pizzas/Day)",
    value=config["project"]["daily_sales_projection"],
    step=1
)
yoy_growth = st.sidebar.slider(
    "Year-over-Year Growth (%)",
    min_value=0.0,
    max_value=25.0,
    value=config["project"]["year_over_year_sales_growth"] * 100,
    step=0.5
) / 100.0

# Sales Scenario
st.sidebar.subheader("Sales Model")

# Add a checkbox to toggle between multi-scenario and custom scenario
use_custom_scenario = st.sidebar.checkbox("Define a Custom Scenario")

scenario_options = list(config["sales_models"].keys())
default_scenarios = ["Scenario_4_Baseline_50pctGross"]
selected_scenarios = []

if use_custom_scenario:
    with st.sidebar.expander("Define Custom Multipliers", expanded=True):
        months = ["May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April"]
        
        if 'custom_multipliers' not in st.session_state:
            st.session_state.custom_multipliers = config["sales_models"]["Scenario_3_Middle_Ground"]

        custom_multipliers_list = []
        for i, month in enumerate(months):
            value = st.slider(
                month, 0.0, 5.0, st.session_state.custom_multipliers[i], 0.05, key=f"slider_{month}"
            )
            custom_multipliers_list.append(value)
        
        st.session_state.custom_multipliers = custom_multipliers_list
    
    # When custom is used, we run only one scenario
    selected_scenarios = ["Custom..."]

else:
    selected_scenarios = st.sidebar.multiselect(
        "Select Scenarios to Compare",
        options=scenario_options,
        default=default_scenarios
    )


# Expense & Capital Settings
st.sidebar.subheader("Startup & Capital")
rent_base = st.sidebar.number_input(
    "Base Monthly Rent",
    value=config["expenses"]["rent"]["base"],
    step=100
)
lead_months = st.sidebar.number_input(
    "Lead Months Before Opening",
    value=config["project"].get("lead_months_before_open", 6),
    step=1
)
rent_free_months = st.sidebar.number_input(
    "Rent-Free Months",
    value=config["project"].get("rent_free_months", 4),
    step=1
)
capital_purchases = st.sidebar.number_input(
    "Capital Purchases",
    value=config["expenses"]["capital_expenses"]["capital_purchases"],
    step=1000
)
startup_costs = st.sidebar.number_input(
    "Startup Costs",
    value=config["expenses"]["capital_expenses"]["startup_costs"],
    step=1000
)

# Labour Assumptions
with st.sidebar.expander("Labour Assumptions"):
    labour_roles = config["labour"]["roles"]
    entry_rate = st.number_input("Entry Rate/hr", value=labour_roles["entry"]["hour_rate"], step=1)
    exp_rate = st.number_input("Experienced Rate/hr", value=labour_roles["experienced"]["hour_rate"], step=1)
    manager_rate = st.number_input("Manager Rate/hr", value=labour_roles["manager"]["hour_rate"], step=1)
    
    labour_rules = config["labour"]["scaling_rules"]
    hours_day = st.slider("Hours/Day", 8, 16, value=labour_rules.get("hours_per_day", 10))
    days_month = st.slider("Days/Month", 20, 31, value=labour_rules.get("days_per_month", 28))
    scaling_threshold = st.number_input("Daily Sales Threshold for Extra Worker", value=labour_rules.get("daily_sales_threshold_for_extra_worker", 100))

# Marketing Budget
with st.sidebar.expander("Marketing Budget"):
    mkt_cfg = config["expenses"]["marketing"]
    mkt_initial = st.number_input("Initial Marketing Cost", value=mkt_cfg["initial_cost"], step=500)
    mkt_monthly = st.number_input("Monthly Marketing Budget", value=mkt_cfg["monthly"], step=50)


# --- Main App ---
st.title("🍕 Pizza Business Financial Simulation")

# Create a copy of the config to pass to the simulation.
# This allows us to modify it with the user inputs from the sidebar.
sim_config = config.copy()
sim_config["project"]["daily_sales_projection"] = daily_sales
sim_config["project"]["year_over_year_sales_growth"] = yoy_growth
sim_config["project"]["lead_months_before_open"] = lead_months
sim_config["project"]["rent_free_months"] = rent_free_months
sim_config["expenses"]["rent"]["base"] = rent_base
sim_config["expenses"]["capital_expenses"]["capital_purchases"] = capital_purchases
sim_config["expenses"]["capital_expenses"]["startup_costs"] = startup_costs

# Update labour and marketing configs from the new widgets
sim_config["labour"]["roles"]["entry"]["hour_rate"] = entry_rate
sim_config["labour"]["roles"]["experienced"]["hour_rate"] = exp_rate
sim_config["labour"]["roles"]["manager"]["hour_rate"] = manager_rate
sim_config["labour"]["scaling_rules"]["hours_per_day"] = hours_day
sim_config["labour"]["scaling_rules"]["days_per_month"] = days_month
sim_config["labour"]["scaling_rules"]["daily_sales_threshold_for_extra_worker"] = scaling_threshold
sim_config["expenses"]["marketing"]["initial_cost"] = mkt_initial
sim_config["expenses"]["marketing"]["monthly"] = mkt_monthly

# If the custom scenario is selected, add it to the simulation config
if use_custom_scenario:
    sim_config["sales_models"]["Custom..."] = st.session_state.custom_multipliers


# Run the simulation for each selected scenario
all_monthly_dfs = []
all_annual_dfs = []
all_ownership_dfs = []

if not selected_scenarios:
    st.warning("Please select at least one scenario to run the simulation.")
    st.stop()

for scenario in selected_scenarios:
    try:
        monthly_df, annual_df, ownership_df = simulate_one_scenario(
            cfg=sim_config,
            scenario_name=scenario,
            years=5
        )
        # Add a 'Scenario' column for multi-scenario charting
        monthly_df['Scenario'] = scenario
        
        all_monthly_dfs.append(monthly_df)
        all_annual_dfs.append(annual_df)
        all_ownership_dfs.append(ownership_df)

    except Exception as e:
        st.error(f"An error occurred running scenario '{scenario}': {e}")
        st.stop()

# Aggregate results
combined_monthly = pd.concat(all_monthly_dfs)
combined_annual = pd.concat(all_annual_dfs)
combined_ownership = pd.concat(all_ownership_dfs)


# --- Display Key Metrics ---
st.header("Key Performance Indicators (Year 1)")

# Create columns for each scenario's KPIs
kpi_cols = st.columns(len(selected_scenarios))

for i, scenario in enumerate(selected_scenarios):
    with kpi_cols[i]:
        st.subheader(scenario)
        scenario_annual_df = combined_annual[combined_annual['Scenario'] == scenario]
        y1_net_profit = scenario_annual_df.loc[scenario_annual_df['Year'] == 1, 'NetProfit'].iloc[0]
        y1_roi = scenario_annual_df.loc[scenario_annual_df['Year'] == 1, 'Annual_ROI'].iloc[0]
        
        st.metric("Year 1 Net Profit", f"${y1_net_profit:,.0f}")
        st.metric("Year 1 Annual ROI", f"{y1_roi:.1%}")

# --- Display Charts ---
st.header("Financial Projections Dashboard")

import plotly.express as px

# Create toggles for the chart series
st.write("### Chart Controls")
cols = st.columns(5)
with cols[0]:
    show_revenue = st.checkbox("Revenue", value=True)
with cols[1]:
    show_cogs = st.checkbox("COGS", value=False)
with cols[2]:
    show_opex = st.checkbox("OPEX", value=False)
with cols[3]:
    show_gp = st.checkbox("Gross Profit", value=False)
with cols[4]:
    show_np = st.checkbox("Net Profit", value=True)

# --- Chart 1: Monthly Trends ---
monthly_to_plot = []
if show_revenue: monthly_to_plot.append("Revenue")
if show_cogs: monthly_to_plot.append("COGS")
if show_opex: monthly_to_plot.append("OPEX_Total")
if show_gp: monthly_to_plot.append("GrossProfit")
if show_np: monthly_to_plot.append("NetProfit")

if monthly_to_plot:
    # Melt the dataframe to have a 'Metric' column for line_dash
    monthly_melted = combined_monthly.melt(
        id_vars=['t', 'Scenario'], 
        value_vars=monthly_to_plot,
        var_name='Metric',
        value_name='Amount'
    )
    fig1 = px.line(
        monthly_melted, 
        x="t", 
        y="Amount", 
        color="Scenario",
        line_dash="Metric",
        title="Monthly Financial Trends",
        labels={"t": "Month", "Amount": "Amount (CAD)"}
    )
    fig1.update_layout(yaxis_tickprefix="$", yaxis_tickformat=".2s")
    st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Annual Summary ---
annual_to_plot = []
if show_revenue: annual_to_plot.append("Revenue")
if show_cogs: annual_to_plot.append("COGS")
if show_opex: annual_to_plot.append("OPEX_Total")
if show_gp: annual_to_plot.append("GrossProfit")
if show_np: annual_to_plot.append("NetProfit")

if annual_to_plot:
    # Melt the dataframe for the bar chart as well
    annual_melted = combined_annual.melt(
        id_vars=['Year', 'Scenario'],
        value_vars=annual_to_plot,
        var_name='Metric',
        value_name='Amount'
    )
    fig2 = px.bar(
        annual_melted,
        x="Year",
        y="Amount",
        color="Scenario",
        barmode="group",
        pattern_shape="Metric",
        title="Annual Financial Summary",
        labels={"Amount": "Amount (CAD)"}
    )
    fig2.update_layout(yaxis_tickprefix="$", yaxis_tickformat=".2s")
    st.plotly_chart(fig2, use_container_width=True)


# --- Display Data Tables ---
st.header("Detailed Data")

st.subheader("Combined Monthly Financials")
st.dataframe(combined_monthly)

st.subheader("Combined Annual Summary")
st.dataframe(combined_annual)

st.subheader("Combined Ownership & Equity Unlock")
st.dataframe(combined_ownership)
