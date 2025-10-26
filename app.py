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
scenario_options = list(config["sales_models"].keys()) + ["Custom..."]
try:
    default_scenario_index = scenario_options.index("Scenario_3_Middle_Ground")
except ValueError:
    default_scenario_index = 0

selected_scenario = st.sidebar.selectbox(
    "Select a Sales Scenario",
    options=scenario_options,
    index=default_scenario_index
)

if selected_scenario == "Custom...":
    with st.sidebar.expander("Define Custom Multipliers", expanded=True):
        months = ["May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April"]
        
        # Use session_state to store and persist the custom multiplier values
        if 'custom_multipliers' not in st.session_state:
            st.session_state.custom_multipliers = config["sales_models"]["Scenario_3_Middle_Ground"]

        custom_multipliers_list = []
        for i, month in enumerate(months):
            # The key ensures each slider is unique
            value = st.slider(
                month, 0.0, 5.0, st.session_state.custom_multipliers[i], 0.05, key=f"slider_{month}"
            )
            custom_multipliers_list.append(value)
        
        # Update session state with the latest values from the sliders
        st.session_state.custom_multipliers = custom_multipliers_list


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
if selected_scenario == "Custom...":
    sim_config["sales_models"]["Custom..."] = st.session_state.custom_multipliers


# Run the simulation
try:
    monthly_df, annual_df, ownership_df = simulate_one_scenario(
        cfg=sim_config,
        scenario_name=selected_scenario,
        years=5
    )

    # --- Display Key Metrics ---
    st.header("Key Performance Indicators (Year 1)")
    y1_net_profit = annual_df.loc[annual_df['Year'] == 1, 'NetProfit'].iloc[0]
    y1_roi = annual_df.loc[annual_df['Year'] == 1, 'Annual_ROI'].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Year 1 Net Profit", f"${y1_net_profit:,.0f}")
    col2.metric("Year 1 Annual ROI", f"{y1_roi:.1%}")

    # --- Display Charts ---
    st.header("Financial Projections Dashboard")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Chart 1: Monthly Revenue vs COGS vs OPEX
    fig1 = make_subplots()
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["Revenue"], name="Revenue", hovertemplate='$%{y:,.0f}'))
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["COGS"], name="COGS", hovertemplate='$%{y:,.0f}'))
    fig1.add_trace(go.Scatter(x=monthly_df["t"], y=monthly_df["OPEX_Total"], name="OPEX", hovertemplate='$%{y:,.0f}'))
    fig1.update_layout(title="Monthly Revenue, COGS, and OPEX", yaxis_tickprefix="$", yaxis_tickformat=".2s")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Annual Financials
    fig4 = make_subplots()
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["Revenue"], name="Revenue", hovertemplate='$%{y:,.0f}'))
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["GrossProfit"], name="Gross Profit", hovertemplate='$%{y:,.0f}'))
    fig4.add_trace(go.Bar(x=annual_df["Year"], y=annual_df["NetProfit"], name="Net Profit", hovertemplate='$%{y:,.0f}'))
    fig4.update_layout(barmode="group", title="Annual Financials", yaxis_tickprefix="$", yaxis_tickformat=".2s")
    st.plotly_chart(fig4, use_container_width=True)


    # --- Display Data Tables ---
    st.header("Detailed Data")

    st.subheader("Monthly Financials")
    st.dataframe(monthly_df)

    st.subheader("Annual Summary")
    st.dataframe(annual_df)

    st.subheader("Ownership & Equity Unlock")
    st.dataframe(ownership_df)


except Exception as e:
    st.error(f"An error occurred during simulation: {e}")
    st.exception(e)
