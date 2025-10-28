import streamlit as st
import pandas as pd
import json
from pathlib import Path
from io import BytesIO
import numpy as np
import plotly.express as px
import openpyxl
from openpyxl.drawing.image import Image

try:
    # This works when run as a package (e.g., on Streamlit Cloud or with `python -m`)
    from .simulate_pizza_business import simulate_one_scenario, simulate_monte_carlo, MCDist
except ImportError:
    # This works when run as a script directly (for local development)
    from simulate_pizza_business import simulate_one_scenario, simulate_monte_carlo, MCDist


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


# --- Session State Initialization ---
# Initialize session_state to hold the values of the widgets
# This is key to allowing the widgets to be updated programmatically
def init_session_state():
    if 'daily_sales' not in st.session_state:
        st.session_state.daily_sales = config["project"]["daily_sales_projection"]
    if 'yoy_growth' not in st.session_state:
        st.session_state.yoy_growth = config["project"]["year_over_year_sales_growth"] * 100
    if 'rent_base' not in st.session_state:
        st.session_state.rent_base = config["expenses"]["rent"]["base"]
    if 'lead_months' not in st.session_state:
        st.session_state.lead_months = config["project"].get("lead_months_before_open", 6)
    if 'rent_free_months' not in st.session_state:
        st.session_state.rent_free_months = config["project"].get("rent_free_months", 4)
    if 'capital_purchases' not in st.session_state:
        st.session_state.capital_purchases = config["expenses"]["capital_expenses"]["capital_purchases"]
    if 'startup_costs' not in st.session_state:
        st.session_state.startup_costs = config["expenses"]["capital_expenses"]["startup_costs"]
    if 'entry_rate' not in st.session_state:
        st.session_state.entry_rate = config["labour"]["roles"]["entry"]["hour_rate"]
    if 'exp_rate' not in st.session_state:
        st.session_state.exp_rate = config["labour"]["roles"]["experienced"]["hour_rate"]
    if 'manager_rate' not in st.session_state:
        st.session_state.manager_rate = config["labour"]["roles"]["manager"]["hour_rate"]
    if 'hours_day' not in st.session_state:
        st.session_state.hours_day = config["labour"]["scaling_rules"].get("hours_per_day", 10)
    if 'days_month' not in st.session_state:
        st.session_state.days_month = config["labour"]["scaling_rules"].get("days_per_month", 28)
    if 'scaling_threshold' not in st.session_state:
        st.session_state.scaling_threshold = config["labour"]["scaling_rules"].get("daily_sales_threshold_for_extra_worker", 100)
    if 'mkt_initial' not in st.session_state:
        st.session_state.mkt_initial = config["expenses"]["marketing"]["initial_cost"]
    if 'mkt_monthly' not in st.session_state:
        st.session_state.mkt_monthly = config["expenses"]["marketing"]["monthly"]
    if 'custom_multipliers' not in st.session_state:
        st.session_state.custom_multipliers = config["sales_models"]["Scenario_3_Middle_Ground"]

init_session_state()


# --- Helper Function for Excel Export ---
def to_excel(params, monthly, annual, ownership):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Parameters Sheet
        params_df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Data Sheets
        monthly.to_excel(writer, sheet_name='Monthly_Data', index=False)
        annual.to_excel(writer, sheet_name='Annual_Data', index=False)
        ownership.to_excel(writer, sheet_name='Ownership_Data', index=False)
        
    processed_data = output.getvalue()
    return processed_data


# --- SIDEBAR (Unified) ---
st.sidebar.header("Load/Save Scenario")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel Report to Load Parameters",
    type="xlsx"
)

if uploaded_file is not None:
    if st.sidebar.button("Load Parameters from File"):
        try:
            params_df = pd.read_excel(uploaded_file, sheet_name="Parameters")
            imported_params = dict(zip(params_df['Parameter'], params_df['Value']))
            
            # --- Mapping from Excel names to session_state keys ---
            param_mapping = {
                "Daily Sales Projection": "daily_sales",
                "YoY Growth": "yoy_growth",
                "Base Monthly Rent": "rent_base",
                "Lead Months Before Opening": "lead_months",
                "Rent-Free Months": "rent_free_months",
                "Capital Purchases": "capital_purchases",
                "Startup Costs": "startup_costs",
                "Entry Rate/hr": "entry_rate",
                "Experienced Rate/hr": "exp_rate",
                "Manager Rate/hr": "manager_rate",
                "Hours/Day": "hours_day",
                "Days/Month": "days_month",
                "Sales Threshold for Extra Worker": "scaling_threshold",
                "Initial Marketing Cost": "mkt_initial",
                "Monthly Marketing Budget": "mkt_monthly"
            }

            for param_name, state_key in param_mapping.items():
                if param_name in imported_params:
                    # No special handling needed now, just load the value as is
                    st.session_state[state_key] = imported_params[param_name]
            
            st.sidebar.success("Parameters loaded successfully!")

        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")


st.sidebar.header("Simulation Inputs")

# Project Settings
st.sidebar.subheader("Project Settings")
st.sidebar.number_input(
    "Daily Sales Projection (Pizzas/Day)",
    step=1,
    key="daily_sales"
)
st.sidebar.slider(
    "Year-over-Year Growth (%)",
    min_value=0.0,
    max_value=25.0,
    step=0.5,
    key="yoy_growth"
)

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
        
        custom_multipliers_list = []
        for i, month in enumerate(months):
            value = st.slider(
                month, 0.0, 5.0, st.session_state.custom_multipliers[i], 0.05, key=f"slider_{month}"
            )
            custom_multipliers_list.append(value)
        
        st.session_state.custom_multipliers = custom_multipliers_list
    
    selected_scenarios = ["Custom..."]

else:
    selected_scenarios = st.sidebar.multiselect(
        "Select Scenarios to Compare",
        options=scenario_options,
        default=default_scenarios
    )


# Expense & Capital Settings
st.sidebar.subheader("Startup & Capital")
st.sidebar.number_input("Base Monthly Rent", step=100, key="rent_base")
st.sidebar.number_input("Lead Months Before Opening", step=1, key="lead_months")
st.sidebar.number_input("Rent-Free Months", step=1, key="rent_free_months")
st.sidebar.number_input("Capital Purchases", step=1000, key="capital_purchases")
st.sidebar.number_input("Startup Costs", step=1000, key="startup_costs")

# Labour Assumptions
with st.sidebar.expander("Labour Assumptions"):
    st.number_input("Entry Rate/hr", step=1, key="entry_rate")
    st.number_input("Experienced Rate/hr", step=1, key="exp_rate")
    st.number_input("Manager Rate/hr", step=1, key="manager_rate")
    st.slider("Hours/Day", 8, 16, key="hours_day")
    st.slider("Days/Month", 20, 31, key="days_month")
    st.number_input("Daily Sales Threshold for Extra Worker", key="scaling_threshold")

# Marketing Budget
with st.sidebar.expander("Marketing Budget"):
    st.number_input("Initial Marketing Cost", step=500, key="mkt_initial")
    st.number_input("Monthly Marketing Budget", step=50, key="mkt_monthly")


# --- MAIN PANEL ---
st.title("🍕 Pizza Business Financial Simulation")

# --- Deterministic Analysis (Always runs) ---
st.header("Deterministic Analysis")

# Create a copy of the config to pass to the simulation.
sim_config = config.copy()
sim_config["project"]["daily_sales_projection"] = st.session_state.daily_sales
sim_config["project"]["year_over_year_sales_growth"] = st.session_state.yoy_growth / 100.0 # Convert from % to decimal for calculation
sim_config["project"]["lead_months_before_open"] = st.session_state.lead_months
sim_config["project"]["rent_free_months"] = st.session_state.rent_free_months
sim_config["expenses"]["rent"]["base"] = st.session_state.rent_base
sim_config["expenses"]["capital_expenses"]["capital_purchases"] = st.session_state.capital_purchases
sim_config["expenses"]["capital_expenses"]["startup_costs"] = st.session_state.startup_costs

# Update labour and marketing configs from the new widgets
sim_config["labour"]["roles"]["entry"]["hour_rate"] = st.session_state.entry_rate
sim_config["labour"]["roles"]["experienced"]["hour_rate"] = st.session_state.exp_rate
sim_config["labour"]["roles"]["manager"]["hour_rate"] = st.session_state.manager_rate
sim_config["labour"]["scaling_rules"]["hours_per_day"] = st.session_state.hours_day
sim_config["labour"]["scaling_rules"]["days_per_month"] = st.session_state.days_month
sim_config["labour"]["scaling_rules"]["daily_sales_threshold_for_extra_worker"] = st.session_state.scaling_threshold
sim_config["expenses"]["marketing"]["initial_cost"] = st.session_state.mkt_initial
sim_config["expenses"]["marketing"]["monthly"] = st.session_state.mkt_monthly

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

# Create toggles for the chart series
st.write("### Chart Controls")
cols = st.columns(5)
with cols[0]:
    show_revenue = st.checkbox("Revenue", value=True, key="det_revenue")
with cols[1]:
    show_cogs = st.checkbox("COGS", value=False, key="det_cogs")
with cols[2]:
    show_opex = st.checkbox("OPEX", value=False, key="det_opex")
with cols[3]:
    show_gp = st.checkbox("Gross Profit", value=False, key="det_gp")
with cols[4]:
    show_np = st.checkbox("Net Profit", value=True, key="det_np")

# --- Chart 1: Monthly Trends ---
monthly_to_plot = []
if show_revenue and "Revenue" in combined_monthly.columns: monthly_to_plot.append("Revenue")
if show_cogs and "COGS" in combined_monthly.columns: monthly_to_plot.append("COGS")
if show_opex and "OPEX_Total" in combined_monthly.columns: monthly_to_plot.append("OPEX_Total")
if show_gp and "GrossProfit" in combined_monthly.columns: monthly_to_plot.append("GrossProfit")
if show_np and "NetProfit" in combined_monthly.columns: monthly_to_plot.append("NetProfit")

if monthly_to_plot:
    # Melt the dataframe to have a 'Metric' column for line_dash
    monthly_melted = combined_monthly.melt(
        id_vars=['t', 'Scenario', 'MonthLabel'], 
        value_vars=monthly_to_plot,
        var_name='Metric',
        value_name='Amount'
    )
    fig1 = px.line(
        monthly_melted, 
        x="MonthLabel", 
        y="Amount", 
        color="Scenario",
        line_dash="Metric",
        title="Monthly Financial Trends",
        labels={"MonthLabel": "Month", "Amount": "Amount (CAD)"}
    )
    # Sort x-axis chronologically
    fig1.update_xaxes(categoryorder='array', categoryarray=combined_monthly['MonthLabel'].unique())
    fig1.update_layout(yaxis_tickprefix="$", yaxis_tickformat=".2s")
    st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Annual Summary ---
annual_to_plot = []
if show_revenue and "Revenue" in combined_annual.columns: annual_to_plot.append("Revenue")
if show_cogs and "COGS" in combined_annual.columns: annual_to_plot.append("COGS")
if show_opex and "OPEX_Total" in combined_annual.columns: annual_to_plot.append("OPEX_Total")
if show_gp and "GrossProfit" in combined_annual.columns: annual_to_plot.append("GrossProfit")
if show_np and "NetProfit" in combined_annual.columns: annual_to_plot.append("NetProfit")

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

# --- Download Button Logic ---
# Collect all parameters for the report
current_params = {
    "Daily Sales Projection": st.session_state.daily_sales,
    "YoY Growth": st.session_state.yoy_growth,
    "Selected Scenarios": ", ".join(selected_scenarios) if not use_custom_scenario else "Custom",
    "Base Monthly Rent": st.session_state.rent_base,
    "Lead Months Before Opening": st.session_state.lead_months,
    "Rent-Free Months": st.session_state.rent_free_months,
    "Capital Purchases": st.session_state.capital_purchases,
    "Startup Costs": st.session_state.startup_costs,
    "Entry Rate/hr": st.session_state.entry_rate,
    "Experienced Rate/hr": st.session_state.exp_rate,
    "Manager Rate/hr": st.session_state.manager_rate,
    "Hours/Day": st.session_state.hours_day,
    "Days/Month": st.session_state.days_month,
    "Sales Threshold for Extra Worker": st.session_state.scaling_threshold,
    "Initial Marketing Cost": st.session_state.mkt_initial,
    "Monthly Marketing Budget": st.session_state.mkt_monthly
}
if use_custom_scenario:
    months = ["May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April"]
    for i, month in enumerate(months):
        current_params[f"Multiplier - {month}"] = st.session_state.custom_multipliers[i]


excel_data = to_excel(
    current_params, 
    combined_monthly, 
    combined_annual, 
    combined_ownership
)

st.sidebar.download_button(
    label="Download Full Report as Excel",
    data=excel_data,
    file_name=f"simulation_report_{'_'.join(selected_scenarios)}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# --- Monte Carlo Analysis (Runs on button click) ---
st.header("Monte Carlo Simulation")
st.write("Run a probabilistic simulation to understand the range of potential outcomes.")

mc_cols = st.columns(4)
with mc_cols[0]:
    n_runs = st.number_input("Number of Runs", 1000, 100000, 10000)
with mc_cols[1]:
    seed = st.number_input("Random Seed", 1, 1000000, 12345)
with mc_cols[2]:
    demand_sigma = st.number_input("Demand Noise Sigma", 0.0, 1.0, 0.1)
with mc_cols[3]:
    kappa = st.number_input("Menu Mix Kappa", 0.0, 10.0, 1.0)

if st.button("Run Monte Carlo Simulation"):
    dist_overrides = {
        "demand_noise": MCDist("lognormal", {"mu": 0, "sigma": demand_sigma}),
        "menu_mix": MCDist("dirichlet", {"alpha": np.array([item['ratio'] for item in config['menu']['items']]) * kappa}),
    }
    
    scenario_name = selected_scenarios[0] if selected_scenarios else list(config['sales_models'].keys())[0]
    mc_results = simulate_monte_carlo(
        sim_config,
        scenario_name=scenario_name,
        n_runs=n_runs,
        seed=seed,
        dist_overrides=dist_overrides
    )
    st.session_state.mc_results = mc_results
    # I don't know why this is necessary, but it is.
    st.session_state.mc_results['scenario_name'] = scenario_name

if 'mc_results' in st.session_state:
    st.subheader("Monte Carlo Results")
    summary = st.session_state.mc_results['summary']
    runs_annual = st.session_state.mc_results['runs_annual']
    mc_scenario_name = st.session_state.mc_results['scenario_name']

    # Display KPIs
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Median Annual Profit (Y1)", f"${summary['annual_profit_p50_y1']:,.0f}")
    kpi_cols[1].metric("P(Break Even by 12m)", f"{summary['prob_break_even_12m']:.1%}")
    kpi_cols[2].metric("Median Break-Even Month", f"{summary['break_even_month_p50']:.0f} months")

    # Display Charts
    y1_profit_hist = px.histogram(
        runs_annual[runs_annual['Year'] == 1], 
        x="NetProfit",
        title="Distribution of Year 1 Net Profit"
    )
    st.plotly_chart(y1_profit_hist, use_container_width=True)
    
    st.subheader("Annual Results per Run")
    st.dataframe(runs_annual.head(100))

    # Add the download button here
    csv = runs_annual.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Annual Runs as CSV",
        data=csv,
        file_name=f"mc_annual_runs_{mc_scenario_name}.csv",
        mime="text/csv",
    )
