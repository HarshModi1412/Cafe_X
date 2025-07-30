import streamlit as st
import pandas as pd
import numpy as np
import json
import chardet
import requests
import re
from io import BytesIO
import plotly.express as px

# --- Gemini API Setup ---
GEMINI_API_KEY = "AIzaSyD9DfnqPz7vMgh5aUHaMAVjeJbg20VZMvU"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Gemini API Call ---
def ask_llm(prompt):
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(GEMINI_URL, headers=headers, json=body)
        res.raise_for_status()
        result = res.json()
        if "candidates" not in result:
            return "No result"
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        st.error(f"❌ Gemini Error: {e}")
        return "Error"

# --- JSON extraction from LLM response ---
def extract_json_from_text(text):
    try:
        match = re.search(r"``````", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()
        if text.strip().startswith("[") or text.strip().startswith("{"):
            return text.strip()
    except Exception as e:
        st.warning(f"⚠️ extract_json_from_text error: {e}")
    return ""

# --- Load file ---
def load_file(file):
    raw = file.read()
    if file.name.endswith(".csv"):
        encoding = chardet.detect(raw)["encoding"] or "utf-8"
        return pd.read_csv(BytesIO(raw), encoding=encoding)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(raw), engine="openpyxl")
    else:
        st.error("Unsupported format. Please upload a CSV or Excel file.")
        return pd.DataFrame()

# --- Fuzzy column matching ---
def fuzzy_match(col, candidates):
    col = col.lower().replace(" ", "")
    for candidate in candidates:
        if col == candidate.lower().replace(" ", ""):
            return candidate
    # Try partial match
    for candidate in candidates:
        if col in candidate.lower().replace(" ", "") or candidate.lower().replace(" ", "") in col:
            return candidate
    return None

# --- KPI Benchmarks for Indian Cafe Segments ---
BENCHMARKS = {
    "Very Small": {
        "RevPASH": (40, 80),  # INR
        "Average Check Size": (80, 150),  # INR
        "Table Turnover Rate": (2, 4),  # per day
        "Net Profit Margin": (5, 10),  # %
        "Gross Margin": (55, 65),  # %
        "Peak Hour Sales": (2000, 6000),  # INR
        "Repeat Customer Rate": (15, 30),  # %
        "Sales per Sqft": (3000, 6000),  # INR per year
    },
    "Small": {
        "RevPASH": (70, 120),
        "Average Check Size": (120, 180),
        "Table Turnover Rate": (3, 6),
        "Net Profit Margin": (7, 12),
        "Gross Margin": (58, 68),
        "Peak Hour Sales": (5000, 15000),
        "Repeat Customer Rate": (20, 35),
        "Sales per Sqft": (5000, 10000),
    },
    "Medium": {
        "RevPASH": (110, 180),
        "Average Check Size": (150, 220),
        "Table Turnover Rate": (5, 8),
        "Net Profit Margin": (10, 15),
        "Gross Margin": (62, 70),
        "Peak Hour Sales": (12000, 30000),
        "Repeat Customer Rate": (25, 45),
        "Sales per Sqft": (8000, 15000),
    },
    "Large": {
        "RevPASH": (170, 300),
        "Average Check Size": (200, 300),
        "Table Turnover Rate": (7, 12),
        "Net Profit Margin": (12, 20),
        "Gross Margin": (65, 72),
        "Peak Hour Sales": (25000, 65000),
        "Repeat Customer Rate": (30, 55),
        "Sales per Sqft": (12000, 25000),
    },
    "Very Large": {
        "RevPASH": (240, 450),
        "Average Check Size": (250, 400),
        "Table Turnover Rate": (10, 16),
        "Net Profit Margin": (15, 25),
        "Gross Margin": (68, 75),
        "Peak Hour Sales": (50000, 150000),
        "Repeat Customer Rate": (35, 60),
        "Sales per Sqft": (20000, 40000),
    }
}

# --- Helper: Check if KPI within benchmark range ---
def check_benchmark(kpi_name, value, scale):
    if scale not in BENCHMARKS or kpi_name not in BENCHMARKS[scale]:
        return "N/A", "No benchmark"
    low, high = BENCHMARKS[scale][kpi_name]
    if value == "❌" or value is None:
        return "Error", "Data missing or error"
    if value < low:
        return "Low", f"Below ideal range ({low}-{high})"
    elif value > high:
        return "High", f"Above typical range ({low}-{high})"
    else:
        return "Good", f"Within target range ({low}-{high})"

# --- Advice text per KPI and status ---
ADVICE_TEXT = {
    "RevPASH": {
        "Low": "Consider maximizing seat utilization during open hours; explore promotions during off-peak hours.",
        "High": "Excellent seat utilization! Maintain service quality to keep customers coming.",
        "Good": "RevPASH is healthy; keep monitoring for seasonal changes.",
        "Error": "Missing data for calculation; please provide all required inputs."
    },
    "Average Check Size": {
        "Low": "Upsell combos or premium items to increase average spend per customer.",
        "High": "Great average spend; consider loyalty programs to retain high-value customers.",
        "Good": "Average check size is in good range.",
        "Error": "Missing data for average check size."
    },
    "Table Turnover Rate": {
        "Low": "Try to optimize seating and speed of service to serve more customers.",
        "High": "High turnover may strain staff—ensure quality and customer experience remain excellent.",
        "Good": "Table turnover rates look healthy.",
        "Error": "Insufficient data for table turnover."
    },
    "Net Profit Margin": {
        "Low": "Review costs and pricing; reduce wastage and control labor expenses.",
        "High": "Strong profitability! Keep optimizing operations.",
        "Good": "Profit margin is within healthy range.",
        "Error": "Net profit margin calculation needs more data."
    },
    "Gross Margin": {
        "Low": "Optimize supplier costs or increase menu prices for low-margin items.",
        "High": "Good margin efficiency; watch quality and pricing balance.",
        "Good": "Gross margin is healthy.",
        "Error": "Missing ingredient cost or revenue data."
    },
    "Peak Hour Sales": {
        "Low": "Increase marketing during peak times or improve service efficiency to boost sales.",
        "High": "Great peak sales performance!",
        "Good": "Peak hour sales are good.",
        "Error": "Insufficient data for peak hour sales."
    },
    "Repeat Customer Rate": {
        "Low": "Implement loyalty programs and personalized marketing to improve retention.",
        "High": "Strong repeat customer base; leverage referrals.",
        "Good": "Repeat customer rate is positive.",
        "Error": "No data on repeat customers."
    },
    "Sales per Sqft": {
        "Low": "Optimize space usage or increase foot traffic with promotions.",
        "High": "Excellent utilization of retail space.",
        "Good": "Sales per square foot are in good range.",
        "Error": "Provide floor area and sales data to calculate."
    },
}

# --- KPI Calculations ---
def calc_revpash(total_revenue, num_seats, open_hours):
    try:
        return round(total_revenue / (num_seats * open_hours), 2)
    except:
        return "❌"

def calc_average_check_size(total_revenue, num_customers):
    try:
        return round(total_revenue / num_customers, 2)
    except:
        return "❌"

def calc_table_turnover(total_parties, num_tables):
    try:
        return round(total_parties / num_tables, 2)
    except:
        return "❌"

def calc_net_profit_margin(net_profit, total_revenue):
    try:
        return round((net_profit / total_revenue) * 100, 2)
    except:
        return "❌"

def calc_gross_margin(total_revenue, cogs):
    try:
        return round(((total_revenue - cogs) / total_revenue) * 100, 2)
    except:
        return "❌"

def calc_peak_hour_sales(df, time_col, revenue_col, peak_hours):
    try:
        # Filter data by peak hours (assumed hours as integers or HH:MM format)
        df_times = pd.to_datetime(df[time_col], errors='coerce')
        if df_times.isnull().all():
            return "❌"
        filtered = df[df_times.dt.hour.isin(peak_hours)]
        return round(filtered[revenue_col].sum(), 2)
    except:
        return "❌"

def calc_repeat_customer_rate(num_returning_customers, total_customers):
    try:
        return round((num_returning_customers / total_customers) * 100, 2)
    except:
        return "❌"

def calc_sales_per_sqft(total_revenue, sqft_area):
    try:
        return round(total_revenue / sqft_area, 2)
    except:
        return "❌"

# --- Main function ---
def run_kpi_analyst():
    st.set_page_config(page_title="📊 Indian Cafe KPI Analyst", layout="wide")
    st.title("📊 Indian Cafe KPI Analyst with Benchmarking & AI Help")

    st.markdown("""
    Upload your sales dataset and provide business details.  
    This app calculates key cafe KPIs, compares them to Indian benchmarks by scale, and offers improvement advice.
    """)

    file = st.file_uploader("Upload your CSV or Excel sales dataset", type=["csv", "xlsx"])

    col1, col2, col3 = st.columns(3)
    with col1:
        industry = st.text_input("Industry", value="Cafe", disabled=True)
    with col2:
        scale = st.selectbox(
            "Select Business Scale",
            ["Very Small", "Small", "Medium", "Large", "Very Large"],
            help="Choose the scale based on your annual revenue."
        )
    with col3:
        goal = st.text_area("Business Goal or Problem Statement", help="Example: Increase profitability and improve customer retention.")

    if not file:
        st.info("Please upload your sales dataset to proceed.")
        return

    if not goal:
        st.info("Please enter your business goal/problem statement.")
        return

    df = load_file(file)
    if df.empty:
        return

    st.subheader("🔍 Data Preview (first 10 rows)")
    st.dataframe(df.head(10))

    # Determine columns (prompt user to specify column mappings for flexibility)
    st.sidebar.header("Map your dataset columns")

    col_total_revenue = st.sidebar.text_input("Total Revenue Column", value=fuzzy_match("total revenue", df.columns) or "")
    col_customers = st.sidebar.text_input("Number of Customers Column", value=fuzzy_match("customers", df.columns) or "")
    col_num_seats = st.sidebar.text_input("Number of Seats Column (if in data)", value=fuzzy_match("seats", df.columns) or "")
    col_open_hours = st.sidebar.text_input("Opening Hours Column (if in data)", value=fuzzy_match("open hours", df.columns) or "")
    col_num_tables = st.sidebar.text_input("Number of Tables Column (if in data)", value=fuzzy_match("tables", df.columns) or "")
    col_net_profit = st.sidebar.text_input("Net Profit Column", value=fuzzy_match("net profit", df.columns) or "")
    col_cogs = st.sidebar.text_input("COGS (Cost of Goods Sold) Column", value=fuzzy_match("cogs", df.columns) or "")
    col_order_time = st.sidebar.text_input("Order Time Column", value=fuzzy_match("time|order time|datetime", df.columns) or "")
    col_revenue_per_order = st.sidebar.text_input("Revenue per Order Column", value=fuzzy_match("revenue|sales", df.columns) or "")
    col_customer_id = st.sidebar.text_input("Customer ID Column", value=fuzzy_match("customer id|customer", df.columns) or "")

    # Collect missing inputs if necessary
    st.sidebar.header("Additional Inputs")
    num_seats_input = None
    open_hours_input = None
    num_tables_input = None
    total_customers_input = None
    num_parties_input = None
    net_profit_input = None
    cogs_input = None
    sqft_area_input = None
    num_returning_customers_input = None

    # Helper to parse numeric columns
    def sum_column(col_name):
        try:
            return df[col_name].sum()
        except:
            return None

    # Total Revenue - required for nearly all
    total_revenue_val = None
    if col_total_revenue in df.columns:
        total_revenue_val = sum_column(col_total_revenue)
    else:
        st.sidebar.warning("Total Revenue column not found in dataset.")

    # Number of Customers
    num_customers_val = None
    if col_customers in df.columns:
        num_customers_val = sum_column(col_customers)
    else:
        num_customers_val = st.sidebar.number_input("Enter total number of customers (period)", min_value=0, step=1)

    # Number of Seats
    if col_num_seats in df.columns:
        seats_val = df[col_num_seats].iloc[0] if not df[col_num_seats].empty else None
    else:
        seats_val = st.sidebar.number_input("Enter number of seats in cafe", min_value=1, step=1)

    # Opening Hours
    if col_open_hours in df.columns:
        open_hours_val = df[col_open_hours].iloc[0] if not df[col_open_hours].empty else None
    else:
        open_hours_val = st.sidebar.number_input("Enter cafe opening hours per day", min_value=1, max_value=24, step=1)

    # Number of Tables
    if col_num_tables in df.columns:
        tables_val = df[col_num_tables].iloc[0] if not df[col_num_tables].empty else None
    else:
        tables_val = st.sidebar.number_input("Enter number of tables", min_value=1, step=1)

    # Net Profit
    if col_net_profit in df.columns:
        net_profit_val = sum_column(col_net_profit)
    else:
        net_profit_val = st.sidebar.number_input("Enter net profit for period (INR)", value=0.0, step=1000.0)

    # COGS
    if col_cogs in df.columns:
        cogs_val = sum_column(col_cogs)
    else:
        cogs_val = st.sidebar.number_input("Enter total COGS for period (INR)", value=0.0, step=1000.0)

    # Floor Area (in sqft)
    sqft_area_val = st.sidebar.number_input("Enter cafe area in sqft", min_value=1, step=10)

    # Total Parties Served (for table turnover) - ask total parties if not found
    parties_served_val = st.sidebar.number_input("Enter total parties served (for table turnover)", min_value=1, step=1)

    # Repeat Customers count (if not inferable from data)
    num_returning_customers_val = st.sidebar.number_input("Enter number of returning customers", min_value=0, step=1)

    # Calculate KPIs
    kp_results = {}

    # RevPASH
    kp_results["RevPASH"] = calc_revpash(total_revenue_val, seats_val, open_hours_val)

    # Average Check Size
    kp_results["Average Check Size"] = calc_average_check_size(total_revenue_val, num_customers_val)

    # Table Turnover Rate
    kp_results["Table Turnover Rate"] = calc_table_turnover(parties_served_val, tables_val)

    # Net Profit Margin
    kp_results["Net Profit Margin"] = calc_net_profit_margin(net_profit_val, total_revenue_val)

    # Gross Margin
    kp_results["Gross Margin"] = calc_gross_margin(total_revenue_val, cogs_val)

    # Peak Hour Sales (assuming peak hours = 11 AM to 2 PM and 5 PM to 8 PM)
    if col_order_time and col_revenue_per_order in df.columns:
        peak_hours = list(range(11, 14)) + list(range(17, 20))
        peak_sales_val = calc_peak_hour_sales(df, col_order_time, col_revenue_per_order, peak_hours)
    else:
        peak_sales_val = st.sidebar.number_input("Enter estimated peak hour sales (INR)", min_value=0, step=100)

    kp_results["Peak Hour Sales"] = peak_sales_val

    # Repeat Customer Rate
    kp_results["Repeat Customer Rate"] = calc_repeat_customer_rate(num_returning_customers_val, num_customers_val)

    # Sales per sqft
    kp_results["Sales per Sqft"] = calc_sales_per_sqft(total_revenue_val, sqft_area_val * 12)  # Monthly to Annual Approximation

    # Show KPIs and bench comparison
    st.subheader("✅ Calculated KPIs with Benchmarks & Advice")

    kpi_data = []
    for kpi, val in kp_results.items():
        status, bench_msg = check_benchmark(kpi, val, scale)
        advice = ADVICE_TEXT[kpi].get(status, "No advice available.")
        kpi_data.append({
            "KPI": kpi,
            "Value": val,
            "Status": status,
            "Benchmark Range": bench_msg,
            "Advice": advice
        })

    kpi_df = pd.DataFrame(kpi_data)
    st.dataframe(kpi_df)

    # Plot KPI comparison for numeric values only
    try:
        df_plot = pd.DataFrame([
            {"KPI": r["KPI"], "Type": "Company", "Value": r["Value"]} for r in kpi_data if isinstance(r["Value"], (int, float, np.float64))
        ] + [
            {
                "KPI": r["KPI"],
                "Type": "Benchmark Low",
                "Value": BENCHMARKS[scale][r["KPI"]][0] if scale in BENCHMARKS and r["KPI"] in BENCHMARKS[scale] else None
            } for r in kpi_data if isinstance(r["Value"], (int, float, np.float64))
        ] + [
            {
                "KPI": r["KPI"],
                "Type": "Benchmark High",
                "Value": BENCHMARKS[scale][r["KPI"]][1] if scale in BENCHMARKS and r["KPI"] in BENCHMARKS[scale] else None
            } for r in kpi_data if isinstance(r["Value"], (int, float, np.float64))
        ])
        fig = px.bar(df_plot, x="KPI", y="Value", color="Type", barmode="group", title=f"KPI Values vs Benchmark for {scale} Cafe Scale")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate plot: {e}")

    # --- AI Chat for follow-up questions ---
    st.subheader("💬 Ask AI Consultant about your KPIs")
    user_question = st.text_area("Ask any question about your business KPIs or how to improve:")

    if user_question:
        # Build contextual prompt including KPI values and user question
        prompt = f"""
You are a helpful cafe business consultant. Based on following KPI values for an Indian cafe business of scale '{scale}':
{json.dumps(kp_results, indent=2)}

User question: {user_question}

Provide detailed, actionable advice addressing the question, using the KPI data above.
"""
        st.markdown("**Consultant is thinking...**")
        ai_response = ask_llm(prompt)
        st.markdown(f"**Answer:** {ai_response}")

if __name__ == "__main__":
    run_kpi_analyst()
