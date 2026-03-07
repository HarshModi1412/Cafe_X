import streamlit as st
import pandas as pd
from typing import Dict, Any

# Internal Modules
from modules.rfm import calculate_rfm, get_campaign_targets
from modules.sales_analytics import (
    render_sales_analytics,
    render_subcategory_trends,
    generate_sales_insights
)
from modules.mapper import classify_and_extract_data
from modules.smart_insights import generate_dynamic_insights

import BA
import KPI_analyst
import chatbot2


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Cafe_X Dashboard",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# UI CLEANUP
# =====================================================

def hide_streamlit_ui():
    hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_style, unsafe_allow_html=True)


hide_streamlit_ui()


# =====================================================
# HEADER
# =====================================================

def render_header():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; }
        </style>

        <h1 style='color:white;'>Cafe_X</h1>
        <hr>
    """, unsafe_allow_html=True)


render_header()


# =====================================================
# SESSION STATE MANAGEMENT
# =====================================================

DEFAULT_STATE = {
    "files_mapped": False,
    "mapped_data": None,
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None,
    "run_sales": False,
    "run_subcat": False
}


def initialize_session_state():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =====================================================
# FILE LOADING
# =====================================================

def load_uploaded_files(uploaded_files) -> Dict[str, Any]:
    """
    Safely loads uploaded CSV/XLSX files into pandas DataFrames
    """

    raw_data = {}

    for idx, file in enumerate(uploaded_files):

        try:

            ext = file.name.split(".")[-1].lower()

            if ext == "csv":
                df = pd.read_csv(file, low_memory=False)

            elif ext == "xlsx":
                df = pd.read_excel(file)

            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            raw_data[f"df_{idx+1}"] = df
            raw_data[f"df_{idx+1}_name"] = file.name

        except Exception as e:
            st.error(f"Failed to read file {file.name}")
            st.exception(e)

    return raw_data


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📁 Upload Data Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

raw_dfs = load_uploaded_files(uploaded_files) if uploaded_files else {}


# =====================================================
# USER GUIDANCE
# =====================================================

def show_upload_guidance():
    if not uploaded_files and not st.session_state.files_mapped:
        st.info("👈 Upload files from the sidebar to begin.")

    elif uploaded_files and not st.session_state.files_mapped:
        st.warning("📤 Files uploaded. Go to **File Mapping tab**.")

    elif st.session_state.files_mapped:
        st.success("✅ Files mapped successfully. Explore analytics.")


show_upload_guidance()


# =====================================================
# FILE MAPPING LOGIC
# =====================================================

def run_file_mapping(uploaded_files):

    if st.session_state.mapped_data is not None:
        return st.session_state.mapped_data

    with st.spinner("🔍 Detecting dataset structures..."):

        mapped_data = classify_and_extract_data(uploaded_files)

        if not mapped_data or not isinstance(mapped_data, dict):
            st.error("File mapping failed. Please check uploaded files.")
            return None

        st.session_state.mapped_data = mapped_data
        st.session_state.files_mapped = True

        st.session_state.txns_df = mapped_data.get("Transactions")
        st.session_state.cust_df = mapped_data.get("Customers")
        st.session_state.prod_df = mapped_data.get("Products")
        st.session_state.promo_df = mapped_data.get("Promotions")

    return mapped_data


# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "📘 Instructions",
    "🗂️ File Mapping",
    "📊 Sales Analytics",
    "🔍 Sub Category Analysis",
    "📊 RFM Segmentation",
    "🤖 Business Analyst",
    "🤖 Chatbot"
])


# =====================================================
# TAB 1 INSTRUCTIONS
# =====================================================

with tabs[0]:

    st.subheader("📘 Instructions")

    st.markdown("""
    **Steps**

    1️⃣ Upload files  
    2️⃣ Map datasets  
    3️⃣ Run analytics  

    Supported datasets:

    - Transactions  
    - Customers  
    - Products  
    - Promotions
    """)


# =====================================================
# TAB 2 FILE MAPPING
# =====================================================

with tabs[1]:

    st.subheader("🗂️ File Mapping")

    if not uploaded_files:
        st.info("Upload files from sidebar.")
        st.stop()

    mapped_data = run_file_mapping(uploaded_files)

    if mapped_data is None:
        st.stop()

    txns_df = mapped_data.get("Transactions")
    cust_df = mapped_data.get("Customers")
    prod_df = mapped_data.get("Products")
    promo_df = mapped_data.get("Promotions")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📄 Available Columns")

        for name, df in mapped_data.items():

            if df is None:
                continue

            st.markdown(f"**{name}**")

            for col in df.columns:
                st.write(col)

            st.divider()

    with col2:

        st.markdown("### 🔗 Mapped Dataset")

        if txns_df is not None:
            st.write("Transactions:", list(txns_df.columns))

        if cust_df is not None:
            st.write("Customers:", list(cust_df.columns))

        if prod_df is not None:
            st.write("Products:", list(prod_df.columns))

        if promo_df is not None:
            st.write("Promotions:", list(promo_df.columns))

    st.success("Mapping completed successfully")


# =====================================================
# TAB 3 SALES ANALYTICS
# =====================================================

with tabs[2]:

    st.subheader("📊 Sales Analytics")

    txns_df = st.session_state.txns_df

    if txns_df is None:
        st.warning("Upload transactions data.")
        st.stop()

    if not st.session_state.run_sales:

        if st.button("Start Sales Analytics"):
            st.session_state.run_sales = True
            st.rerun()

    else:

        render_sales_analytics(txns_df)

        st.divider()

        insights = generate_sales_insights(txns_df)
        generate_dynamic_insights(insights)


# =====================================================
# TAB 4 SUB CATEGORY ANALYSIS
# =====================================================

with tabs[3]:

    st.subheader("🔍 Sub Category Analysis")

    txns_df = st.session_state.txns_df

    if txns_df is None:
        st.warning("Upload transactions data.")
        st.stop()

    if not st.session_state.run_subcat:

        if st.button("Start Analysis"):
            st.session_state.run_subcat = True
            st.rerun()

    else:

        render_subcategory_trends(txns_df)


# =====================================================
# TAB 5 RFM
# =====================================================

with tabs[4]:

    st.subheader("🚦 RFM Segmentation")

    txns_df = st.session_state.txns_df

    if txns_df is None:
        st.warning("Upload transactions data.")
        st.stop()

    if st.button("Run RFM Analysis"):

        with st.spinner("Running RFM analysis..."):

            rfm_df = calculate_rfm(txns_df)

        st.dataframe(rfm_df.head(10), use_container_width=True)

        st.download_button(
            "Download RFM",
            rfm_df.to_csv(index=False),
            "rfm_output.csv"
        )

        campaign_df = get_campaign_targets(rfm_df)

        if campaign_df is not None:

            st.dataframe(campaign_df.head(10))

            st.download_button(
                "Download Campaign Targets",
                campaign_df.to_csv(index=False),
                "campaign_targets.csv"
            )


# =====================================================
# TAB 6 BUSINESS ANALYST
# =====================================================

with tabs[5]:

    st.subheader("🧠 Business Analyst AI")

    if not raw_dfs:
        st.warning("Upload data files.")
        st.stop()

    BA.run_business_analyst_tab(raw_dfs)

    st.divider()

    KPI_analyst.run_kpi_analyst(raw_dfs)


# =====================================================
# TAB 7 CHATBOT
# =====================================================

with tabs[6]:

    st.subheader("🤖 Data Chatbot")

    if not raw_dfs:
        st.warning("Upload data files.")
        st.stop()

    chatbot2.run_chat(raw_dfs)


# =====================================================
# RESET
# =====================================================

if st.sidebar.button("🔄 Reset App"):

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()
