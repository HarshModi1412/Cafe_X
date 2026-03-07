import streamlit as st
import pandas as pd
from typing import Dict

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


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Cafe_X Dashboard",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------------------------
# HIDE STREAMLIT UI
# ----------------------------------------------------

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown("""
<h1 style='color:white;'>Cafe_X</h1>
<hr>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------

DEFAULT_STATE = {
    "mapped_data": None,
    "files_mapped": False,
    "manual_mapping": {},
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None
}

for k,v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ----------------------------------------------------
# FILE LOADING
# ----------------------------------------------------

def load_files(files):

    raw = {}

    for i,f in enumerate(files):

        try:

            if f.name.endswith("csv"):
                df = pd.read_csv(f, low_memory=False)

            else:
                df = pd.read_excel(f)

            raw[f"df_{i+1}"] = df
            raw[f"df_{i+1}_name"] = f.name

        except Exception as e:
            st.error(f"Error reading {f.name}")
            st.exception(e)

    return raw


# ----------------------------------------------------
# SIDEBAR UPLOAD
# ----------------------------------------------------

st.sidebar.title("Upload Data")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx"],
    accept_multiple_files=True
)

raw_dfs = load_files(uploaded_files) if uploaded_files else {}


# ----------------------------------------------------
# AUTO MAPPING (FAST)
# ----------------------------------------------------

@st.cache_data(show_spinner=False)
def auto_map(files):
    return classify_and_extract_data(files)


def run_mapping(files):

    if st.session_state.mapped_data is not None:
        return st.session_state.mapped_data

    with st.spinner("Running automatic mapping..."):

        mapped = auto_map(files)

        if not mapped or not isinstance(mapped,dict):
            st.error("Automatic mapping failed")
            return None

        st.session_state.mapped_data = mapped
        st.session_state.files_mapped = True

        st.session_state.txns_df = mapped.get("Transactions")
        st.session_state.cust_df = mapped.get("Customers")
        st.session_state.prod_df = mapped.get("Products")
        st.session_state.promo_df = mapped.get("Promotions")

    return mapped


# ----------------------------------------------------
# SIDEBAR MAPPING VIEW
# ----------------------------------------------------

def show_mapping_sidebar(mapped):

    st.sidebar.markdown("### Mapping Status")

    mapped_cols=set()
    all_cols=set()

    for df in mapped.values():

        if df is None:
            continue

        cols=list(df.columns)

        mapped_cols.update(cols)
        all_cols.update(cols)

    unmapped=list(all_cols-mapped_cols)

    st.sidebar.write("Mapped Columns")
    st.sidebar.write(list(mapped_cols))

    st.sidebar.write("Unmapped Columns")
    st.sidebar.write(unmapped)


# ----------------------------------------------------
# TABS
# ----------------------------------------------------

tabs = st.tabs([ "📘 Instructions", "🗂️ File Mapping", "📊 Sales Analytics", "🔍 Sub Category Analysis", "📊 RFM Segmentation", "🤖 Business Analyst", "🤖 Chatbot" ])


# ----------------------------------------------------
# TAB 1
# ----------------------------------------------------

with tabs[0]:

    st.markdown("""
### How to Use

1 Upload files  
2 Automatic mapping runs  
3 Review mapped/unmapped columns  
4 Fix mapping manually if needed  
5 Run analytics
""")


# ----------------------------------------------------
# TAB 2 MAPPING
# ----------------------------------------------------

with tabs[1]:

    st.subheader("File Mapping")

    if not uploaded_files:
        st.info("Upload files first")
        st.stop()

    mapped=run_mapping(uploaded_files)

    if mapped is None:
        st.stop()

    show_mapping_sidebar(mapped)

    txns_df=mapped.get("Transactions")

    col1,col2=st.columns(2)

    with col1:

        st.markdown("### Available Columns")

        for name,df in mapped.items():

            if df is None:
                continue

            st.write(f"**{name}**")

            for c in df.columns:
                st.write(c)

            st.divider()

    with col2:

        st.markdown("### Auto Mapping")

        for name,df in mapped.items():

            if df is None:
                continue

            st.write(f"{name} → {list(df.columns)}")


# ----------------------------------------------------
# MANUAL MAPPING
# ----------------------------------------------------

    st.markdown("### Manual Mapping (Optional)")

    if txns_df is not None:

        all_cols=list(txns_df.columns)

        required_fields=[
            "customer_id",
            "transaction_date",
            "product_id",
            "sales",
            "quantity"
        ]

        mapped_cols=[]

        for field in required_fields:

            col=st.selectbox(
                f"Map {field}",
                ["None"]+[c for c in all_cols if c not in mapped_cols],
                key=field
            )

            if col!="None":
                mapped_cols.append(col)

                st.session_state.manual_mapping[field]=col


# ----------------------------------------------------
# TAB 3 SALES
# ----------------------------------------------------

with tabs[2]:

    txns_df=st.session_state.txns_df

    if txns_df is None:
        st.warning("Transactions data required")
        st.stop()

    render_sales_analytics(txns_df)

    st.divider()

    insights=generate_sales_insights(txns_df)
    generate_dynamic_insights(insights)


# ----------------------------------------------------
# TAB 4 SUBCATEGORY
# ----------------------------------------------------

with tabs[3]:

    txns_df=st.session_state.txns_df

    if txns_df is None:
        st.warning("Transactions data required")
        st.stop()

    render_subcategory_trends(txns_df)


# ----------------------------------------------------
# TAB 5 RFM
# ----------------------------------------------------

with tabs[4]:

    txns_df=st.session_state.txns_df

    if txns_df is None:
        st.warning("Transactions data required")
        st.stop()

    if st.button("Run RFM"):

        with st.spinner("Running segmentation"):

            rfm=calculate_rfm(txns_df)

        st.dataframe(rfm.head())

        st.download_button(
            "Download RFM",
            rfm.to_csv(index=False),
            "rfm.csv"
        )

        camp=get_campaign_targets(rfm)

        if camp is not None:

            st.dataframe(camp.head())

            st.download_button(
                "Download Campaign",
                camp.to_csv(index=False),
                "campaign.csv"
            )


# ----------------------------------------------------
# TAB 6 BUSINESS ANALYST
# ----------------------------------------------------

with tabs[5]:

    if not raw_dfs:
        st.warning("Upload data files")
        st.stop()

    BA.run_business_analyst_tab(raw_dfs)

    st.divider()

    KPI_analyst.run_kpi_analyst(raw_dfs)


# ----------------------------------------------------
# TAB 7 CHATBOT
# ----------------------------------------------------

with tabs[6]:

    if not raw_dfs:
        st.warning("Upload files")
        st.stop()

    chatbot2.run_chat(raw_dfs)


# ----------------------------------------------------
# RESET
# ----------------------------------------------------

if st.sidebar.button("Reset App"):

    for k in list(st.session_state.keys()):
        del st.session_state[k]

    st.rerun()

