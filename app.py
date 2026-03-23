import streamlit as st
import pandas as pd

# Modules
from modules.rfm import calculate_rfm, get_campaign_targets, generate_personal_offer
from modules.sales_analytics import render_sales_analytics, render_subcategory_trends, generate_sales_insights
from modules.mapper import classify_and_extract_data
from modules.smart_insights import generate_dynamic_insights
import BA
import KPI_analyst
import chatbot2

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cafe_X Dashboard", page_icon="📊", layout="wide")

# ---------------- UI CLEAN ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='margin-bottom:0;'>Cafe_X</h1>
<hr style='margin-top:0;'>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
defaults = {
    "uploaded_files": None,
    "last_uploaded_files": None,
    "raw_dfs": {},
    "files_mapped": False,
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None,
    "start_sales_analysis": False,
    "start_subcat_analysis": False,
    "run_rfm": False,
    "mapping_submitted": False,
    "mapped_data_cache": None# 🔑 ONLY FLAG NEEDED
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- SIDEBAR ----------------
st.sidebar.title("📁 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ---------------- FILE UPLOAD ----------------
if uploaded_files and st.session_state["last_uploaded_files"] != uploaded_files:

    st.session_state["last_uploaded_files"] = uploaded_files

    with st.spinner("🔐 Implementing Auto Mapping..."):
        raw_dfs = {}

        for i, file in enumerate(uploaded_files):
            ext = file.name.split('.')[-1].lower()
            df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)

            raw_dfs[f"df_{i+1}"] = df
            raw_dfs[f"df_{i+1}_name"] = file.name

    st.session_state["raw_dfs"] = raw_dfs
    st.success("✅ Files uploaded")

raw_dfs = st.session_state.get("raw_dfs", {})
txns_df = st.session_state["txns_df"]

# ---------------- TABS ----------------
tabs = st.tabs([
    "📘 Instructions",
    "🗂️ File Mapping",
    "📊 Sales Analytics",
    "🔍 Sub-Category",
    "📊 RFM",
    "🤖 Analyst AI",
    "🤖 Chatbot"
])

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.subheader("Instructions")
    st.markdown("Upload → Map → Analyze")

# ---------------- TAB 2 (FINAL FIX) ----------------
with tabs[1]:
    st.subheader("File Mapping")

    if uploaded_files:

        mapped_data, confirmed = classify_and_extract_data(uploaded_files)

        if confirmed:

            with st.spinner("💾 Saving mapping..."):

                st.session_state["txns_df"] = mapped_data.get("Transactions")
                st.session_state["cust_df"] = mapped_data.get("Customers")
                st.session_state["prod_df"] = mapped_data.get("Products")
                st.session_state["promo_df"] = mapped_data.get("Promotions")

                st.session_state["files_mapped"] = True

            st.success("✅ Mapping completed successfully")
            st.info("👉 You can now proceed to Analytics tabs")

            # 🔑 CRITICAL FIX: force fresh rerun with updated state
            st.rerun()

        elif st.session_state.get("files_mapped", False):

            st.success("✅ Mapping already completed")

            if st.session_state["txns_df"] is not None:
                st.dataframe(
                    st.session_state["txns_df"].head(),
                    width="stretch"
                )

    else:
        st.info("Upload files first")

# ---------------- TAB 3 ----------------
with tabs[2]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if not st.session_state["start_sales_analysis"]:
            if st.button("▶️ Start Sales Analytics"):
                with st.spinner("📊 Doing analysis..."):
                    st.session_state["start_sales_analysis"] = True
                st.rerun()
        else:
            render_sales_analytics(txns_df)
            insights = generate_sales_insights(txns_df)
            generate_dynamic_insights(insights)

# ---------------- TAB 4 ----------------
with tabs[3]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if not st.session_state["start_subcat_analysis"]:
            if st.button("▶️ Start Sub-Category Analysis"):
                with st.spinner("📊 Doing analysis..."):
                    st.session_state["start_subcat_analysis"] = True
                st.rerun()
        else:
            render_subcategory_trends(txns_df)

# ---------------- TAB 5 ----------------
with tabs[4]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if not st.session_state["run_rfm"]:
            if st.button("▶️ Run RFM Analysis"):
                with st.spinner("📊 Doing analysis..."):
                    st.session_state["run_rfm"] = True
                st.rerun()
        else:
            with st.spinner("📊 Doing analysis..."):
                rfm_df = calculate_rfm(txns_df)

            st.dataframe(rfm_df.head(), width="stretch")

# ---------------- TAB 6 ----------------
with tabs[5]:
    if raw_dfs:
        BA.run_business_analyst_tab(raw_dfs)
        KPI_analyst.run_kpi_analyst(raw_dfs)
    else:
        st.warning("Upload files")

# ---------------- TAB 7 ----------------
with tabs[6]:
    if raw_dfs:
        chatbot2.run_chat(raw_dfs)
    else:
        st.warning("Upload files")

# ---------------- RESET ----------------
if st.sidebar.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()
