import streamlit as st
import pandas as pd

# Modules
from modules.rfm import calculate_rfm
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

st.markdown("<h1 style='margin-bottom:0;'>Cafe_X</h1><hr>", unsafe_allow_html=True)

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
    "page": "📘 Instructions"
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

# ---------------- FILE HANDLING ----------------
if uploaded_files and st.session_state["last_uploaded_files"] != uploaded_files:

    st.session_state["last_uploaded_files"] = uploaded_files

    with st.spinner("🔐 Processing files..."):
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

# ---------------- NAVIGATION ----------------
pages = [
    "📘 Instructions",
    "🗂️ File Mapping",
    "📊 Sales Analytics",
    "🔍 Sub-Category",
    "📊 RFM",
    "🤖 Analyst AI",
    "🤖 Chatbot"
]

selected_page = st.sidebar.radio(
    "📌 Navigation",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected_page

# ---------------- PAGE ROUTING ----------------

# 📘 Instructions
if selected_page == "📘 Instructions":
    st.subheader("Instructions")
    st.markdown("Upload → Map → Analyze")

# 🗂️ File Mapping
elif selected_page == "🗂️ File Mapping":
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

        elif st.session_state["files_mapped"]:
            st.success("✅ Mapping already completed")

            if st.session_state["txns_df"] is not None:
                st.dataframe(st.session_state["txns_df"].head(), use_container_width=True)

    else:
        st.info("Upload files first")

# 📊 Sales Analytics
elif selected_page == "📊 Sales Analytics":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        render_sales_analytics(txns_df)
        insights = generate_sales_insights(txns_df)
        generate_dynamic_insights(insights)

# 🔍 Sub-Category
elif selected_page == "🔍 Sub-Category":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        render_subcategory_trends(txns_df)

# 📊 RFM
elif selected_page == "📊 RFM":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        rfm_df = calculate_rfm(txns_df)
        st.dataframe(rfm_df, use_container_width=True)

# 🤖 Analyst AI
elif selected_page == "🤖 Analyst AI":
    if raw_dfs:
        BA.run_business_analyst_tab(raw_dfs)
        KPI_analyst.run_kpi_analyst(raw_dfs)
    else:
        st.warning("Upload files")

# 🤖 Chatbot
elif selected_page == "🤖 Chatbot":
    if raw_dfs:
        chatbot2.run_chat(raw_dfs)
    else:
        st.warning("Upload files")

# ---------------- RESET ----------------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()
