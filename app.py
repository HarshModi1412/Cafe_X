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

# ---------------- UI ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Cafe_X</h1><hr>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
defaults = {
    "uploaded_files": None,
    "raw_dfs": {},
    "files_mapped": False,
    "txns_df": None,
    "start_sales_analysis": False,
    "start_subcat_analysis": False,
    "run_rfm": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

# ---------------- TAB 1: UPLOAD ----------------
with tabs[0]:
    st.subheader("📁 Upload Data")

    uploaded_files = st.file_uploader(
        "Upload CSV / Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:

        with st.spinner("🔐 Reading your precious data (safe with us)..."):
            raw_dfs = {}

            for i, file in enumerate(uploaded_files):
                ext = file.name.split('.')[-1].lower()
                df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)

                raw_dfs[f"df_{i+1}"] = df
                raw_dfs[f"df_{i+1}_name"] = file.name

        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["raw_dfs"] = raw_dfs

        st.success("✅ Files uploaded successfully")
        st.info("👉 Go to File Mapping tab")

# ---------------- COMMON DATA ----------------
uploaded_files = st.session_state.get("uploaded_files")
raw_dfs = st.session_state.get("raw_dfs", {})
txns_df = st.session_state.get("txns_df")

# ---------------- TAB 2: MAPPING ----------------
with tabs[1]:
    st.subheader("🗂️ File Mapping")

    if uploaded_files:

        mapped_data, confirmed = classify_and_extract_data(uploaded_files)

        if confirmed and mapped_data:

            with st.spinner("💾 Saving mapping..."):

                st.session_state["txns_df"] = mapped_data["Transactions"]
                st.session_state["files_mapped"] = True

            st.success("✅ Mapping completed successfully")
            st.info("👉 Go to Sales Analytics tab")

            # 🔑 Important: rerun so analytics sees updated state
            st.rerun()

        elif st.session_state.get("files_mapped"):

            st.success("✅ Mapping already completed")

            if st.session_state["txns_df"] is not None:
                st.dataframe(st.session_state["txns_df"].head(), use_container_width=True)

    else:
        st.warning("⚠️ Upload files first")

# ---------------- TAB 3: SALES ----------------
with tabs[2]:
    st.subheader("📊 Sales Analytics")

    if txns_df is None:
        st.warning("⚠️ Upload & Map data first")
    else:
        if not st.session_state["start_sales_analysis"]:
            if st.button("▶️ Start Sales Analytics"):
                st.session_state["start_sales_analysis"] = True
                st.rerun()
        else:
            render_sales_analytics(txns_df)
            insights = generate_sales_insights(txns_df)
            generate_dynamic_insights(insights)

# ---------------- TAB 4: SUBCATEGORY ----------------
with tabs[3]:
    st.subheader("🔍 Sub-Category Analysis")

    if txns_df is None:
        st.warning("⚠️ Upload & Map data first")
    else:
        if not st.session_state["start_subcat_analysis"]:
            if st.button("▶️ Start Sub-Category Analysis"):
                st.session_state["start_subcat_analysis"] = True
                st.rerun()
        else:
            render_subcategory_trends(txns_df)

# ---------------- TAB 5: RFM ----------------
with tabs[4]:
    st.subheader("📊 RFM Analysis")

    if txns_df is None:
        st.warning("⚠️ Upload & Map data first")
    else:
        if not st.session_state["run_rfm"]:
            if st.button("▶️ Run RFM Analysis"):
                st.session_state["run_rfm"] = True
                st.rerun()
        else:
            with st.spinner("📊 Doing analysis..."):
                rfm_df = calculate_rfm(txns_df)

            st.dataframe(rfm_df.head(), use_container_width=True)

# ---------------- TAB 6: ANALYST ----------------
with tabs[5]:
    if raw_dfs:
        BA.run_business_analyst_tab(raw_dfs)
        KPI_analyst.run_kpi_analyst(raw_dfs)
    else:
        st.warning("⚠️ Upload files first")

# ---------------- TAB 7: CHATBOT ----------------
with tabs[6]:
    if raw_dfs:
        chatbot2.run_chat(raw_dfs)
    else:
        st.warning("⚠️ Upload files first")

# ---------------- RESET ----------------
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()
