import streamlit as st
import pandas as pd

# Modules
from modules.rfm import calculate_rfm, get_campaign_targets, generate_personal_offer
from modules.profiler import generate_customer_profile
from modules.customer_journey import map_customer_journey_and_affinity, generate_behavioral_recommendation_with_impact
from modules.discount import generate_discount_insights, assign_offer_codes
from modules.personalization import compute_customer_preferences
from modules.sales_analytics import render_sales_analytics, render_subcategory_trends, generate_sales_insights
from modules.mapper import classify_and_extract_data
from modules.smart_insights import generate_dynamic_insights
import BA
import KPI_analyst
import chatbot2

# ---------------- UI CLEANUP ----------------
st.set_page_config(page_title="Cafe_X Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:white;'>Cafe_X</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
defaults = {
    "uploaded_files": None,
    "files_mapped": False,
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None,
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

# ---------------- FILE LOADING ----------------
raw_dfs = {}

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

    with st.spinner("📤 Uploading and processing files..."):
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):
            ext = file.name.split('.')[-1].lower()

            try:
                if ext == "csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                raw_dfs[f"df_{i+1}"] = df
                raw_dfs[f"df_{i+1}_name"] = file.name

            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

            progress.progress((i + 1) / len(uploaded_files))

    st.success("✅ Files uploaded successfully!")

# ---------------- STATUS ----------------
if not uploaded_files and not st.session_state["files_mapped"]:
    st.info("👈 Upload files from sidebar")
elif uploaded_files and not st.session_state["files_mapped"]:
    st.warning("📤 Files uploaded. Go to File Mapping tab")
elif st.session_state["files_mapped"]:
    st.success("✅ Files mapped. Ready to use")

# ---------------- LOAD SESSION DATA ----------------
txns_df = st.session_state["txns_df"]
cust_df = st.session_state["cust_df"]
prod_df = st.session_state["prod_df"]
promo_df = st.session_state["promo_df"]

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

# ---------------- TAB 2 (MAPPING) ----------------
with tabs[1]:
    st.subheader("File Mapping")

    if uploaded_files:

        if not st.session_state["files_mapped"]:

            with st.spinner("🧠 Mapping columns..."):
                mapped_data = classify_and_extract_data(uploaded_files)

            if mapped_data:
                with st.spinner("💾 Saving mapping..."):
                    st.session_state["txns_df"] = mapped_data.get("Transactions")
                    st.session_state["cust_df"] = mapped_data.get("Customers")
                    st.session_state["prod_df"] = mapped_data.get("Products")
                    st.session_state["promo_df"] = mapped_data.get("Promotions")
                    st.session_state["files_mapped"] = True

                st.success("✅ Mapping complete!")
                st.rerun()

        else:
            st.dataframe(txns_df.head() if txns_df is not None else "No Transactions")

    else:
        st.info("Upload files first")

# ---------------- TAB 3 ----------------
with tabs[2]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if st.button("▶ Start Analysis"):
            with st.spinner("Analyzing..."):
                render_sales_analytics(txns_df)

# ---------------- TAB 4 ----------------
with tabs[3]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if st.button("▶ Run Drilldown"):
            with st.spinner("Processing..."):
                render_subcategory_trends(txns_df)

# ---------------- TAB 5 ----------------
with tabs[4]:
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        if st.button("▶ Run RFM"):
            with st.spinner("Running RFM..."):
                rfm = calculate_rfm(txns_df)
                st.dataframe(rfm.head())

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
