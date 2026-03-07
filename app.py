import streamlit as st
import pandas as pd

# Module Imports
from modules.rfm import calculate_rfm, get_campaign_targets, generate_personal_offer
from modules.sales_analytics import render_sales_analytics, render_subcategory_trends, generate_sales_insights
from modules.mapper import classify_and_extract_data
from modules.smart_insights import generate_dynamic_insights
import BA
import KPI_analyst
import chatbot2


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Cafe_X Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------
# HIDE STREAMLIT DEFAULT UI
# ---------------------------------------------------

hide_ui = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_ui, unsafe_allow_html=True)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>

<h1 style='color:white;'>Cafe_X</h1>
<hr>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------

session_defaults = {
    "files_mapped": False,
    "mapped_data": None,
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None
}

for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------
# SIDEBAR FILE UPLOAD
# ---------------------------------------------------

st.sidebar.title("📁 Upload Data Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or Excel Files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)


# ---------------------------------------------------
# LOAD RAW FILES
# ---------------------------------------------------

raw_dfs = {}

if uploaded_files:

    for i, file in enumerate(uploaded_files):

        ext = file.name.split(".")[-1].lower()

        try:
            if ext == "csv":
                df = pd.read_csv(file, low_memory=False)
            else:
                df = pd.read_excel(file)

            raw_dfs[f"df_{i+1}"] = df
            raw_dfs[f"df_{i+1}_name"] = file.name

        except:
            st.error(f"❌ Error reading {file.name}")


# ---------------------------------------------------
# USER GUIDANCE
# ---------------------------------------------------

if not uploaded_files and not st.session_state.files_mapped:
    st.info("👈 Upload files from the sidebar to begin.")

elif uploaded_files and not st.session_state.files_mapped:
    st.warning("📤 Files uploaded. Please go to **File Mapping tab**.")

elif st.session_state.files_mapped:
    st.success("✅ Files mapped successfully. Explore analytics!")


# ---------------------------------------------------
# LOAD MAPPED DATA
# ---------------------------------------------------

txns_df = st.session_state.txns_df
cust_df = st.session_state.cust_df
prod_df = st.session_state.prod_df
promo_df = st.session_state.promo_df


# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tabs = st.tabs([
    "📘 Instructions",
    "🗂️ File Mapping",
    "📊 Sales Analytics",
    "🔍 Sub Category Analysis",
    "📊 RFM Segmentation",
    "🤖 Business Analyst",
    "🤖 Chatbot"
])


# ===================================================
# TAB 1 INSTRUCTIONS
# ===================================================

with tabs[0]:

    st.subheader("📘 Instructions")

    st.markdown("""
    **Steps to Use**

    1️⃣ Upload data files from sidebar  
    2️⃣ Map the files in File Mapping tab  
    3️⃣ Run analytics modules  

    Supported files:

    • Transactions  
    • Customers  
    • Products  
    • Promotions
    """)


# ===================================================
# TAB 2 FILE MAPPING
# ===================================================

with tabs[1]:

    st.subheader("🗂️ File Mapping")

    if not uploaded_files:
        st.info("Upload files from sidebar.")
        st.stop()

    # RUN MAPPING ONLY ONCE
    if st.session_state.mapped_data is None:

        with st.spinner("🔍 Detecting file structures..."):

            mapped_data = classify_and_extract_data(uploaded_files)

            st.session_state.mapped_data = mapped_data

            st.session_state.txns_df = mapped_data.get("Transactions")
            st.session_state.cust_df = mapped_data.get("Customers")
            st.session_state.prod_df = mapped_data.get("Products")
            st.session_state.promo_df = mapped_data.get("Promotions")

            st.session_state.files_mapped = True

    mapped_data = st.session_state.mapped_data

    txns_df = mapped_data.get("Transactions")
    cust_df = mapped_data.get("Customers")
    prod_df = mapped_data.get("Products")
    promo_df = mapped_data.get("Promotions")

    col1, col2 = st.columns(2)

    # AVAILABLE COLUMNS
    with col1:

        st.markdown("### 📄 Available Columns")

        for name, df in mapped_data.items():

            if df is not None:

                st.markdown(f"**{name} File**")

                for c in df.columns:
                    st.write(c)

                st.markdown("---")

    # MAPPED STRUCTURE
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

    st.success("✅ Mapping Completed")


# ===================================================
# TAB 3 SALES ANALYTICS
# ===================================================

with tabs[2]:

    st.subheader("📊 Sales Analytics")

    if txns_df is None:
        st.warning("Upload Transactions data.")
        st.stop()

    if "run_sales" not in st.session_state:
        st.session_state.run_sales = False

    if not st.session_state.run_sales:

        if st.button("▶️ Start Sales Analytics"):

            st.session_state.run_sales = True
            st.rerun()

    else:

        render_sales_analytics(txns_df)

        st.markdown("---")

        st.subheader("💡 Smart Insights")

        insights = generate_sales_insights(txns_df)

        generate_dynamic_insights(insights)


# ===================================================
# TAB 4 SUBCATEGORY ANALYSIS
# ===================================================

with tabs[3]:

    st.subheader("🔍 Sub Category Analysis")

    if txns_df is None:
        st.warning("Upload transactions data.")
        st.stop()

    if "run_subcat" not in st.session_state:
        st.session_state.run_subcat = False

    if not st.session_state.run_subcat:

        if st.button("▶️ Start Analysis"):

            st.session_state.run_subcat = True
            st.rerun()

    else:

        render_subcategory_trends(txns_df)


# ===================================================
# TAB 5 RFM
# ===================================================

with tabs[4]:

    st.subheader("🚦 RFM Segmentation")

    if txns_df is None:
        st.warning("Upload transactions data.")
        st.stop()

    if st.button("▶️ Run RFM"):

        with st.spinner("Running RFM..."):

            rfm_df = calculate_rfm(txns_df)

            st.session_state.rfm_df = rfm_df

        st.success("RFM Complete")

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


# ===================================================
# TAB 6 BUSINESS ANALYST AI
# ===================================================

with tabs[5]:

    st.subheader("🧠 Business Analyst AI")

    if not raw_dfs:
        st.warning("Upload data files.")
        st.stop()

    BA.run_business_analyst_tab(raw_dfs)

    st.markdown("---")

    KPI_analyst.run_kpi_analyst(raw_dfs)


# ===================================================
# TAB 7 CHATBOT
# ===================================================

with tabs[6]:

    st.subheader("🤖 Data Chatbot")

    if not raw_dfs:
        st.warning("Upload data files.")
        st.stop()

    chatbot2.run_chat(raw_dfs)


# ---------------------------------------------------
# RESET BUTTON
# ---------------------------------------------------

if st.sidebar.button("🔄 Reset App"):

    for k in list(st.session_state.keys()):
        del st.session_state[k]

    st.rerun()
