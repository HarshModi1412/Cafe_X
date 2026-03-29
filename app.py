import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# =========================
# CONFIG FILES
# =========================
USERS_FILE = "user.csv"
USAGE_FILE = "usage_logs.csv"

MAX_USAGE = 5
WINDOW_HOURS = 5

# =========================
# LOGIN SYSTEM
# =========================
def load_users():
    if not os.path.exists(USERS_FILE):
        st.error("❌ users.csv not found")
        return {}

    df = pd.read_csv(USERS_FILE)

    if "email" not in df.columns or "password" not in df.columns:
        st.error("❌ users.csv must have 'email' and 'password'")
        return {}

    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["password"] = df["password"].astype(str).str.strip()

    return dict(zip(df["email"], df["password"]))


def login():
    st.set_page_config(page_title="Login", layout="centered")

    st.title("🔐 Cafe_X Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    users = load_users()

    if st.button("Login"):
        email_clean = email.strip().lower()

        if email_clean in users and users[email_clean] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = email_clean
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")


# =========================
# RATE LIMIT (CSV BASED)
# =========================
def init_usage_file():
    if not os.path.exists(USAGE_FILE):
        df = pd.DataFrame(columns=["email", "feature", "timestamp"])
        df.to_csv(USAGE_FILE, index=False)


def check_usage_limit(email, feature):
    init_usage_file()

    df = pd.read_csv(USAGE_FILE)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    now = datetime.now()

    user_logs = df[
        (df["email"] == email) &
        (df["feature"] == feature)
    ]

    # last 5 hours only
    user_logs = user_logs[
        user_logs["timestamp"] > (now - timedelta(hours=WINDOW_HOURS))
    ]

    if len(user_logs) >= MAX_USAGE:
        oldest = user_logs["timestamp"].min()
        remaining = timedelta(hours=WINDOW_HOURS) - (now - oldest)
        mins = int(remaining.total_seconds() // 60)

        st.error(f"❌ Limit reached. Try again in ~{mins} mins.")
        return False

    # log usage
    new_row = pd.DataFrame([{
        "email": email,
        "feature": feature,
        "timestamp": now
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(USAGE_FILE, index=False)

    return True


def get_remaining_usage(email, feature):
    init_usage_file()

    df = pd.read_csv(USAGE_FILE)

    if df.empty:
        return MAX_USAGE

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    now = datetime.now()

    user_logs = df[
        (df["email"] == email) &
        (df["feature"] == feature)
    ]

    user_logs = user_logs[
        user_logs["timestamp"] > (now - timedelta(hours=WINDOW_HOURS))
    ]

    return MAX_USAGE - len(user_logs)


# =========================
# SESSION CHECK
# =========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()


# =========================
# IMPORT MODULES
# =========================
from modules.rfm import calculate_rfm
from modules.sales_analytics import render_sales_analytics, render_subcategory_trends, generate_sales_insights
from modules.mapper import classify_and_extract_data
from modules.smart_insights import generate_dynamic_insights
import BA
import KPI_analyst
import chatbot2

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Cafe_X Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Cafe_X</h1><hr>", unsafe_allow_html=True)

# =========================
# SESSION DEFAULTS
# =========================
defaults = {
    "raw_dfs": {},
    "last_uploaded_files": None,
    "files_mapped": False,
    "txns_df": None,
    "cust_df": None,
    "prod_df": None,
    "promo_df": None,
    "page": "📘 Instructions"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# SIDEBAR
# =========================
user_email = st.session_state["user"]

st.sidebar.success(f"👤 {user_email}")
st.sidebar.title("📁 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# Usage display
st.sidebar.markdown("### ⚡ Usage Limits")
st.sidebar.info(f"🤖 Chatbot left: {get_remaining_usage(user_email, 'chatbot')}")
st.sidebar.info(f"📊 Analyst AI left: {get_remaining_usage(user_email, 'analyst_ai')}")

# =========================
# FILE PROCESSING
# =========================
if uploaded_files and st.session_state["last_uploaded_files"] != uploaded_files:

    st.session_state["last_uploaded_files"] = uploaded_files

    raw_dfs = {}

    with st.spinner("Processing files..."):
        for i, file in enumerate(uploaded_files):
            ext = file.name.split('.')[-1].lower()
            df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)

            raw_dfs[f"df_{i+1}"] = df
            raw_dfs[f"df_{i+1}_name"] = file.name

    st.session_state["raw_dfs"] = raw_dfs
    st.success("✅ Files uploaded")

raw_dfs = st.session_state.get("raw_dfs", {})
txns_df = st.session_state["txns_df"]

# =========================
# NAVIGATION
# =========================
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
    "Navigation",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected_page

# =========================
# PAGE ROUTING
# =========================
if selected_page == "📘 Instructions":
    st.subheader("Instructions")
    st.write("Upload → Map → Analyze")

elif selected_page == "🗂️ File Mapping":
    if uploaded_files:
        mapped_data, confirmed = classify_and_extract_data(uploaded_files)

        if confirmed:
            st.session_state["txns_df"] = mapped_data.get("Transactions")
            st.session_state["cust_df"] = mapped_data.get("Customers")
            st.session_state["prod_df"] = mapped_data.get("Products")
            st.session_state["promo_df"] = mapped_data.get("Promotions")
            st.session_state["files_mapped"] = True

            st.success("✅ Mapping complete")

    else:
        st.warning("Upload files first")

elif selected_page == "📊 Sales Analytics":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        render_sales_analytics(txns_df)
        insights = generate_sales_insights(txns_df)
        generate_dynamic_insights(insights)

elif selected_page == "🔍 Sub-Category":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        render_subcategory_trends(txns_df)

elif selected_page == "📊 RFM":
    if txns_df is None:
        st.warning("Upload Transactions")
    else:
        rfm_df = calculate_rfm(txns_df)
        st.dataframe(rfm_df, use_container_width=True)

elif selected_page == "🤖 Analyst AI":
    if raw_dfs:
        if check_usage_limit(user_email, "analyst_ai"):
            BA.run_business_analyst_tab(raw_dfs)
            KPI_analyst.run_kpi_analyst(raw_dfs)
    else:
        st.warning("Upload files")

elif selected_page == "🤖 Chatbot":
    if raw_dfs:
        if check_usage_limit(user_email, "chatbot"):
            chatbot2.run_chat(raw_dfs)
    else:
        st.warning("Upload files")

# =========================
# LOGOUT / RESET
# =========================
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()
