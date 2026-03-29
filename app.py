import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# =========================
# CONFIG
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
        return {}

    df = pd.read_csv(USERS_FILE)

    if "email" not in df.columns or "password" not in df.columns:
        return {}

    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["password"] = df["password"].astype(str).str.strip()

    return dict(zip(df["email"], df["password"]))


def login_block():
    st.markdown('<div class="card"><h3>🔒 Login Required</h3></div>', unsafe_allow_html=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    users = load_users()

    if st.button("Login"):
        email_clean = email.strip().lower()

        if email_clean in users and users[email_clean] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = email_clean
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")


# =========================
# RATE LIMIT
# =========================
def init_usage_file():
    if not os.path.exists(USAGE_FILE):
        pd.DataFrame(columns=["email", "feature", "timestamp"]).to_csv(USAGE_FILE, index=False)


def check_usage_limit(email, feature):
    init_usage_file()
    df = pd.read_csv(USAGE_FILE)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    now = datetime.now()

    user_logs = df[(df["email"] == email) & (df["feature"] == feature)]
    user_logs = user_logs[user_logs["timestamp"] > (now - timedelta(hours=WINDOW_HOURS))]

    if len(user_logs) >= MAX_USAGE:
        st.error("Limit reached. Try later.")
        return False

    df = pd.concat([df, pd.DataFrame([{
        "email": email,
        "feature": feature,
        "timestamp": now
    }])], ignore_index=True)

    df.to_csv(USAGE_FILE, index=False)
    return True


def get_remaining_usage(email, feature):
    init_usage_file()
    df = pd.read_csv(USAGE_FILE)

    if df.empty:
        return MAX_USAGE

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    now = datetime.now()

    user_logs = df[(df["email"] == email) & (df["feature"] == feature)]
    user_logs = user_logs[user_logs["timestamp"] > (now - timedelta(hours=WINDOW_HOURS))]

    return MAX_USAGE - len(user_logs)


# =========================
# SESSION INIT
# =========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None


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
# UI (PREMIUM)
# =========================
st.set_page_config(page_title="Cafe_X", layout="wide")

st.markdown("""
<style>
body {background:#0b0f19;color:#e5e7eb;font-family:Inter;}
.card {background:#111827;padding:20px;border-radius:12px;border:1px solid #1f2937;margin-bottom:15px;}
.hero {background:linear-gradient(135deg,#1e3a8a,#6d28d9);padding:30px;border-radius:14px;margin-bottom:20px;}
.subtle {color:#9ca3af;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
<h1>🚀 Cafe_X Intelligence</h1>
<p>Turn raw data into business decisions instantly</p>
</div>
""", unsafe_allow_html=True)

# FEATURE CARDS
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="card"><h4>📊 Analytics</h4><p class="subtle">Sales insights</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h4>🤖 AI Analyst</h4><p class="subtle">Smart recommendations</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><h4>💬 Chatbot</h4><p class="subtle">Query your data</p></div>', unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
if st.session_state["logged_in"]:
    st.sidebar.markdown(f"""
    <div class="card">
        <div class="subtle">User</div>
        <div>{st.session_state['user']}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="card">
        <div class="subtle">Mode</div>
        <div>Guest</div>
        <div class="subtle">Login for AI</div>
    </div>
    """, unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)

if st.session_state["logged_in"]:
    user_email = st.session_state["user"]
    st.sidebar.markdown("### Usage")
    st.sidebar.write("Chatbot:", get_remaining_usage(user_email, "chatbot"))
    st.sidebar.write("Analyst:", get_remaining_usage(user_email, "analyst_ai"))


# =========================
# FILE HANDLING
# =========================
if uploaded_files:
    raw_dfs = {}
    for i, file in enumerate(uploaded_files):
        df = pd.read_csv(file)
        raw_dfs[f"df_{i}"] = df
    st.session_state["raw_dfs"] = raw_dfs

raw_dfs = st.session_state.get("raw_dfs", {})
txns_df = st.session_state.get("txns_df")


# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("", [
    "Instructions",
    "Mapping",
    "Analytics",
    "SubCategory",
    "RFM",
    "Analyst AI",
    "Chatbot"
])


# =========================
# PAGES
# =========================
if page == "Instructions":
    st.markdown('<div class="card">Upload → Map → Analyze</div>', unsafe_allow_html=True)

elif page == "Mapping":
    if uploaded_files:
        mapped, confirmed = classify_and_extract_data(uploaded_files)
        if confirmed:
            st.session_state["txns_df"] = mapped.get("Transactions")
            st.success("Mapping done")

elif page == "Analytics":
    if txns_df is not None:
        render_sales_analytics(txns_df)

elif page == "SubCategory":
    if txns_df is not None:
        render_subcategory_trends(txns_df)

elif page == "RFM":
    if txns_df is not None:
        st.dataframe(calculate_rfm(txns_df))

elif page == "Analyst AI":
    if not st.session_state["logged_in"]:
        login_block()
    else:
        if check_usage_limit(st.session_state["user"], "analyst_ai"):
            BA.run_business_analyst_tab(raw_dfs)

elif page == "Chatbot":
    if not st.session_state["logged_in"]:
        login_block()
    else:
        if check_usage_limit(st.session_state["user"], "chatbot"):
            chatbot2.run_chat(raw_dfs)


# =========================
# LOGOUT
# =========================
if st.session_state["logged_in"]:
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
