import streamlit as st
import pandas as pd
from collections import defaultdict

# ------------------ CONFIG ------------------

REQUIRED_FIELDS = {
    "Transactions": {
        "Invoice ID": ["invoice_id", "bill_no"],
        "Date": ["date", "invoice_date"],
        "Sub Category": ["subcategory"],
        "Invoice Total": ["amount", "sales"],
        "Quantity": ["qty", "quantity"],
        "Discount": ["discount"],
        "Description": ["description"],
        "Transaction Type": ["type"],
        "Production Cost": ["cost"],
        "Unit Cost": ["unit_cost"],
        "Product ID": ["productid"],
        "Customer ID": ["customerid"],
        "Unit Price": ["unit_price", "price"]
    }
}

# ------------------ HELPERS ------------------

def normalize(col):
    return col.strip().lower().replace(" ", "_")


@st.cache_data
def load_files(files):
    dfs = []
    for file in files:
        ext = file.name.split('.')[-1].lower()
        df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)
        dfs.append((file.name, df))
    return dfs


def build_inventory(file_dfs):
    inv = defaultdict(list)
    for fname, df in file_dfs:
        for col in df.columns:
            inv[normalize(col)].append((fname, col))
    return inv


def auto_map(role, inventory):
    mapping = {}
    for field, aliases in REQUIRED_FIELDS[role].items():
        candidates = [field] + aliases
        candidates = [normalize(c) for c in candidates]

        for c in candidates:
            if c in inventory:
                mapping[field] = inventory[c][0]
                break
    return mapping


# ------------------ MAIN ------------------

def classify_and_extract_data(uploaded_files):

    file_dfs = load_files(uploaded_files)
    inventory = build_inventory(file_dfs)

    all_cols = sorted({col for _, df in file_dfs for col in df.columns})

    # ---------------- FORM ----------------
    with st.form("mapping_form"):

        auto_mapping = auto_map("Transactions", inventory)

        for field in REQUIRED_FIELDS["Transactions"]:
            key = f"Transactions_{field}"

            if key not in st.session_state:
                st.session_state[key] = auto_mapping.get(field, ("", "--"))[1] if field in auto_mapping else "--"

            st.selectbox(field, ["--"] + all_cols, key=key)

        submitted = st.form_submit_button("✅ Confirm Mapping")

    # ---------------- PROCESS ----------------
    if submitted:

        df_final = pd.DataFrame()

        # 🔑 STRICT MAPPING
        for field in REQUIRED_FIELDS["Transactions"]:
            col_name = st.session_state.get(f"Transactions_{field}")

            if col_name and col_name != "--":
                found = False

                for fname, df in file_dfs:
                    if col_name in df.columns:
                        df_final[field] = df[col_name]
                        found = True
                        break

                if not found:
                    st.error(f"{field} not found in any file")

        # 🔑 NUMERIC FIX
        for col in ["Quantity", "Unit Price", "Invoice Total"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        # 🔑 CRITICAL: CALCULATE TOTAL
        if "Invoice Total" not in df_final.columns or df_final["Invoice Total"].isna().all():
            if "Quantity" in df_final and "Unit Price" in df_final:
                df_final["Invoice Total"] = df_final["Quantity"] * df_final["Unit Price"]

        return {"Transactions": df_final}, True

    return None, False
