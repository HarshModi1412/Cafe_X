import streamlit as st
import pandas as pd
from collections import defaultdict

# ------------------ CONFIG ------------------

REQUIRED_FIELDS = {
    "Transactions": {
        "Invoice ID": ["invoice_id", "bill_no", "invoice number", "InvoiceNo", "Invoice No","orderid","order id","order_id","transaction_id"],
        "Date": ["date", "invoice_date", "purchase_date", "orderdate"],
        "Sub Category": ["subcat", "product_type", "subcategory"],
        "Invoice Total": ["amount", "invoice_amount", "total_amount", "grand_total", "sales"],
        "Quantity": ["qty", "units", "number of items", "quantity"],
        "Discount": ["discount"],
        "Description": ["description"],
        "Transaction Type": ["transaction_type", "type"],
        "Production Cost": ["cost", "production_cost"],
        "Unit Cost": ["unit_cost"],
        "Product ID": ["prod_id", "item_code", "stockcode"],
        "Customer ID": ["customer", "cust_id", "client_id"],
        "Unit Price": ["unit_price", "price"]
    }
}

# ------------------ HELPERS ------------------

def normalize(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


@st.cache_data(show_spinner=False)
def load_file(file):
    ext = file.name.lower().split('.')[-1]

    if ext == "csv":
        try:
            return pd.read_csv(file, encoding="utf-8")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    elif ext in ("xlsx", "xls"):
        return pd.read_excel(file, engine="openpyxl")

    return None


@st.cache_data(show_spinner=False)
def build_column_inventory(files):
    inventory = defaultdict(list)
    file_dfs = []

    for file in files:
        df = load_file(file)
        if df is None:
            continue

        file_dfs.append((file.name, df))

        for col in df.columns:
            inventory[normalize(col)].append((file.name, col))

    return inventory, file_dfs


def auto_map_fields(role, inventory):
    mapping = {}

    for field, aliases in REQUIRED_FIELDS[role].items():
        candidates = [field] + aliases
        candidates = [normalize(c) for c in candidates]

        for c in candidates:
            if c in inventory:
                mapping[field] = inventory[c][0]
                break

    return mapping


def build_dataframe_from_mapping(mapping, file_dfs, required_fields):
    file_lookup = {name: df for name, df in file_dfs}

    columns = {}
    max_len = 0

    # -------- SAFE COLUMN EXTRACTION --------
    for field, (fname, col) in mapping.items():

        df = file_lookup.get(fname)

        if df is not None and col in df.columns:
            s = df[col].reset_index(drop=True)
        else:
            s = pd.Series([pd.NA])

        columns[field] = s
        max_len = max(max_len, len(s))

    df = pd.DataFrame({
        field: columns.get(field, pd.Series([pd.NA] * max_len))
        for field in required_fields
    })

    # -------- FIX: NUMERIC CONVERSION --------
    for col in ["Quantity", "Unit Price", "Invoice Total", "Production Cost", "Unit Cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------- FIX: CALCULATE INVOICE TOTAL --------
    if "Invoice Total" not in df.columns or df["Invoice Total"].isna().all():
        if "Quantity" in df.columns and "Unit Price" in df.columns:
            df["Invoice Total"] = df["Quantity"] * df["Unit Price"]

    # -------- FIX: CALCULATE PRODUCTION COST --------
    if "Production Cost" not in df.columns or df["Production Cost"].isna().all():
        if "Unit Cost" in df.columns and "Quantity" in df.columns:
            df["Production Cost"] = df["Unit Cost"] * df["Quantity"]

    return df


# ------------------ MAIN ------------------

def classify_and_extract_data(uploaded_files):

    inventory, file_dfs = build_column_inventory(uploaded_files)

    # ---------------- SESSION STATE INIT ----------------
    if "mapping_store" not in st.session_state:
        st.session_state.mapping_store = {}

    if "confirm_mapping_clicked" not in st.session_state:
        st.session_state.confirm_mapping_clicked = False

    all_cols = sorted({
        col for _, df in file_dfs for col in df.columns
    })

    # ---------------- UI ----------------
    for role in REQUIRED_FIELDS:

        st.markdown(f"### 🗂 Mapping for `{role}`")

        auto_mapping = auto_map_fields(role, inventory)

        for field in REQUIRED_FIELDS[role]:

            key = f"{role}_{field}"

            default_val = "--"
            if field in auto_mapping:
                default_val = auto_mapping[field][1]

            if key not in st.session_state.mapping_store:
                st.session_state.mapping_store[key] = default_val

            current_val = st.session_state.mapping_store[key]

            selection = st.selectbox(
                f"{field}",
                ["--"] + all_cols,
                index=(["--"] + all_cols).index(current_val) if current_val in all_cols else 0,
                key=key
            )

            st.session_state.mapping_store[key] = selection

    # ---------------- BUTTON ----------------
    if st.button("✅ Confirm Mapping"):
        st.session_state.confirm_mapping_clicked = True

    # ---------------- SAVE ----------------
    if st.session_state.confirm_mapping_clicked:

        final_data = {}

        with st.spinner("💾 Saving mapping..."):

            for role in REQUIRED_FIELDS:

                role_mapping = {}

                for field in REQUIRED_FIELDS[role]:
                    sel = st.session_state.mapping_store.get(f"{role}_{field}")

                    if sel and sel != "--":

                        matched = False

                        for fname, df in file_dfs:
                            if sel in df.columns:
                                role_mapping[field] = (fname, sel)
                                matched = True
                                break

                        if not matched:
                            st.warning(f"⚠️ Column '{sel}' not found")

                final_data[role] = build_dataframe_from_mapping(
                    role_mapping,
                    file_dfs,
                    list(REQUIRED_FIELDS[role].keys())
                )

        st.session_state.confirm_mapping_clicked = False

        return final_data, True

    return None, False
