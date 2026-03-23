import streamlit as st
import pandas as pd
from collections import defaultdict

# ------------------ CONFIG ------------------

REQUIRED_FIELDS = {
    "Transactions": {
        "Invoice ID": ["invoice_id", "bill_no", "invoice number", "InvoiceNo", "Invoice No", "orderid", "order id", "order_id","transaction_id","transaction id","transactionid"],
        "Date": ["date", "invoice_date", "purchase_date", "Invoicedate", "orderdate", "order date","transaction_date","transactiondate","transaction date"],
        "Sub Category": ["subcat", "product_type", "subcategory", "sub-category"],
        "Invoice Total": ["amount", "invoice_amount", "total_amount", "grand_total", "Sales", "sales"],
        "Quantity": ["qty", "units", "number of items"],
        "Discount": ["discount_amt", "disc", "offer_discount", "discount"],
        "Description": ["offer", "promo_desc", "discount name", "description"],
        "Transaction Type": ["transaction_type", "type", "return"],
        "Production Cost": ["cost", "production_cost", "item_cost"],
        "Unit Cost": ["unit_cost", "unit cost", "cost per unit"],
        "Product ID": ["prod_id", "item_code", "stockcode", "ProductID"],
        "Customer ID": ["customer", "cust_id", "cust number", "client_id", "CustomerID"],
        "Unit Price": ["unit", "price", "unit_price", "product price", "unitprice"]
    },
    "Customers": {
        "Customer ID": ["customer", "cust_id", "cust number", "client_id", "CustomerID"],
        "Gender": ["sex", "customer_gender"],
        "Name": ["name", "NAME", "customer_name"],
        "Telephone": ["telephone", "phone", "number"],
        "Email": ["email", "mail"],
        "Date Of Birth": ["date_of_birth", "dob"]
    },
    "Products": {
        "Product ID": ["prod_id", "item_code", "stockcode", "ProductID"],
        "Sub Category": ["subcategory", "subcat", "product_type", "catagory"],
        "Category": ["cat", "product_cat", "segment"]
    },
    "Promotions": {
        "Description": ["offer_desc", "campaign_desc", "promotion_name"],
        "Start": ["start", "from_date", "campaign_start", "start_date"],
        "End": ["end", "to_date", "campaign_end", "end_date"],
        "Discont": ["discount_value", "discount_rate", "disc"]
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

    for field, (fname, col) in mapping.items():
        s = file_lookup[fname][col].reset_index(drop=True)
        columns[field] = s
        max_len = max(max_len, len(s))

    df = pd.DataFrame({
        field: columns.get(field, pd.Series([pd.NA] * max_len))
        for field in required_fields
    })

    return df


# ------------------ MAIN ------------------

def classify_and_extract_data(uploaded_files):

    inventory, file_dfs = build_column_inventory(uploaded_files)

    # ✅ Persist mapping state
    if "mapping_store" not in st.session_state:
        st.session_state["mapping_store"] = {}

    all_cols = sorted({
        col for _, df in file_dfs for col in df.columns
    })

    # ---------------- UI ----------------
    for role in REQUIRED_FIELDS:

        st.markdown(f"### 🗂 Mapping for `{role}`")

        auto_mapping = auto_map_fields(role, inventory)

        for field in REQUIRED_FIELDS[role]:

            key = f"{role}_{field}"

            # Pre-fill from auto-mapping
            default_val = "--"
            if field in auto_mapping:
                default_val = auto_mapping[field][1]

            if key not in st.session_state["mapping_store"]:
                st.session_state["mapping_store"][key] = default_val

            selection = st.selectbox(
                f"{field}",
                ["--"] + all_cols,
                index=(["--"] + all_cols).index(
                    st.session_state["mapping_store"][key]
                ) if st.session_state["mapping_store"][key] in all_cols else 0,
                key=key
            )

            # Save selection (but DO NOT trigger final mapping)
            st.session_state["mapping_store"][key] = selection

    # ---------------- CONFIRM BUTTON ----------------
    if st.button("✅ Confirm Mapping"):

        final_data = {}

        with st.spinner("💾 Saving mapping..."):

            for role in REQUIRED_FIELDS:

                role_mapping = {}

                for field in REQUIRED_FIELDS[role]:
                    sel = st.session_state["mapping_store"].get(f"{role}_{field}")

                    if sel and sel != "--":
                        for fname, df in file_dfs:
                            if sel in df.columns:
                                role_mapping[field] = (fname, sel)
                                break

                final_data[role] = build_dataframe_from_mapping(
                    role_mapping,
                    file_dfs,
                    list(REQUIRED_FIELDS[role].keys())
                )

        return final_data

    return None
