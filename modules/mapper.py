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

    try:
        if ext == "csv":
            try:
                return pd.read_csv(file, encoding="utf-8")
            except:
                file.seek(0)
                return pd.read_csv(file, encoding="latin1")

        elif ext in ("xlsx", "xls"):
            return pd.read_excel(file, engine="openpyxl")

    except Exception as e:
        return None

    return None


@st.cache_data(show_spinner=False)
def build_column_inventory(files):
    inventory = defaultdict(list)
    file_dfs = []

    for file in files:
        df = load_file(file)
        if df is None:
            st.warning(f"Skipping file: {file.name}")
            continue

        file_dfs.append((file.name, df))

        for col in df.columns:
            inventory[normalize(col)].append((file.name, col))

    return inventory, file_dfs


def auto_map_fields(role, inventory):
    if role not in REQUIRED_FIELDS:
        raise ValueError(f"Invalid role: {role}")

    mapping = {}

    for field, aliases in REQUIRED_FIELDS[role].items():
        candidates = [field] + aliases
        candidates = [normalize(c) for c in candidates]

        for c in candidates:
            if c in inventory:
                mapping[field] = inventory[c][0]  # (file, column)
                break

    return mapping


def build_dataframe_from_mapping(mapping, file_dfs, required_fields):
    file_lookup = {name: df for name, df in file_dfs}

    columns = {}
    max_len = 0

    for field, (fname, col) in mapping.items():
        series = file_lookup[fname][col].reset_index(drop=True)
        columns[field] = series
        max_len = max(max_len, len(series))

    df = pd.DataFrame({
        field: columns.get(field, pd.Series([pd.NA] * max_len))
        for field in required_fields
    })

    # Derived fields
    if "Invoice Total" in df and df["Invoice Total"].isna().all():
        if {"Unit Price", "Quantity"}.issubset(df.columns):
            df["Invoice Total"] = (
                pd.to_numeric(df["Unit Price"], errors="coerce") *
                pd.to_numeric(df["Quantity"], errors="coerce")
            )

    if "Production Cost" in df and df["Production Cost"].isna().all():
        if {"Unit Cost", "Quantity"}.issubset(df.columns):
            df["Production Cost"] = (
                pd.to_numeric(df["Unit Cost"], errors="coerce") *
                pd.to_numeric(df["Quantity"], errors="coerce")
            )

    return df


# ------------------ MAIN ------------------

def classify_and_extract_data(uploaded_files):
    inventory, file_dfs = build_column_inventory(uploaded_files)

    all_mappings = {}
    final_data = {}

    for role in REQUIRED_FIELDS.keys():

        auto_mapping = auto_map_fields(role, inventory)
        missing = [f for f in REQUIRED_FIELDS[role] if f not in auto_mapping]

        st.markdown(f"### 🗂 Mapping for `{role}`")

        manual_mapping = {}

        if missing:
            st.warning(f"Manual mapping needed: {', '.join(missing)}")

            all_cols = sorted({
                col for _, df in file_dfs for col in df.columns
            })

            for field in missing:
                sel = st.selectbox(
                    f"{role} → {field}",
                    ["--"] + all_cols,
                    key=f"{role}_{field}"
                )

                if sel != "--":
                    for fname, df in file_dfs:
                        if sel in df.columns:
                            manual_mapping[field] = (fname, sel)
                            break

        all_mappings[role] = {**auto_mapping, **manual_mapping}

    if st.button("✅ Confirm and Start Analytics"):
        for role, mapping in all_mappings.items():
            fields = list(REQUIRED_FIELDS[role].keys())

            final_data[role] = build_dataframe_from_mapping(
                mapping, file_dfs, fields
            )

        return final_data

    return None
