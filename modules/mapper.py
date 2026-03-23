import streamlit as st
import pandas as pd
from collections import defaultdict

# ------------------ CONFIG ------------------
REQUIRED_FIELDS = {...}  # keep same
IMPORTANT_FIELDS = ["Invoice Total", "Quantity", "Unit Price"]

# ------------------ HELPERS ------------------

@st.cache_data(show_spinner=False)
def load_file(file):
    """Fast file loader with fallback encoding"""
    file_ext = file.name.lower().split('.')[-1]

    if file_ext == "csv":
        try:
            return pd.read_csv(file, encoding="utf-8")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    elif file_ext in ("xlsx", "xls"):
        return pd.read_excel(file, engine="openpyxl")

    else:
        return None


def normalize_columns(columns):
    return {col: col.strip().lower().replace(" ", "_") for col in columns}


@st.cache_data(show_spinner=False)
def build_column_inventory(files):
    inventory = defaultdict(list)
    file_dfs = []

    for file in files:
        df = load_file(file)
        if df is None:
            continue

        norm_map = normalize_columns(df.columns)
        file_dfs.append((file.name, df, norm_map))

        for col, norm in norm_map.items():
            inventory[norm].append((file.name, col))

    return inventory, file_dfs


def auto_map_fields(role, inventory):
    mapping = {}

    for field, aliases in REQUIRED_FIELDS[role].items():
        candidates = [field] + aliases
        candidates = [c.lower().replace(" ", "_") for c in candidates]

        for c in candidates:
            if c in inventory:
                mapping[field] = inventory[c][0]  # only metadata
                break

    return mapping


def build_dataframe_from_mapping(mapping, file_dfs, required_fields):
    columns = {}

    # Create quick lookup
    file_lookup = {name: df for name, df, _ in file_dfs}

    for field, (fname, col) in mapping.items():
        df = file_lookup[fname]
        columns[field] = df[col].reset_index(drop=True)

    max_len = max((len(s) for s in columns.values()), default=0)

    df = pd.DataFrame({
        field: columns.get(field, pd.Series([pd.NA] * max_len))
        for field in required_fields
    })

    # Vectorized computations
    if "Invoice Total" in df:
        if df["Invoice Total"].isna().all():
            if {"Unit Price", "Quantity"}.issubset(df.columns):
                df["Invoice Total"] = (
                    pd.to_numeric(df["Unit Price"], errors="coerce") *
                    pd.to_numeric(df["Quantity"], errors="coerce")
                )

    if "Production Cost" in df:
        if df["Production Cost"].isna().all():
            if {"Unit Cost", "Quantity"}.issubset(df.columns):
                df["Production Cost"] = (
                    pd.to_numeric(df["Unit Cost"], errors="coerce") *
                    pd.to_numeric(df["Quantity"], errors="coerce")
                )

    return df


# ------------------ MAIN ------------------

def classify_and_extract_data(uploaded_files):

    inventory, file_dfs = build_column_inventory(uploaded_files)

    final_data = {}
    all_mappings = {}

    for role in REQUIRED_FIELDS:

        auto_mapping = auto_map_fields(role, inventory)
        missing = [f for f in REQUIRED_FIELDS[role] if f not in auto_mapping]

        st.markdown(f"### 🗂 Mapping for `{role}`")

        manual_mapping = {}

        if missing:
            st.warning(f"Manual mapping needed: {', '.join(missing)}")

            all_cols = sorted({
                col for _, df, _ in file_dfs for col in df.columns
            })

            for field in missing:
                sel = st.selectbox(
                    f"{role} → {field}",
                    ["--"] + all_cols,
                    key=f"{role}_{field}"
                )

                if sel != "--":
                    for fname, df, _ in file_dfs:
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
