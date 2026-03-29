import streamlit as st
import pandas as pd
import json
import re
import numpy as np
import plotly.express as px
from openai import OpenAI

# -------------------------
# INIT CLIENT
# -------------------------
def get_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OPENAI_API_KEY missing")
        st.stop()
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -------------------------
# CHATGPT CALL
# -------------------------
def ask_llm(prompt):
    client = get_client()

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=1000
        )
        return response.output_text
    except Exception as e:
        st.error(f"❌ OpenAI Error: {e}")
        return ""


# -------------------------
# EXTRACT JSON
# -------------------------
def extract_json_from_text(text):
    try:
        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
    except:
        pass
    return ""


# -------------------------
# FUZZY MATCH
# -------------------------
def fuzzy_match(col, candidates):
    col = col.lower().replace(" ", "")
    for candidate in candidates:
        if col in candidate.lower().replace(" ", ""):
            return candidate
    return None


# -------------------------
# KPI GENERATION
# -------------------------
def get_kpi_list(data_preview, industry, scale, goal):

    prompt = f"""
You are a senior business analyst.

Dataset columns:
{data_preview.splitlines()[0]}

Industry: {industry}
Scale: {scale}
Goal: {goal}

Return ONLY JSON:

[
  {{
    "name": "KPI Name",
    "operation": "SUM / COUNT / AVERAGE / RATIO",
    "aggregation_map": {{
        "Column A": "SUM",
        "Column B": "COUNT"
    }},
    "group_by": ["Column X"],
    "why": "Reason"
  }}
]
"""

    raw = ask_llm(prompt)
    cleaned = extract_json_from_text(raw)

    try:
        return json.loads(cleaned)
    except:
        return []


# -------------------------
# KPI CALCULATION
# -------------------------
def calculate_kpis(df, kpis):

    results = []

    for kpi in kpis:
        try:
            agg_results = {}

            for col_key, agg_type in kpi["aggregation_map"].items():
                actual_col = fuzzy_match(col_key, df.columns)

                if not actual_col:
                    raise ValueError(f"{col_key} not found")

                if agg_type == "SUM":
                    agg_results[col_key] = df[actual_col].sum()
                elif agg_type == "COUNT":
                    agg_results[col_key] = df[actual_col].count()
                elif agg_type == "AVERAGE":
                    agg_results[col_key] = df[actual_col].mean()

            op = kpi["operation"].upper()

            if op in ["SUM", "COUNT", "AVERAGE"]:
                value = list(agg_results.values())[0]

            elif op == "RATIO":
                keys = list(agg_results.keys())
                value = agg_results[keys[0]] / agg_results[keys[1]] if agg_results[keys[1]] != 0 else 0

            else:
                value = None

            kpi["value"] = round(value, 2) if isinstance(value, (int, float, np.float64)) else value

        except Exception as e:
            kpi["value"] = "❌"
            kpi["error"] = str(e)

        results.append(kpi)

    return results


# -------------------------
# BENCHMARKS
# -------------------------
def add_benchmarks(kpis):
    for kpi in kpis:
        try:
            kpi["benchmark"] = round(float(kpi["value"]) * 1.1, 2)
        except:
            kpi["benchmark"] = "N/A"
    return kpis


# -------------------------
# INSIGHTS
# -------------------------
def get_insights(kpis, industry, scale, goal):

    prompt = f"""
You are a business consultant.

Based on KPIs below, give 3 insights.

Return JSON:

[
 {{
  "kpi": "...",
  "observation": "...",
  "action": "...",
  "impact": "..."
 }}
]

KPIs:
{kpis}
"""

    raw = ask_llm(prompt)
    cleaned = extract_json_from_text(raw)

    try:
        return json.loads(cleaned)
    except:
        return []


# -------------------------
# PLOT
# -------------------------
def plot_kpis(kpis):
    df_plot = pd.DataFrame([
        {"KPI": k["name"], "Type": "Company", "Value": k["value"]} for k in kpis if isinstance(k["value"], (int, float))
    ] + [
        {"KPI": k["name"], "Type": "Benchmark", "Value": k["benchmark"]} for k in kpis if isinstance(k["benchmark"], (int, float))
    ])

    return px.bar(df_plot, x="KPI", y="Value", color="Type", barmode="group")


# -------------------------
# MAIN FUNCTION
# -------------------------
def run_kpi_analyst(raw_dfs):

    st.header("📊 KPI Analyst")

    for key, df in raw_dfs.items():

        # ✅ Skip metadata keys
        if not isinstance(df, pd.DataFrame):
            continue

        # ✅ Clean name instead of df_1
        file_name = raw_dfs.get(f"{key}_name", key).replace(".csv", "").replace(".xlsx", "")

        st.subheader(f"📄 {file_name}")

        industry = st.text_input(f"Industry - {file_name}", key=f"ind_{key}")
        scale = st.text_input(f"Scale - {file_name}", key=f"scale_{key}")
        goal = st.text_area(f"Goal - {file_name}", key=f"goal_{key}")

        if not (industry and scale and goal):
            st.warning("Enter all fields")
            continue

        st.dataframe(df.head(10), use_container_width=True)

        kpis = get_kpi_list(df.head(10).to_string(index=False), industry, scale, goal)

        if not kpis:
            st.warning("No KPIs generated")
            continue

        kpis = calculate_kpis(df, kpis)
        kpis = add_benchmarks(kpis)

        st.markdown("### ✅ KPIs")
        st.dataframe(pd.DataFrame(kpis), use_container_width=True)

        st.markdown("### 📊 Comparison")
        fig = plot_kpis(kpis)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Insights")

        insights = get_insights(kpis, industry, scale, goal)

        for ins in insights:
            st.markdown(f"**{ins.get('kpi')}**")
            st.markdown(f"- {ins.get('observation')}")
            st.markdown(f"- Action: {ins.get('action')}")
            st.markdown(f"- Impact: {ins.get('impact')}")
