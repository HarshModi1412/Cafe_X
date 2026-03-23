import streamlit as st
import pandas as pd
import json
import plotly.express as px
import hashlib
from openai import OpenAI

# -------------------------
# INIT
# -------------------------
def get_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OPENAI_API_KEY missing")
        st.stop()
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -------------------------
# UTILS
# -------------------------
def stable_key(*args):
    """Generate deterministic unique key"""
    raw = "_".join([str(a) for a in args])
    return hashlib.md5(raw.encode()).hexdigest()


def df_hash(df):
    """Lightweight dataframe fingerprint"""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


# -------------------------
# LLM CALL (CACHED)
# -------------------------
@st.cache_data(show_spinner=False)
def ask_llm_cached(prompt):
    client = get_client()

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=900
        )
        return response.output_text
    except Exception as e:
        return f"❌ OpenAI Error: {e}"


# -------------------------
# INSIGHTS (CACHED)
# -------------------------
@st.cache_data(show_spinner=True)
def get_insights(df_hash_val, df_preview, df_stats, columns, dtypes):

    prompt = f"""
You are a senior business consultant.

Dataset columns:
{columns}

Column types:
{dtypes}

Statistical summary:
{df_stats}

Sample data:
{df_preview}

Goal: Improve profitability

Return ONLY JSON:

[
{{
"decision":"short insight",
"observation":"data-backed observation",
"why_it_matters":"business reasoning",
"action":"recommended action",
"impact":"estimated impact"
}}
]
"""

    raw = ask_llm_cached(prompt)

    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except:
        return []


# -------------------------
# CHART SPEC (CACHED)
# -------------------------
@st.cache_data(show_spinner=False)
def get_chart_spec(df_hash_val, insight_text, columns):

    prompt = f"""
You are a data visualization expert.

Columns:
{columns}

Insight:
"{insight_text}"

Return ONLY JSON:

{{
"chart_type": "bar | line | scatter | pie",
"x": "column",
"y": "column OR ['col1','col2']",
"title": "title"
}}
"""

    raw = ask_llm_cached(prompt)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return None


# -------------------------
# CHART GENERATION
# -------------------------
def generate_chart(df, spec):

    try:
        chart_type = spec["chart_type"].lower()
        x = spec["x"]
        y = spec["y"]
        title = spec.get("title", "Chart")

        if x not in df.columns:
            return None

        if isinstance(y, str):
            if y not in df.columns:
                return None

            d = df[[x, y]].dropna()
            d = d.groupby(x)[y].mean().reset_index()

            if chart_type == "bar":
                fig = px.bar(d, x=x, y=y, title=title)
            elif chart_type == "line":
                fig = px.line(d, x=x, y=y, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(d, x=x, y=y, title=title)
            elif chart_type == "pie":
                fig = px.pie(d, names=x, values=y, title=title)
            else:
                return None

        else:
            d = df[[x] + y].dropna()
            d = d.melt(id_vars=x, var_name="Series", value_name="Value")

            if chart_type == "bar":
                fig = px.bar(d, x=x, y="Value", color="Series", title=title)
            elif chart_type == "line":
                fig = px.line(d, x=x, y="Value", color="Series", title=title)
            elif chart_type == "scatter":
                fig = px.scatter(d, x=x, y="Value", color="Series", title=title)
            else:
                return None

        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=100),
            plot_bgcolor="rgba(0,0,0,0)"
        )

        return fig

    except Exception:
        return None


# -------------------------
# MAIN UI
# -------------------------
def run_business_analyst_tab(raw_dfs):

    st.title("🤖 AI Business Analyst")

    for filename, df in raw_dfs.items():

        if not isinstance(df, pd.DataFrame):
            continue

        st.header(f"📄 {filename}")

        st.dataframe(df.head(20))

        # ---- PREP ----
        df_hash_val = df_hash(df)

        preview = df.head(15).to_string(index=False)
        stats = df.describe(include="all").to_string()
        columns = ", ".join(df.columns)
        dtypes = df.dtypes.to_string()

        # ---- INSIGHTS ----
        insights = get_insights(
            df_hash_val, preview, stats, columns, dtypes
        )

        if not insights:
            st.warning("No insights generated")
            continue

        # ---- DISPLAY ----
        for i, ins in enumerate(insights):

            st.markdown(f"### 🔎 {ins.get('decision')}")
            st.markdown(f"**Observation:** {ins.get('observation')}")
            st.markdown(f"**Why:** {ins.get('why_it_matters')}")
            st.markdown(f"**Action:** {ins.get('action')}")
            st.markdown(f"**Impact:** {ins.get('impact')}")

            # ---- CHART ----
            spec = get_chart_spec(df_hash_val, ins.get("decision"), columns)

            if spec:
                fig = generate_chart(df, spec)

                if fig:
                    key = stable_key(filename, i, ins.get("decision"))

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=key
                    )
                else:
                    st.warning("Chart generation failed")
            else:
                st.warning("No chart spec")
