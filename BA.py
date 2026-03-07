import streamlit as st
import pandas as pd
import json
import plotly.express as px
from openai import OpenAI


def run_business_analyst_tab(raw_dfs):

    # 🔑 Put your OpenAI API Key here
    OPENAI_API_KEY = "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"

    client = OpenAI(api_key=OPENAI_API_KEY)

    # -------------------------
    # LLM CALL
    # -------------------------
    def ask_llm(prompt):

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
    # INSIGHT GENERATION
    # -------------------------
    def get_insights_list(df):

        preview = df.head(15).to_string(index=False)
        stats = df.describe(include="all").to_string()
        columns = ", ".join(df.columns.tolist())
        dtypes = df.dtypes.to_string()

        prompt = f"""
You are a senior business consultant.

A company wants to improve profitability.

Dataset columns:
{columns}

Column types:
{dtypes}

Statistical summary:
{stats}

Sample data:
{preview}

Your job:

1. Profit = Revenue - Cost
2. Analyze revenue drivers
3. Analyze cost drivers
4. Identify root causes
5. Suggest actions
6. Estimate impact

Return ONLY JSON.

Format:

[
{{
"decision":"short actionable insight title",
"observation":"data observation with numbers",
"why_it_matters":"business reasoning",
"action":"what company should do",
"impact":"estimated profitability impact"
}}
]

Rules:
- 3–5 insights
- short sentences
- include numbers where possible
- focus on profitability improvement
"""

        raw = ask_llm(prompt)

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            raw_json = raw[start:end]
            return json.loads(raw_json)

        except Exception as e:
            st.warning(f"⚠️ Failed to parse insights JSON: {e}")
            return []

    # -------------------------
    # CHART SPEC GENERATION
    # -------------------------
    def get_chart_spec_from_insight(df, insight_text):

        columns = ", ".join(df.columns.tolist())

        prompt = f"""
You are a data visualization expert.

Dataset columns:
{columns}

Insight:
"{insight_text}"

Return ONLY JSON.

{{
"chart_type": "bar | line | scatter | pie",
"x": "column name",
"y": "column OR ['col1','col2']",
"title": "chart title"
}}

Rules:
- choose chart that best explains insight
- prefer averages or ratios over totals when comparing groups
- use only dataset columns
"""

        raw = ask_llm(prompt)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw_json = raw[start:end]
            return json.loads(raw_json)

        except Exception as e:
            st.warning(f"⚠️ Failed to parse chart spec: {e}")
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

                df_chart = df[[x, y]].dropna()
                df_chart = df_chart.groupby(x)[y].mean().reset_index()

                if chart_type == "bar":
                    fig = px.bar(df_chart, x=x, y=y, title=title)

                elif chart_type == "line":
                    fig = px.line(df_chart, x=x, y=y, title=title)

                elif chart_type == "scatter":
                    fig = px.scatter(df_chart, x=x, y=y, title=title)

                elif chart_type == "pie":
                    fig = px.pie(df_chart, names=x, values=y, title=title)

                else:
                    return None

            else:

                df_chart = df[[x] + y].dropna()

                df_melt = df_chart.melt(
                    id_vars=x,
                    value_vars=y,
                    var_name="Series",
                    value_name="Value"
                )

                if chart_type == "bar":
                    fig = px.bar(df_melt, x=x, y="Value", color="Series", title=title)

                elif chart_type == "line":
                    fig = px.line(df_melt, x=x, y="Value", color="Series", title=title)

                elif chart_type == "scatter":
                    fig = px.scatter(df_melt, x=x, y="Value", color="Series", title=title)

                else:
                    return None

            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=40, r=40, t=60, b=120),
                plot_bgcolor="rgba(0,0,0,0)"
            )

            return fig

        except Exception as e:
            st.error(f"Chart error: {e}")
            return None

    # -------------------------
    # STREAMLIT UI
    # -------------------------
    st.title("🤖 AI Business Analyst")

    for filename, df in raw_dfs.items():

        if not isinstance(df, pd.DataFrame):
            st.info(f"⏭️ Skipping non-DataFrame entry: {filename}")
            continue

        st.header(f"📄 Analysis for: {filename}")

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(20))

        st.subheader("📈 AI Insights")

        with st.spinner("Analyzing dataset..."):
            insights = get_insights_list(df)

        if not insights:
            st.warning("⚠️ No insights generated.")
            continue

        for i, ins in enumerate(insights):

            st.markdown(f"### 🔎 Insight {i+1}: {ins.get('decision')}")
            st.markdown(f"- **Observation:** {ins.get('observation')}")
            st.markdown(f"- **Why it matters:** {ins.get('why_it_matters')}")
            st.markdown(f"- **Action:** {ins.get('action')}")
            st.markdown(f"- **Impact:** {ins.get('impact')}")

            with st.spinner("Generating visualization..."):

                spec = get_chart_spec_from_insight(df, ins.get("decision"))

                if spec:

                    fig = generate_chart(df, spec)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("⚠️ Could not generate chart.")

                else:
                    st.warning("⚠️ No chart suggested.")
