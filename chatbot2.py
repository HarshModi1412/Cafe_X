import streamlit as st
import pandas as pd
import plotly.express as px
import re
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
def ask_chatgpt(messages, df_context=None, first_time=False):
    client = get_client()

    try:
        structured_messages = []

        # SYSTEM MESSAGE
        structured_messages.append({
            "role": "system",
            "content": "You are a smart business consultant. Give short, practical, data-backed advice. Avoid long answers."
        })

        # INITIAL PROMPT (ONLY FIRST TIME - NOT STORED)
        if first_time and df_context:
            structured_messages.append({
                "role": "user",
                "content": f"""
Give 3 short, practical tips to improve revenue or profit.

Format:
- 📌 Tip 1: ...
- 📌 Tip 2: ...
(Chart: X vs Y) ← only if useful

Keep it simple.

{df_context}
"""
            })

        # REAL CHAT HISTORY ONLY
        for msg in messages:
            structured_messages.append({
                "role": msg["role"],
                "content": msg["parts"][0]["text"]
            })

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=structured_messages,
            temperature=0.3,
            max_output_tokens=800
        )

        return response.output_text

    except Exception as e:
        return f"❌ OpenAI Error: {e}"


# -------------------------
# AUTO CHART DETECTION
# -------------------------
def try_plot_instruction(text, df):
    try:
        match = re.search(r"([A-Za-z0-9_ ]+)\s+vs\s+([A-Za-z0-9_ ]+)", text)
        if match:
            x_col = match.group(1).strip()
            y_col = match.group(2).strip()

            x_match = next(
                (col for col in df.columns if x_col.lower().replace(" ", "") in col.lower().replace(" ", "")),
                None
            )
            y_match = next(
                (col for col in df.columns if y_col.lower().replace(" ", "") in col.lower().replace(" ", "")),
                None
            )

            if x_match and y_match:
                return px.scatter(df, x=x_match, y=y_match, title=f"{y_match} vs {x_match}")

    except:
        return None

    return None


# -------------------------
# MAIN CHATBOT
# -------------------------
def run_chat(raw_dfs):

    st.set_page_config(page_title="Smart Business Consultant 📊", layout="wide")
    st.title("ChatGPT-Powered Business Consultant")

    # -------------------------
    # SESSION STATE
    # -------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "used_initial_prompt" not in st.session_state:
        st.session_state.used_initial_prompt = False

    # -------------------------
    # DATA CHECK
    # -------------------------
    if not raw_dfs:
        st.warning("⚠️ No data provided.")
        return

    valid_dfs = [df for df in raw_dfs.values() if isinstance(df, pd.DataFrame)]

    if not valid_dfs:
        st.warning("⚠️ No valid dataframes found.")
        return

    df_combined = pd.concat(valid_dfs, ignore_index=True)

    # -------------------------
    # DISPLAY CHAT HISTORY
    # -------------------------
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["parts"][0]["text"])

    # -------------------------
    # INPUT AT BOTTOM (FIXED)
    # -------------------------
    st.divider()
    user_input = st.chat_input("Ask something about your business...")

    # -------------------------
    # PROCESS INPUT
    # -------------------------
    if user_input:

        # Show user message
        st.chat_message("user").markdown(user_input)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "parts": [{"text": user_input}]
        })

        # Data context
        preview_json = df_combined.head(30).to_json(orient="records")
        df_context = f"Here is sample business data:\n{preview_json}"

        is_first = not st.session_state.used_initial_prompt

        # Call ChatGPT
        with st.spinner("Thinking..."):
            raw_response = ask_chatgpt(
                st.session_state.messages,
                df_context=df_context,
                first_time=is_first
            )

        st.session_state.used_initial_prompt = True

        # Clean response
        response = re.sub(r"```(json)?", "", raw_response, flags=re.DOTALL).strip("` \n")

        # Show assistant response
        st.chat_message("assistant").markdown(response)

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "parts": [{"text": response}]
        })

        # -------------------------
        # AUTO CHART
        # -------------------------
        fig = try_plot_instruction(response, df_combined)

        if fig:
            st.plotly_chart(fig, use_container_width=True)


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    run_chat({})
