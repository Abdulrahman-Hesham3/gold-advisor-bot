"""
Gold Advisor Bot - web interface (Streamlit).

Run locally:  streamlit run streamlit_app.py
All logic lives in advisor_core.py. This file only lays out the page.
NOT FINANCIAL ADVICE. Student project demonstration.
"""
import os
import json

import pandas as pd
import streamlit as st

import advisor_core as core

st.set_page_config(page_title="Gold Advisor Bot", page_icon="📈", layout="centered")

# The Gemini key lives in Streamlit's secret store and is never shown on the page.
try:
    if not os.environ.get("GEMINI_API_KEY") and "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
except Exception:
    pass


@st.cache_resource(ttl=core.REFRESH_SECONDS, show_spinner="Loading the model and the latest gold prices...")
def get_engine():
    return core.Engine()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_explanation(facts_json):
    return core.explain_text(json.loads(facts_json))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_answer(question, facts_json):
    return core.answer_text(question, json.loads(facts_json))


EXAMPLE_QUESTIONS = ["What does RSI mean?", "Why is the recommendation what it is?",
                     "How often is this model right?", "Should I sell my gold now?"]

eng = get_engine()
first, last = eng.table.index[0].date(), eng.table.index[-1].date()

st.title("Gold Advisor Bot")
st.markdown("Pick a day and get the model's **Buy / Hold / Sell** call for the next trading day, "
            "with the reasons in plain English and an honest look at how reliable it is. "
            "*Student project. Not financial advice.*")

if "day" not in st.session_state:
    st.session_state.day = last
left, right = st.columns([3, 1], vertical_alignment="bottom")
with left:
    st.date_input("Day to analyse", key="day", min_value=first, max_value=last, format="DD/MM/YYYY",
                  help=f"Any day from {first:%d %b %Y} (after the model's training period) "
                       f"to {last:%d %b %Y}. Weekends and holidays use the previous trading day.")
with right:
    st.button("Latest day", on_click=lambda: st.session_state.update(day=last))

rec = eng.recommend(pd.Timestamp(st.session_state.day))
facts_json = json.dumps(core.facts_for_llm(rec, eng), sort_keys=True)

st.caption(f"Prices: {eng.source}")
st.markdown(core.card_html(rec, eng), unsafe_allow_html=True)
st.write("")

why, chart, reliable, ask, about = st.tabs(
    ["Why this call?", "Price chart", "How reliable is this?", "Ask a question", "About"])

with chart:
    st.pyplot(core.price_chart(eng, rec["date"]))

with reliable:
    st.markdown(core.reliability_md(eng))
    st.pyplot(core.backtest_chart(eng))
    st.markdown(core.FINDINGS_MD)

with ask:
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Your question", placeholder="e.g. What does RSI mean?")
        sent = st.form_submit_button("Ask")
    st.caption("Try: " + " · ".join(f"*{q}*" for q in EXAMPLE_QUESTIONS))
    if sent and question.strip():
        with st.spinner("Thinking..."):
            st.markdown(cached_answer(question.strip()[:500], facts_json))

with about:
    st.markdown(core.ABOUT_MD)

with why:  # filled last so the charts don't wait for the AI
    with st.spinner("Writing a plain-English explanation..."):
        text, problem = cached_explanation(facts_json)
    st.markdown(text)
    if problem:
        st.caption(f"AI rewrite unavailable ({problem}). Showing the built-in explanation.")
    with st.expander("The rule-based reasons behind it"):
        st.markdown("\n".join(f"- {r}" for r in rec["reasons"]))

st.divider()
st.caption("Demonstration only. Past performance does not predict future results. Not financial advice.")
