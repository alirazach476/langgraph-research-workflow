"""Streamlit UI for the LangGraph research workflow."""

import streamlit as st

from workflow import OUTPUT_DIR, run_research

st.set_page_config(page_title="LangGraph Research", page_icon="🔬", layout="wide")
st.title("LangGraph Research Workflow")
st.caption("plan → web research → draft → critique loop → final report")

topic = st.text_input("Research topic", placeholder="e.g. Edge AI for manufacturing quality control")
if st.button("Run research workflow", type="primary") and topic.strip():
    with st.spinner("Running multi-step LangGraph workflow..."):
        result = run_research(topic.strip())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Plan")
        for q in result["plan"]:
            st.write(f"- {q}")
        st.subheader("Critique")
        st.write(result.get("critique", ""))
        st.write(f"Approved: {result['approved']} | Revisions: {result['revision_count']}")
    with col2:
        st.subheader("Report")
        st.markdown(result["report"])
    st.info(f"Reports also saved in `{OUTPUT_DIR}`")
