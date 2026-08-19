"""Reid's Distillery operations assistant."""

import streamlit as st

import document_qa
import event_assistant
import inventory_helper

st.set_page_config(
    page_title="Reid's Distillery Operations Assistant",
    layout="wide",
)

st.title("Reid's Distillery Operations Assistant")
st.write(
    "Three tools for the admin work around making gin: drafting replies to event "
    "enquiries, tracking stock and what the current stockouts cost, and answering "
    "questions from your own documents."
)
st.divider()

enquiries, inventory, documents = st.tabs(
    ["Event Enquiries", "Inventory", "Document Q&A"]
)

with enquiries:
    event_assistant.render()

with inventory:
    inventory_helper.render()

with documents:
    document_qa.render()

st.sidebar.subheader("About")
st.sidebar.write(
    "Built by Rahul Verma for the Reid's Distillery Fall 2026 co-op application."
)
st.sidebar.markdown(
    "[GitHub](https://github.com/rahulverma-hp) | "
    "[Portfolio](https://rahulverma-hp.github.io)"
)
st.sidebar.write("Python, Streamlit, scikit-learn, Pandas, DeepSeek via OpenRouter.")
st.sidebar.caption("George Brown College, Applied A.I. Solutions Development")
