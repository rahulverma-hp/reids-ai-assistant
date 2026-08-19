"""Document Q&A tab. Search logic lives in retrieval.py."""

from pathlib import Path

import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

import config
from retrieval import Index, chunk_document, format_context

SAMPLE_DOC = Path(__file__).parent / "sample_docs" / "tasting_room_faq.md"

SUGGESTED_QUESTIONS = [
    "What do we offer guests who don't drink alcohol?",
    "How long is the Signature tour and what's included?",
    "Can I tell a customer their gift voucher never expires?",
    "What does orris root do in the Signature Gin?",
]


def get_client() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


@st.cache_resource(show_spinner=False)
def get_index() -> Index:
    """Build the index once per session, seeded with the sample document."""
    index = Index()
    if SAMPLE_DOC.exists():
        index.add(chunk_document(SAMPLE_DOC.read_text(encoding="utf-8"), SAMPLE_DOC.name))
    return index


def extract_text(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def answer_question(question: str, k: int = 4):
    """Search the documents and answer from what comes back.

    Returns (answer, hits). If nothing clears the relevance threshold this says
    so rather than asking the model to answer from a weak match.
    """
    hits = get_index().search(question, k=k)
    if not hits:
        return (
            "Nothing in the indexed documents is a close enough match to answer "
            "that. Try rephrasing, or upload the document that would cover it.",
            [],
        )

    response = get_client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an operations assistant for Reid's Distillery, a craft "
                    "distillery in Toronto that also runs a retail shop, tours, "
                    "cocktail classes and private events.\n\n"
                    "Answer using only the context provided. Cite the source document "
                    "and chunk number for each claim. If the context does not contain "
                    "the answer, say so plainly instead of guessing. A confident wrong "
                    "answer about liquor regulations, a recipe spec or a refund policy "
                    "costs more than no answer. If the context says a detail must be "
                    "confirmed with a person, pass that instruction on rather than "
                    "answering around it. Be concise and practical."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content, hits


def render():
    st.header("Document Q&A")
    st.write(
        "For the documents staff keep asking about: AGCO and excise paperwork, "
        "botanical recipes and batch specs, tasting room scripts, venue policies. "
        "Every answer cites the document it came from."
    )

    index = get_index()

    with st.expander("Add your own documents"):
        files = st.file_uploader(
            "PDF, text or markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if files and st.button("Index documents"):
            added = sum(index.add(chunk_document(extract_text(f), f.name)) for f in files)
            st.success(f"Indexed {added} chunks from {len(files)} file(s).")

    st.info(
        f"{len(index)} chunks indexed from: {', '.join(index.sources)}. A sample "
        "tasting room reference is preloaded so you can try this immediately. "
        "Upload your real documents to add to it."
    )

    st.write("**Try one of these:**")
    cols = st.columns(2)
    for i, question_text in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(question_text, width="stretch", key=f"sq_{i}"):
            st.session_state.dq_question = question_text

    question = st.text_input(
        "Or ask your own question",
        value=st.session_state.get("dq_question", ""),
    )

    if not question:
        return

    if not config.DEEPSEEK_API_KEY:
        st.error("OpenRouter API key not set. Add it to your .env file.")
        return

    with st.spinner("Searching documents..."):
        answer, hits = answer_question(question)

    st.write("**Answer**")
    st.markdown(answer)

    if hits:
        with st.expander(f"Passages this answer was based on ({len(hits)})"):
            for hit in hits:
                st.write(
                    f"**{hit.chunk.source}, chunk {hit.chunk.index}** "
                    f"(relevance {hit.score:.2f})"
                )
                st.text(hit.chunk.text)
