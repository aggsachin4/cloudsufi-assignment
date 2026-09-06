from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag import RAGIndex, answer_question, create_chat_model, extract_documents, get_model_settings

load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
st.title("Document Q&A")
st.caption("Upload up to three PDFs, then ask questions grounded in their content.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Choose 1-3 PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned PDFs need OCR before upload.",
    )
    process_documents = st.button("Process documents", type="primary", use_container_width=True)
    st.divider()
    st.caption("Chroma retrieves relevant excerpts; Gemini returns a structured answer with citations.")

if "index" not in st.session_state:
    st.session_state.index = None
if "document_names" not in st.session_state:
    st.session_state.document_names = []

if process_documents:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF first.")
    elif len(uploaded_files) > 3:
        st.sidebar.error("Please upload no more than three PDFs.")
    elif not os.getenv("GEMINI_API_KEY", "").strip():
        st.sidebar.error("Add GEMINI_API_KEY to .env, then restart the app.")
    else:
        try:
            with st.spinner("Extracting text and creating Chroma embeddings..."):
                documents = extract_documents([(file.name, file) for file in uploaded_files])
                _, embedding_model = get_model_settings()
                index = RAGIndex(embedding_model)
                index.build(documents)
                st.session_state.index = index
                st.session_state.document_names = [file.name for file in uploaded_files]
            st.sidebar.success(f"Indexed {len(documents)} chunks in Chroma.")
        except Exception as error:
            st.sidebar.error(f"Could not process documents: {error}")

if st.session_state.index is None:
    st.info("Upload your PDFs in the sidebar and select Process documents to begin.")
else:
    st.success("Ready to answer questions about: " + ", ".join(st.session_state.document_names))
    with st.form("question_form"):
        question = st.text_area(
            "Ask a question",
            placeholder="What are the main requirements described in the documents?",
            height=100,
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                chat_model_name, _ = get_model_settings()
                with st.spinner("Searching Chroma and writing a grounded answer..."):
                    retrieved = st.session_state.index.search(question, top_k=5)
                    result = answer_question(
                        create_chat_model(chat_model_name), question, retrieved
                    )
                st.subheader("Answer")
                st.write(result.answer)
                if result.citations:
                    st.caption(
                        "Cited excerpts: "
                        + ", ".join(f"Source {item.source_number}" for item in result.citations)
                    )
                st.subheader("Retrieved sources")
                for item in retrieved:
                    metadata = item.document.metadata
                    label = f"Source {item.source_number}: {metadata['source']}, page {metadata['page']}"
                    with st.expander(label):
                        st.caption(f"Relevance score: {item.score:.3f}")
                        st.write(item.document.page_content)
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
