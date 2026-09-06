from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag import answer_question, create_gemini_client, extract_chunks, get_model_settings, RAGIndex

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
    st.caption("Answers use the selected excerpts and include document/page citations.")

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
            with st.spinner("Extracting text and creating embeddings..."):
                files = [(file.name, file) for file in uploaded_files]
                chunks = extract_chunks(files)
                _, embedding_model = get_model_settings()
                index = RAGIndex(create_gemini_client(), embedding_model)
                index.build(chunks)
                st.session_state.index = index
                st.session_state.document_names = [file.name for file in uploaded_files]
            st.sidebar.success(f"Indexed {len(chunks)} chunks.")
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
                chat_model, _ = get_model_settings()
                with st.spinner("Searching the documents and writing an answer..."):
                    retrieved = st.session_state.index.search(question, top_k=5)
                    answer = answer_question(
                        st.session_state.index.client,
                        chat_model,
                        question,
                        retrieved,
                    )
                st.subheader("Answer")
                st.markdown(answer)
                st.subheader("Retrieved sources")
                for number, result in enumerate(retrieved, start=1):
                    label = f"Source {number}: {result.chunk.source}, page {result.chunk.page}"
                    with st.expander(label):
                        st.caption(f"Similarity score: {result.score:.3f}")
                        st.write(result.chunk.text)
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
