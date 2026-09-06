# Document Q&A

A small RAG-based PDF question-answering application built for the Cloudsufi take-home assignment. Upload one to three text-based PDFs, ask a natural-language question, and receive an answer grounded in retrieved document excerpts with document and page citations.

## Run locally

### Prerequisites

- Python 3.10 or newer
- A Gemini API key from Google AI Studio

### Setup

```bash
python -m venv .venv

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your key in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the app with one command:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload 1-3 PDFs, click **Process documents**, and ask a question.

## Architecture

1. **Ingestion:** `pypdf` extracts text page by page. Each chunk retains the original filename and page number.
2. **Chunking:** LangChain's `RecursiveCharacterTextSplitter` splits each page into approximately 3,500-character chunks with 500 characters of overlap, favouring natural paragraph and sentence boundaries.
3. **Indexing:** LangChain's Gemini embeddings create vectors in an in-memory Chroma collection. Each chunk keeps filename, page, and chunk-number metadata.
4. **Retrieval:** the top five chunks are selected for each question.
5. **Generation:** LangChain's Gemini chat model receives only the selected excerpts and returns a Pydantic-validated answer with structured source-number citations.
6. **UI citations:** each source expands to show the filename, page, similarity score, and exact retrieved text.

The separation between `rag.py` and `app.py` keeps the RAG logic testable and the UI thin.

## Testing

The deterministic splitting, metadata, and Pydantic validation tests can run without an API key:

```bash
pytest -q
```

## Configuration

- `GEMINI_API_KEY`: required API key.
- `GEMINI_CHAT_MODEL`: defaults to `gemini-2.5-flash`.
- `GEMINI_EMBEDDING_MODEL`: defaults to `gemini-embedding-001`.

Do not commit `.env`; `.env.example` contains only the configuration shape.

## Known limitations

- Only text-based PDFs are supported; scanned PDFs need OCR before upload.
- Chroma runs in memory for the active Streamlit session, so documents and vectors are rebuilt after a restart.
- Retrieval uses semantic similarity only. It does not yet apply metadata filters, hybrid search, or reranking.
- The model is instructed and validated to cite retrieved sources, but LLM output is not a guarantee of factual correctness.
- The app is intentionally limited to a small, single-user demo: it has no authentication, persistent storage, or background ingestion.

## Improvements with more time

- Add OCR and table-aware PDF parsing for image-based and complex documents.
- Persist Chroma collections using document hashes to avoid re-embedding unchanged files.
- Add hybrid retrieval, reranking, and an evaluation set that measures retrieval recall, citation precision, and answer faithfulness.
- Add streaming answers, retries, rate limiting, structured logs, metrics, authentication, and multi-user isolation.
- Store text offsets or section headings to provide more precise citations than page-level references.

