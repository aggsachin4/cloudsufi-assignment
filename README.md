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
2. **Chunking:** text is split into approximately 900-word chunks with 150 words of overlap. Overlap reduces the chance of losing context at chunk boundaries.
3. **Indexing:** Gemini `gemini-embedding-001` creates vectors. For this assignment's small input size, vectors are stored in memory and ranked with cosine similarity using NumPy.
4. **Retrieval:** the top five chunks are selected for each question.
5. **Generation:** Gemini `gemini-2.5-flash` receives only the selected excerpts. The system prompt requires it to answer from those excerpts and cite them as `[Source N]`.
6. **UI citations:** each source expands to show the filename, page, similarity score, and exact retrieved text.

The separation between `rag.py` and `app.py` keeps the RAG logic testable and the UI thin.

## Testing

The deterministic chunking and metadata tests can run without an API key:

```bash
pytest -q
```

## Configuration

- `GEMINI_API_KEY`: required API key.
- `GEMINI_CHAT_MODEL`: defaults to `gemini-2.5-flash`.
- `GEMINI_EMBEDDING_MODEL`: defaults to `gemini-embedding-001`.

Do not commit `.env`; `.env.example` contains only the configuration shape.

