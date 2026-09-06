# Interview Guide: Document Q&A

## 30-second summary

This is a deliberately small Retrieval-Augmented Generation application. It accepts up to three PDFs, extracts page-aware text chunks, embeds them, retrieves the most relevant chunks for a question, and asks a language model to answer only from those chunks. The answer cites numbered excerpts, while the UI exposes the exact filename and page for each citation.

## Demo flow

1. Start the app with `streamlit run app.py`.
2. Upload up to three related PDFs from any domain. For example, use a healthcare
	provider comparison, a healthcare benefits guide, and a patient-care report.
3. Click **Process documents** and point out that indexing is an explicit step.
4. Ask a question whose answer should be found in the uploaded corpus, such as:
	`Which healthcare provider has the strongest coverage for preventive care?`
5. Open the retrieved source panels and show that the answer is traceable to
	specific documents and pages.
6. Repeat the demo with another domain when useful: upload three legal PDFs and
	ask `What is the POSH Act?`, or upload three product catalogues and ask
	`Which watch has the best battery life?`
7. Ask an unrelated question, such as: `What is the weather tomorrow?`, and
	explain that the grounded prompt instructs the model to say it cannot find
	unsupported information in the uploaded documents.

The important point is that the app is not trained for one particular subject.
The PDFs define the temporary knowledge base, so the same workflow works for
healthcare, law, product research, or another document-heavy domain. Answers
should be framed as comparisons supported by the uploaded documents, not as
unqualified universal facts such as an objectively best provider or watch.

## Why this design

- **Why RAG?** The answer should be based on user-provided documents, and retrieval makes the relevant evidence explicit instead of putting the entire document in every prompt.
- **Why in-memory NumPy instead of a vector database?** The input is limited to 1-3 documents. An in-memory index removes infrastructure and makes the retrieval logic easy to inspect. The `RAGIndex` boundary can later be replaced by FAISS, Chroma, or a hosted store.
- **Why page-aware chunks?** Page metadata makes citations useful to a human reviewer and avoids returning an answer with no traceable evidence.
- **Why overlap?** A concept can cross an arbitrary chunk boundary. A modest overlap retains neighboring context.
- **Why two model calls?** Embeddings handle semantic retrieval; the chat model converts the retrieved evidence into a readable answer. These are separate concerns with different model behavior and cost profiles.
- **Why a low temperature?** Factual document QA benefits from predictable, less creative generation.

## Important tradeoffs

The implementation prioritizes clarity and a working single-command demo over production infrastructure. It does not persist indexes, support multiple users, OCR scanned documents, or guarantee perfect answers. The prompt reduces hallucinations but is not a formal proof of faithfulness, which is why the retrieved excerpts remain visible in the interface.

## Likely questions and answers

**How would you evaluate this system?**

I would create a small labelled set of questions with expected source pages and answers. I would measure retrieval recall@k, citation precision, answer correctness, and unsupported-claim rate. I would inspect failures separately because a wrong answer can come from retrieval or generation.

**What happens if a document is too large?**

The current index is intentionally sized for the assignment. For larger inputs I would batch embeddings, persist vectors in a vector store, add metadata filters, and possibly use a reranker after initial retrieval.

**How would you improve citations?**

Store character offsets or sentence spans during parsing, make the model return structured citation IDs, validate those IDs against retrieved chunks, and render exact quoted evidence beside the answer.

**How would you handle prompt injection inside a PDF?**

Treat document text as untrusted data, clearly delimit it from instructions, tell the model that excerpts are evidence rather than instructions, validate citations, and add adversarial evaluation cases. For a higher-risk deployment I would also add content filtering and tool permissions outside the model.

**What if retrieval returns irrelevant chunks?**

Expose scores and sources for debugging, tune chunk size/overlap, add metadata-aware filtering, use hybrid keyword plus vector search, and add a reranker. Evaluation should tell us whether the problem is recall or ranking.

**What would you change for production?**

Add authentication and tenant isolation, durable storage, background ingestion, OCR, rate limits, retries and timeouts, structured logs, metrics, encrypted secrets, model/version pinning, and an evaluation/monitoring pipeline.

## Code tour

- `app.py`: Streamlit workflow, upload validation, session state, and rendering.
- `rag.py`: PDF parsing, chunking, embedding index, retrieval, and grounded generation.
- `test_rag.py`: offline tests for deterministic behavior.
- `README.md`: setup, architecture, limitations, and extension points.
