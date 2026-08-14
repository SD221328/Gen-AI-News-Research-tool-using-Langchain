# Gen-AI-News-Research-Tool-using-Langchain

A Retrieval-Augmented Generation (RAG) application that lets a user paste in up to 3 news-article URLs, builds a searchable knowledge base from them, and then answers natural-language questions **grounded only in those articles**, citing the source URL for every answer.

---

## 1. Objective

Reading through multiple long news/financial articles to answer a specific question (e.g. *"Is NVIDIA overvalued?"*) is slow and error-prone. The objective of this project is to build a tool that:

1. Ingests raw web articles from user-supplied URLs.
2. Converts their content into a searchable vector index.
3. Uses an LLM to answer a user's question **using only the retrieved article content** (not the model's general knowledge), reducing hallucination.
4. Returns the answer together with the exact source URL(s) it came from, so the user can verify it.

This mirrors a real-world **RAG (Retrieval-Augmented Generation)** pipeline — the same pattern used in production research/QA copilots.

---

## 2. Variables / Components

Rather than statistical variables, this is a pipeline of configurable components. The "inputs" and "moving parts" of the system are:

| Component | Role | Value used in this project |
|---|---|---|
| **Input URLs** | Up to 3 user-supplied news article links (sidebar text inputs) | e.g. financial news, stock analysis articles |
| **Document Loader** | `UnstructuredURLLoader` | Fetches and parses raw HTML into plain text `Document` objects, each tagged with `metadata={'source': url}` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | `chunk_size=1000`, `chunk_overlap=200`, separators `["\n\n", "\n", ".", ","]` — breaks long articles into LLM-sized, context-preserving chunks |
| **Embedding Model** | `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) | Converts each text chunk into a dense vector for similarity search |
| **Vector Store** | `FAISS` | In-memory similarity-search index built from the chunk embeddings; persisted to disk as `faiss_store_openai.pkl` |
| **Retriever** | `vectorstore.as_retriever()` | Given a question, finds the most semantically similar chunks from the FAISS index |
| **LLM** | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`), `temperature=0.9` | Generates the final answer from the retrieved chunks |
| **Prompt Template** | `ChatPromptTemplate` | Instructs the model to answer *only* from the given `{context}`, and say "I don't know" if the answer isn't present |
| **Retrieval Chain** | `create_retrieval_chain` + `create_stuff_documents_chain` | Wires retriever → prompt → LLM into a single callable RAG pipeline |
| **UI** | Streamlit | URL inputs, "Process URLs" button, question box, answer + sources display |

---

## 3. Methodology

The system follows the standard 5-stage RAG pipeline:

### Stage 1 — Ingest
When the user clicks **"Process URLs"**, `UnstructuredURLLoader` fetches each article and extracts its raw text plus a `source` metadata tag (the original URL), so answers can later be traced back to their origin.

### Stage 2 — Chunk
Full articles are too long to feed an LLM directly (token limits) and too coarse for precise retrieval. `RecursiveCharacterTextSplitter` recursively tries larger separators first (`\n\n`, then `\n`, then `.`, then `,`) and only falls back to smaller ones if a chunk still exceeds `chunk_size=1000` characters. A `chunk_overlap=200` keeps some shared context between adjacent chunks so an idea split across a chunk boundary isn't lost.

### Stage 3 — Embed & Index
Each chunk is converted into a 384-dimension vector using the `all-MiniLM-L6-v2` sentence-transformer model (a small, fast, local embedding model — no external API call needed for this step). All vectors are loaded into a **FAISS** index, which supports fast nearest-neighbor similarity search. The whole index (chunks + vectors + metadata) is pickled to `faiss_store_openai.pkl` so it persists between app reruns without needing to re-scrape and re-embed the same URLs.

### Stage 4 — Retrieve
When the user asks a question, the same embedding model encodes the query into a vector, and FAISS returns the chunks whose vectors are closest to it — i.e. the passages most semantically relevant to the question, regardless of exact keyword overlap.

### Stage 5 — Generate (Augmented Answering)
The retrieved chunks are "stuffed" into a prompt template as `{context}`, alongside the user's `{input}` (question). The prompt explicitly constrains the LLM: *answer only from the given context; say "I don't know" if it's not there.* This is what makes the system **retrieval-augmented** rather than a free-form chatbot — the LLM (`gemini-2.5-flash`) is grounded in the fetched articles, not its own training data. Finally, the unique `source` URLs of every chunk used in the answer are collected and displayed under a "Sources" section.

### Architecture diagram (data flow)
URLs → UnstructuredURLLoader → raw Documents
→ RecursiveCharacterTextSplitter → chunks
→ HuggingFaceEmbeddings → vectors
→ FAISS.from_documents() → vector index (saved to .pkl)

Question → embed → FAISS similarity search → top-k relevant chunks
→ ChatPromptTemplate (context + question)
→ ChatGoogleGenerativeAI (gemini-2.5-flash) → answer
→ collect chunk.metadata['source'] → display sources

---

## 4. Results

Functionally, the app delivers:

- **Grounded Q&A:** answers are constrained to the content of the processed URLs, not general model knowledge — reducing hallucination risk versus a plain chatbot.
- **Source attribution:** every answer is accompanied by the exact article URL(s) the answer was derived from, so the user can verify it — a key requirement for financial/news research tools.
- **Reusable index:** because the FAISS index is persisted to disk (`faiss_store_openai.pkl`), the same set of articles can be queried repeatedly without re-processing the URLs each time.
- **Fast, local embeddings:** using a HuggingFace sentence-transformer instead of a paid embedding API keeps the ingestion step free and fast, while only the final generation step calls a paid LLM (Gemini).
- **Simple, low-friction UI:** a 3-URL sidebar input plus a single question box is enough to go from "raw articles" to "verified answer" in two clicks.

*(No formal quantitative evaluation — e.g. answer accuracy or retrieval precision — was run in this version; results are qualitative/functional.)*

---

## 5. Conclusions

1. **RAG meaningfully reduces hallucination risk** for domain-specific Q&A: by forcing the LLM to answer strictly from retrieved context and explicitly instructing it to say "I don't know" otherwise, the system avoids confidently fabricating facts not present in the source articles.
2. **Chunking strategy matters as much as the model.** The choice of `RecursiveCharacterTextSplitter` with a 200-character overlap directly affects retrieval quality — too-small chunks lose context, too-large chunks dilute the embedding's specificity and blow past the LLM's effective context window.
3. **Local embeddings + hosted LLM is a cost-effective split.** Running `all-MiniLM-L6-v2` locally for embeddings avoids per-token embedding API costs and keeps ingestion offline-capable, while only the (comparatively rarer) generation step needs a hosted LLM call.
4. **Persisting the vector index (pickle to disk) is a simple but effective caching layer** — it avoids redundant re-scraping/re-embedding of the same URLs across app sessions, at the cost of needing to invalidate/rebuild the index if source articles change.
5. **Source citation is essential for trust** in a research tool — returning the answer alone, without the originating URL(s), would make the tool far less useful for verification-sensitive use cases like financial news research.

### Limitations / possible improvements
- `UnstructuredURLLoader` can fail or return low-quality text on JS-heavy or paywalled sites; a more robust scraper (e.g. Playwright-based) would improve ingestion reliability.
- The FAISS index is a single flat in-memory store per session — it doesn't scale to large article collections or support incremental updates without a full rebuild.
- No retrieval evaluation (e.g. hit rate, answer faithfulness scoring) is currently implemented; adding this would validate whether the chunking/embedding choices are actually optimal.
- The API key handling in `main.py` currently checks `OPENAI_API_KEY` as a fallback/guardrail even though the active LLM is Gemini (`GOOGLE_API_KEY`) — this naming should be reconciled to avoid confusing setup errors.

---

## 6. Repository contents

| File | Description |
|---|---|
| `main.py` | Streamlit RAG application (RockyBot) — URL ingestion, chunking, embedding, FAISS indexing, retrieval-augmented Q&A |
| `gen_ai_project.pdf` / supplementary notebook | Learning/reference material covering the underlying building blocks used in `main.py`: sentence-transformer embeddings, FAISS indexing basics, LangChain document loaders (`TextLoader`, `CSVLoader`, `UnstructuredURLLoader`), and text splitters (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`) |
| `faiss_store_openai.pkl` *(generated at runtime)* | Persisted FAISS vector index built from the processed URLs |

## 7. Tech stack
Python · Streamlit · LangChain (`langchain-community`, `langchain-classic`, `langchain-core`) · LangChain Google GenAI (`gemini-2.5-flash`) · HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) · FAISS · python-dotenv · Pickle

## 8. Setup
```bash
pip install streamlit langchain-community langchain-classic langchain-core \
            langchain-huggingface langchain-google-genai faiss-cpu \
            sentence-transformers unstructured python-dotenv

# .env file:
GOOGLE_API_KEY=your_google_api_key_here

streamlit run main.py
```
