# RAG on Legal Documents

A local, free-to-run RAG pipeline for question-answering over legal contracts (NDAs, merger agreements, annotated clauses, privacy policies) — built with LangChain, Chroma, and Ollama.

## Objective

Build a full RAG system — document loading, cleaning, exploratory analysis, chunking, embedding, vector storage, retrieval, and generation — to answer natural-language questions over a corpus of legal contracts, and evaluate the system's retrieval and generation quality using benchmark question-answer pairs.

## Dataset

The corpus (`rag_legal/corpus/`) contains four categories of legal documents as `.txt` files:

| Folder | Contents |
|---|---|
| `contractnli` | Non-disclosure and confidentiality agreements |
| `cuad` | Contracts with annotated legal clauses |
| `maud` | Merger/acquisition contracts and agreements |
| `privacy_qa` | Privacy policy question-answering dataset |

Matching benchmark files (`rag_legal/benchmarks/`) provide ground-truth question-answer pairs with source spans, used for evaluation.

## Pipeline Overview

**1. Data Loading & Preprocessing**
- Loaded all `.txt` files via LangChain's `DirectoryLoader`, with per-file category/filename metadata for traceability.
- Maintained **two cleaned versions** of the corpus:
  - `doc_analysis` — aggressively cleaned (lowercased, stopwords removed via spaCy) for EDA only.
  - `documents` — lightly cleaned (contact info, page-break noise removed; case and grammar preserved) for chunking/embedding, since legal text relies on capitalization and negation (e.g. "shall not," "Confidential Information") for meaning.

**2. Exploratory Data Analysis**
- Document length statistics (avg/min/max).
- Word frequency analysis (top/bottom 20 words, stopwords excluded).
- TF-IDF cosine similarity across documents (first 10 vs. 10 random), visualized as heatmaps.

**3. Chunking**
- `RecursiveCharacterTextSplitter` (chunk size 800, overlap 120, paragraph-aware separators) to preserve clause-level coherence in legal text.

**4. Vector Database**
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2` (free, local, no API key required).
- Stored in a persistent **Chroma** vector database (~85K chunks).

**5. RAG Chain**
- LangChain LCEL pipeline: retriever → prompt (explicit anti-hallucination instruction) → LLM → parsed output.
- Generation via **Ollama (`llama3.2:3b`)**, run fully locally — no API cost or rate limits. Code also includes a HuggingFace Inference API-based alternative (`meta-llama/Llama-3.1-8B-Instruct`), used during initial development before switching to Ollama for unlimited local inference.

**6. Evaluation**
- Benchmark questions/answers extracted from all four JSON files.
- Evaluated using **Ragas** (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) on a sampled subset of questions, due to local CPU inference time constraints.

## Key Findings

- **Retrieval quality was strong** — context precision scored 1.0 on successfully evaluated queries, indicating consistently relevant chunk retrieval.
- **Generation was grounded and cautious** — the model correctly declined to answer when context was insufficient, rather than hallucinating.
- **Local small LLMs (3B) struggled as automated judges** — several Ragas faithfulness computations failed due to inconsistent structured JSON output, and CPU-only inference caused timeouts under Ragas's internal multi-call evaluation, a distinct limitation from the model's adequacy as a generator.

Full discussion in the notebook's Conclusion section.

## Tech Stack

- **Framework:** LangChain (LCEL)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** Chroma
- **LLM:** Ollama (`llama3.2:3b`), local
- **Evaluation:** Ragas
- **Analysis:** pandas, scikit-learn (TF-IDF), spaCy, seaborn/matplotlib

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install and run Ollama locally
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

> Note: OpenAI/HuggingFace API keys are only needed if using those LLM/embedding options instead of the local Ollama + HuggingFace-embeddings setup used by default.

Place the corpus and benchmark files under `rag_legal/` (see Dataset section above) before running the notebook.

## Notebook

See [`RAG_Legal_Docs_Piyush_Sharma.ipynb`](./RAG_Legal_Docs_Piyush_Sharma.ipynb) for the full implementation, analysis, and results.

## Author

**Piyush Sharma**
Executive PG Diploma in Data Science and AI, IIIT Bangalore
