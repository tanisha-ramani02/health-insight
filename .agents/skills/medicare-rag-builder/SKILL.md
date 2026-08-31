---
name: medicare-rag-builder
description: Master development skill and step-by-step workflow guide for building, testing, evaluating, and deploying the Medicare RAG Retrieval and Question Answering System with dynamic chunking, Groq primary and Gemini fallback with 3-key round-robin rotation, hybrid search, and FastAPI microservice endpoints.
---

# Medicare RAG Builder — Development & Operational Skill

This skill provides the authoritative engineering playbook, architectural standards, code patterns, and verification protocols for developing and maintaining the **Medicare Intelligent Healthcare Insight & RAG Retrieval Microservice** based on the official CMS *Medicare & You 2025* handbook (`medicare.pdf`).

---

## 1. Project Overview & Mandatory Constraints

### Core Mandates (from Assignment.md):
1. **Dynamic Chunk Sizing**: Chunk size must NOT be statically pre-defined by the user. It must be computed algorithmically based on content complexity, evaluated against strong mathematical metrics (**Shannon Information Entropy**, **Semantic Coherence**, and **Syntactic Boundary Integrity**).
2. **Separation of Concerns**: Document Ingestion, Dynamic Chunking, Vector Storage, Hybrid Retrieval, and LLM Generation must be decoupled modular components.
3. **Structured JSON Output**: Every query response must conform to:
   ```json
   {
     "answer": "...",
     "source_page": 15,
     "confidence_score": 0.94,
     "chunk_size": 284
   }
   ```
4. **Multi-Key Dual-Cloud Provider Architecture**:
   - **Primary Provider**: Groq Cloud (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) with 3 API keys rotating round-robin.
   - **Secondary Fallback**: Google Gemini (`gemini-3.6-flash`, `gemini-3.5-flash-lite`) with 3 API keys rotating round-robin.
   - Priority and chain controlled via `.env`:
     ```ini
     PRIMARY_PROVIDER=groq
     PROVIDER_CHAIN=groq,gemini
     ```
5. **Zero-Downtime Resilience**: 3-key round-robin rotation for both Groq and Gemini with automatic failover on HTTP 429 rate limits.
6. **FastAPI Microservice**: Async REST API with OpenAPI/Swagger documentation at `/docs`.

---

## 2. Environment & Dependency Management (`uv`)

Always use `uv` for virtual environment management and package execution:

```bash
# Initialize project and create Python 3.11 virtual environment
uv init
uv venv --python 3.11

# Sync all project dependencies
uv sync

# Run scripts and commands within the environment
uv run python <script.py>
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
uv run pytest tests/
```

### Required Dependencies in `pyproject.toml`:
- `pymupdf>=1.25.0`: High-speed C++ PDF extraction, page metadata, TOC bookmarks.
- `langchain>=0.3.0`, `langchain-core>=0.3.0`, `langchain-community>=0.3.0`: Standard RAG primitives.
- `langchain-groq>=0.2.0`: Groq LPU ultra-fast open LLM inference (`openai/gpt-oss-120b`).
- `langchain-google-genai>=2.0.0`: Google Gemini chat models (`gemini-3.6-flash`).
- `fastapi>=0.110.0`, `uvicorn>=0.28.0`: Async REST microservice.
- `pydantic>=2.7.0`, `pydantic-settings>=2.2.0`: Strict JSON schemas and environment validation.
- `python-dotenv>=1.0.0`: Environment variable loading.
- `numpy>=1.26.0`: Mathematical entropy, vector similarities, and metric evaluation.

---

## 3. Directory & Package Architecture

Follow this exact modular structure:

```
health-insight/
├── data/
│   ├── medicare.pdf                  # 128-page CMS 2025 Handbook
│   └── cache/                        # Cached vector store index
├── src/
│   ├── __init__.py
│   ├── config.py                     # App settings & key pool configuration
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_loader.py             # PyMuPDF extractor with 1-indexed page mapping
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── dynamic_chunker.py        # Entropy-adaptive dynamic chunking engine
│   │   ├── entropy_metrics.py        # Shannon entropy & semantic coherence calculators
│   │   └── chunk_evaluator.py        # Dynamic vs static evaluation suite
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # Dense vector index
│   │   ├── bm25_retriever.py         # Sparse BM25 keyword index
│   │   └── hybrid_engine.py          # Reciprocal Rank Fusion (RRF)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider_manager.py       # Key rotation & Groq/Gemini failover
│   │   ├── prompt_templates.py       # Grounded system prompts & JSON schema
│   │   └── confidence_scorer.py      # Calibrated confidence scoring engine
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py                 # Pydantic QueryRequest & QueryResponse models
│   └── api/
│       ├── __init__.py
│       ├── app.py                    # FastAPI application & lifespan events
│       └── routes.py                 # Endpoints (/query, /chunk-evaluation, /health)
├── tests/
│   ├── test_dynamic_chunking.py      # Dynamic chunking metric verification
│   ├── test_retrieval.py             # Citation and precision tests
│   ├── test_llm_providers.py         # Key rotation & failover tests
│   └── test_api_endpoints.py         # End-to-end FastAPI test suite
├── understanding/                    # 8 Complete Architecture & Design documents
├── settings.py                       # Pydantic settings loading .env
├── .env                              # Multi-key credentials
├── pyproject.toml                    # UV configuration
└── README.md                         # Run instructions
```

---

## 4. Development Workflow: Step-by-Step

### Step 4.1: Configuration Module (`settings.py` / `src/config.py`)
Load credentials from `.env` with multi-key pool support:
- `GROQ_API_KEY1`, `GROQ_API_KEY2`, `GROQ_API_KEY3`
- `GOOGLE_API_KEY1`, `GOOGLE_API_KEY2`, `GOOGLE_API_KEY3`
- `PRIMARY_PROVIDER=groq`, `PROVIDER_CHAIN=groq,gemini`
- Models: `openai/gpt-oss-120b`, `gemini-3.6-flash`
- Dynamic chunking thresholds: $L_{\min}=120, L_{\max}=650, L_{\text{base}}=300, \alpha=0.35, \beta=0.25$

### Step 4.2: PDF Extraction (`src/ingestion/pdf_loader.py`)
- Open `medicare.pdf` using `pymupdf`.
- Extract text per page while preserving:
  - `page_number`: 1-indexed ($1 \dots 128$)
  - `total_pages`: 128
  - `section_title`: Bookmark from TOC
  - Cleaned text free of duplicate whitespace and broken hyphenations.
- Return a list of LangChain `Document` objects.

### Step 4.3: Dynamic Adaptive Chunking (`src/chunking/`)
Implement the multi-factor chunking algorithm:
1. **Shannon Information Entropy**:
   $$H(W) = -\sum p(w_i) \log_2 p(w_i)$$
   High entropy ($H > 4.5$) produces compact chunks ($150 - 280$ chars); narrative text produces larger chunks ($350 - 600$ chars).
2. **Semantic Coherence**: Cosine distance between adjacent sentence embeddings triggers natural boundary cuts.
3. **Syntactic Boundary Integrity**: Preserve complete sentences, bullet points, and table lines ($B_{\text{comp}} = 1.0$).
4. Compute dynamic size:
   $$\text{ChunkSize}_{\text{dynamic}} = \text{Clamp}\left( L_{\text{base}} \times \left(1 + \alpha \cdot \frac{S_{\text{coh}} - 0.5}{0.5}\right) \times \left(1 - \beta \cdot \frac{H - \mu_H}{\sigma_H}\right), L_{\min}, L_{\max} \right)$$
5. Attach metadata to every chunk: `source_page`, `chunk_size`, `entropy`, `coherence_score`.

### Step 4.4: Hybrid Indexing & Retrieval (`src/retrieval/`)
1. **Dense Vector Store**: Embed chunks into persistent Chroma store.
2. **Sparse BM25 Index**: Build BM25 index over chunk tokens for exact keyword matches (e.g. "Form CMS-10106", "October 15", "Part D").
3. **Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + \text{Rank}_m(d)} \quad (k = 60)$$
4. Return top-$K$ candidate chunks sorted by combined score.

### Step 4.5: Multi-Provider LLM & Key Rotation (`src/llm/provider_manager.py`)
1. **Key Rotation Pool**: Round-robin iterator across configured Groq (primary) and Google (fallback) keys.
2. **Error Interception**: Catch HTTP 429 Too Many Requests, mark key with temporary cooldown, and immediately rotate to next key.
3. **Provider Failover Chain**:
   $$\text{Groq 120B (Key 1..3)} \xrightarrow{429 / Outage} \text{Gemini 3.6 Flash (Key 1..3)}$$
4. **Prompt Construction**: Inject retrieved context chunks with strict system instructions to cite only provided facts and format output in JSON.

### Step 4.6: Confidence Scoring & Schema Enforcement (`src/llm/`)
1. Compute multi-factor calibrated confidence score:
   $$\text{Confidence} = 0.45 \cdot S_{\text{vector}} + 0.25 \cdot S_{\text{bm25}} + 0.30 \cdot S_{\text{grounding}}$$
2. Enforce Pydantic schema: `answer`, `source_page`, `confidence_score`, `chunk_size`.
3. If retrieval similarity $< \tau_{\min} (0.40)$, return structured out-of-scope response without hallucinating.

### Step 4.7: FastAPI Service Layer (`src/api/`)
1. Implement async endpoints:
   - `POST /api/v1/query`: Core query handler returning standard JSON.
   - `POST /api/v1/chunk-evaluation`: Dynamic chunking metrics inspector.
   - `GET /api/v1/health`: Vector store and document status.
   - `GET /api/v1/providers`: Active keys and model health.
2. Add CORS middleware and lifespan startup event to load/warm vector indices.
