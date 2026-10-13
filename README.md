# AskRef
Hybrid RAG (BM25 + vector search) on the IFAB Laws of the Game and FRF ROAF for football referees and enthusiasts. Runs with Ollama + BGE-M3 + cross-encoder reranking.

# AskRef ⚽

**Hybrid RAG system on the Laws of the Game (IFAB LOG) + ROAF (FRF) for football referee training.**

A question-answering system grounded in the official football rulebook, built with hybrid BM25 + vector search retrieval, cross-encoder reranking, and local inference via Ollama. Designed to run fully offline on consumer hardware.

---

## Architecture

```
User question
      │
      ▼
Query Rewriting  (Ollama llama3.1:8b)
  ├── BM25 keywords
  ├── semantic query
  └── domain detection (LOG / ROAF / BOTH)
      │
      ▼
Hybrid Retrieval  (multi-query)
  ├── BM25 (top-30, on keywords + raw query)
  └── Vector Search BGE-M3 (top-30, on semantic + raw query)
      │
      ▼
Cross-Encoder Rerank  (BGE-reranker-v2-m3)
  └── top-5, relative threshold (scale-free)
      │
      ├── LEGISLATIVE CONTEXT  (cleaned PDFs + curated facts CSV)
      └── SIMILAR CASES        (Referee-IQ bank, separate index, informative only)
      │
      ▼
Final generation  (Ollama llama3.1:8b)
  └── self-correction pass  (verifies chosen letter vs. decision text)
      │
      ▼
Structured answer  (Phase analysis / Law / Decision / Answer letter)
```

---

## Repository structure

| File | Purpose |
|---|---|
| `rag_hybrid_search_v5.py` | Full pipeline (retrieval + generation) |
| `eval_all_questions.py` | Automated evaluation on all Referee-IQ questions |
| `sync_referee_iq.py` | Sync question bank from Referee-IQ API |
| `curated_facts.csv` | Manually curated facts (patches for PDF knowledge gaps) |
| `diagnose_image_pages.py` | Diagnose PDF pages with diagrams and low text extraction |
| `whiteout_pdf_corners.py` | Clean PDF corners (removes headers/footers/page numbers) |

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB | 6 GB (e.g. GTX 1660 Ti) |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores (e.g. Ryzen 7) |
| Storage | 20 GB | 50 GB |

> **Note:** Embedding and reranker run on CPU to keep VRAM free for Ollama. With 6 GB VRAM you can comfortably run `llama3.1:8b` or `deepseek-r1:7b`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/username/askref.git
cd askref
```

### 2. Install Python dependencies

```bash
pip install torch torchvision
pip install llama-index llama-index-embeddings-huggingface
pip install llama-index-readers-file pymupdf
pip install rank-bm25 sentence-transformers
pip install ollama numpy requests
pip install python-dotenv  # optional, for .env support
```

### 3. Install and start Ollama

```bash
# Download from https://ollama.com
ollama pull llama3.1:8b
ollama serve
```

### 4. Configure paths in `rag_hybrid_search_v5.py`

```python
DIR_PATH               = r"path\to\pdf\documents"
PERSIST_DIR            = r"path\to\storage"
CURATED_FACTS_CSV_PATH = r"path\to\curated_facts.csv"
QA_DB_PATH             = r"path\to\referee_iq.db"
```

### 5. Add PDF documents

Place LOG and ROAF PDF files in `DIR_PATH`. Include "LOG" or "ROAF" in the filename (e.g. `LOG_2026-2027.pdf`, `ROAF_FRF.pdf`) — the pipeline detects the domain automatically from the filename.

> **Optional:** Run `whiteout_pdf_corners.py` before indexing to remove headers, footers and page numbers from the PDFs.

### 6. First run (builds the index)

```bash
python rag_hybrid_search_v5.py
```

On the first run the vector index is built automatically and saved to `PERSIST_DIR`. Subsequent runs load it directly in seconds.

---

## Referee-IQ question bank sync

If you have access to the Referee-IQ API:

```bash
# Windows PowerShell
$env:REFEREE_IQ_API_KEY="riq_live_..."

python sync_referee_iq.py
```

Syncs all questions (LJ + ROAF categories) into a local `referee_iq.db` SQLite database. The script is resumable — run it at any time to add new questions without creating duplicates.

---

## Automated evaluation

```bash
python eval_all_questions.py
```

Runs all questions from `referee_iq.db` through the pipeline and compares the model's answer against the correct option. Produces two output files:

- `eval_report.csv` — per-question summary (correct/wrong + failure cause)
- `eval_details.jsonl` — full details including retrieved fragments

**Automatic failure classification:**

| Category | Meaning |
|---|---|
| `RETRIEVAL` | Correct fragment did not reach the context |
| `REASONING` | Context was good, model decided incorrectly |
| `FORMAT` | Correct decision in text, wrong letter at the end |
| `TRUNCATED` | Response cut off by token limit |

> Set `EVAL_LIMIT = 30` for a quick smoke test before running all 384 questions.

---

## Curated facts (curated_facts.csv)

A CSV file for patching knowledge gaps in the source PDFs — rules that are fragmented across pages, buried in diagrams, or lost between chunks. Each row carries the same authority as the official rulebook text.

**Format:**
```csv
id,domain,law_reference,text
CF001,LOG,Law 11,"Offside position is judged at the moment the ball LEAVES the foot of the passer..."
```

**Recommended workflow:**
1. Run the full evaluation
2. Identify `RETRIEVAL` or `REASONING` failures caused by missing context
3. Add a clearly worded row to `curated_facts.csv`
4. Delete the `storage` folder and reindex
5. Re-run evaluation and compare

---

## Key parameters

```python
# Models
EMBED_MODEL_NAME   = "BAAI/bge-m3"                # multilingual embedding
RERANK_MODEL_NAME  = "BAAI/bge-reranker-v2-m3"    # cross-encoder reranker
OLLAMA_MODEL       = "llama3.1:8b"                 # generation + query rewriting

# Chunking
CHUNK_SIZE         = 768
CHUNK_OVERLAP      = 200

# Retrieval
BM25_TOP_K         = 30    # candidates per BM25 channel
VECTOR_TOP_K       = 30    # candidates per vector channel
RERANK_TOP_K       = 5     # final fragments passed to generator

# Generation
OLLAMA_NUM_PREDICT = 2500  # max tokens per response
OLLAMA_TEMPERATURE = 0.0   # full determinism
```

---

## .gitignore

```gitignore
# Vector index (regeneratable)
storage/

# Local question bank
referee_iq.db

# Evaluation reports (large files)
eval_report.csv
eval_details.jsonl

# API keys
.env
*.env

# Python cache
__pycache__/
*.pyc

# PDF source documents (copyright)
Documents/
*.pdf
```

> **Important:** Do not commit the LOG/ROAF PDF files — they are copyrighted by IFAB and FRF respectively.

---

## Roadmap

- [ ] Test `deepseek-r1:7b` and `qwen3:7b` for improved reasoning
- [ ] Systematic rewrite of `curated_facts.csv` covering all 17 Laws
- [ ] FastAPI wrapper for web deployment
- [ ] Replace local generation with external API (Anthropic / OpenAI) for scalability
- [ ] Simple frontend for referees (React or plain HTML)

---

## License

Source code: **MIT**

> **Disclaimer:** The regulatory documents used as source data (IFAB Laws of the Game, FRF ROAF) are copyright of their respective owners and are **not** included in this repository. Commercial use of this system may require explicit approval from IFAB and FRF.

---

## Contributing

Pull requests are welcome, especially:
- New entries in `curated_facts.csv` (rules verified against the official text)
- Prompt improvements for specific refereeing edge cases
- Retrieval optimisations benchmarked against the evaluation set
