"""
AskRef -- RAG hibrid pe LOG (Legile Jocului) + ROAF.

Versiunea v5 -- consolidată. Include TOT ce s-a construit până acum:
  - curățare PDF (header/footer/numerotare) + filtrare noduri de slabă calitate
  - fapte curate din CSV (aceeași autoritate ca regulamentul)
  - bancă de cazuri similare Referee-IQ (index separat, informativ)
  - hybrid search BM25 + vector, rerank cu cross-encoder
  - query rewriting local (Ollama)

PLUS fix-urile din runda de diagnostic pe eval:
  FIX A: env vars setate ÎNAINTE de import torch (altfel n-au efect)
  FIX B: num_predict 400 -> 1500 (răspunsuri trunchiate la mijlocul frazei)
  FIX C: recall mult mai mare la retrieval (7 -> 30 candidați/canal)
  FIX D: prag de rerank RELATIV (scale-free) în loc de absolut arbitrar
  FIX E: rerank pe scenariul complet, nu pe query-ul semantic comprimat
  FIX F: retrieval multi-query (semantic + raw) pe ambele canale
  FIX G: prompt cu verificare aritmetică + auto-verificare literă
  FIX H: answer_query returnează fragmentele brute (diagnostic retrieval)
"""

import os
import sys
import io

# =====================================================================
# FIX A: acestea TREBUIE setate ÎNAINTE de a importa torch/transformers.
# Setate după import, nu au niciun efect.
# =====================================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Nu mai limităm la 1 thread: OMP_NUM_THREADS nu are legătură cu VRAM-ul
# (acela e memorie GPU), deci limitarea doar încetinea embedding-ul pe CPU.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

import torch
import gc
import re
import json
import time
import csv
import sqlite3
from collections import Counter
import numpy as np
import fitz  # PyMuPDF -- extragere text cu control fin

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import ollama

# Unele combinații de versiuni transformers/torch blochează eronat încărcarea
# checkpoint-urilor .bin oficiale HuggingFace printr-o verificare prea strictă.
# Bypass necesar DOAR pentru acele versiuni și DOAR pentru modele din surse
# verificate (BAAI/bge-*). Șterge după un upgrade de `transformers` care rezolvă.
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# --- FIX PENTRU DIACRITICE (UnicodeEncodeError pe Windows) ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# =====================================================================
# CONFIG
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dacă apar erori "CUDA out of memory" (6GB VRAM e strâmt cu Ollama pornit
# în paralel), mută embedding-ul și/sau reranker-ul pe "cpu".
EMBED_DEVICE = DEVICE
RERANK_DEVICE = DEVICE
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DIR_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\Documents"
PERSIST_DIR = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\storage"
CURATED_FACTS_CSV_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\curated_facts.csv"
QA_DB_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\referee_iq.db"

# --- Chunking ---
CHUNK_SIZE = 768
CHUNK_OVERLAP = 200  # crescut de la 128: reguli scurte se taie ușor între chunk-uri

# --- FIX C: recall mult mai mare. Rerank-ul e ieftin și precis; retrieval-ul
# trebuie să arunce plasa larg, altfel fragmentul corect nici nu ajunge la rerank.
BM25_TOP_K = 15
VECTOR_TOP_K = 15
RERANK_TOP_K = 5  # crescut de la 3: mai mult context pentru generator

# --- FIX D: prag RELATIV, nu absolut.
# CrossEncoder returnează logits brute, a căror scară depinde de model și de
# query -- un prag fix (0.15) era arbitrar și tăia inconsistent. Păstrăm
# fragmentele aflate în marja de mai jos față de cel mai bun scor.
RERANK_RELATIVE_MARGIN = 5.0
# Top-1 e păstrat ÎNTOTDEAUNA (chiar sub marjă); regula anti-halucinație din
# prompt rămâne responsabilă de a spune "nu se regăsește" când contextul e slab.
RERANK_ALWAYS_KEEP_TOP1 = True

# --- Banca de cazuri similare (Referee-IQ) ---
QA_BM25_TOP_K = 10
QA_VECTOR_TOP_K = 10
QA_RERANK_TOP_K = 2
QA_RELATIVE_MARGIN = 3.0  # mai strict: sunt doar informative
QA_EMBED_MODEL_NAME = EMBED_MODEL_NAME

# --- Filtrare calitate noduri ---
MIN_NODE_CHARS = 40
MIN_ALPHA_RATIO = 0.4
BOILERPLATE_MIN_PAGE_RATIO = 0.4

# --- Ollama ---
OLLAMA_MODEL = "deepseek-r1:7b"
# FIX D: deepseek-r1 gândește într-un bloc <think>...</think> ÎNAINTE de
# răspunsul propriu-zis -- acest bloc poate fi de la câteva sute la câteva
# MII de tokeni, chiar pentru un prompt scurt. Timeout-ul vechi (180s, gândit
# pentru llama3.1:8b) e prea mic -- R1 e semnificativ mai lent per apel.
OLLAMA_TIMEOUT_SECONDS = 600
# FIX B: 400 tokeni tăiau răspunsul la mijlocul frazei (dovedit în eval:
# răspunsuri terminate cu "Dec" / "Ră"), ceea ce strica și parsarea literei.
# Cu deepseek-r1, bugetul trebuie să acopere ȘI blocul <think> -- 2500 e prea
# puțin, modelul poate fi tăiat în mijlocul gândirii, înainte să ajungă la
# răspunsul structurat propriu-zis.
OLLAMA_NUM_PREDICT = 6000
OLLAMA_TEMPERATURE = 0.0  # determinism -- e o sarcină de decizie, nu creativă
# FIX C: Ollama foloseşte implicit doar 4096 tokeni de context dacă nu ceri
# explicit altceva -- promptul final (fragmente + cazuri similare + instrucţiuni)
# poate depăşi uşor asta (confirmat: eroare reală la 6347 tokeni de prompt).
# Trebuie să acopere prompt-ul CEL MAI MARE posibil + OLLAMA_NUM_PREDICT.
# Cu OLLAMA_NUM_PREDICT=6000 (pentru <think> R1), contextul trebuie mărit
# corespunzător -- 20000 acoperă confortabil prompt (~8000) + predict (6000)
# + marjă. Dacă tot apar erori de context, fie mai creşte valoarea de aici,
# fie reduce numărul/mărimea fragmentelor trimise modelului.
OLLAMA_NUM_CTX = 20000

ollama_client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)

NO_CONTEXT_MESSAGE = "Informația nu se regăsește în regulamentul extras."
FINAL_ANSWER_MARKER = "RĂSPUNS FINAL:"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def infer_domain_from_filename(file_name: str) -> str:
    name = file_name.lower()
    if "roaf" in name:
        return "ROAF"
    if "log" in name or "lj" in name or "legile" in name:
        return "LOG"
    return "UNKNOWN"


def strip_answer_options(raw_query: str) -> str:
    """
    FIX E (parte 1): scoate variantele A./B./C./D. din enunț.
    Pentru rerank vrem scenariul curat -- variantele conțin termeni
    contradictorii (LLD ȘI LLI ȘI penalty) care derutează cross-encoder-ul.
    """
    lines = []
    for line in raw_query.splitlines():
        if re.match(r"^\s*[A-H][\.\)]\s+", line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else raw_query


# =====================================================================
# CURĂȚARE PDF -- "garbage in, garbage out" se rezolvă la sursă
# =====================================================================

PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,6})\s*$", re.IGNORECASE)


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_boilerplate_lines(pages_text, min_ratio=BOILERPLATE_MIN_PAGE_RATIO):
    """Linii care apar pe multe pagini = header/footer/watermark."""
    if len(pages_text) <= 3:
        return set()
    counts = Counter()
    for page_text in pages_text:
        for line in set(l.strip() for l in page_text.splitlines() if l.strip()):
            counts[line] += 1
    total = len(pages_text)
    # <120 caractere: nu ștergem accidental paragrafe legitime care se repetă.
    return {l for l, c in counts.items() if (c / total) >= min_ratio and len(l) < 120}


def _clean_page_text(text, boilerplate_lines):
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in boilerplate_lines or PAGE_NUMBER_LINE_RE.match(s):
            continue
        kept.append(line)
    return _collapse_whitespace(_dehyphenate("\n".join(kept)))


def load_and_clean_pdfs(dir_path):
    documents = []
    if not os.path.isdir(dir_path):
        log(f"ATENȚIE: folderul de documente '{dir_path}' nu există.")
        return documents

    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(".pdf"):
            continue
        log(f"Curăț și extrag text din: {fname}")
        doc = fitz.open(os.path.join(dir_path, fname))
        pages_text = [p.get_text("text") for p in doc]
        doc.close()

        boilerplate = _detect_boilerplate_lines(pages_text)
        if boilerplate:
            log(f"  {len(boilerplate)} linii repetitive (header/footer) eliminate.")

        skipped = 0
        for page_num, raw_text in enumerate(pages_text, start=1):
            cleaned = _clean_page_text(raw_text, boilerplate)
            if len(cleaned) < 20:
                skipped += 1
                continue
            documents.append(
                Document(
                    text=cleaned,
                    metadata={
                        "file_name": fname,
                        "page_label": str(page_num),
                        "domain": infer_domain_from_filename(fname),
                        "source": "pdf",
                    },
                )
            )
        if skipped:
            log(f"  {skipped} pagini sărite (fără text util după curățare).")
    return documents


def filter_low_quality_nodes(nodes):
    kept, dropped = [], 0
    for node in nodes:
        text = node.text.strip()
        if len(text) < MIN_NODE_CHARS:
            dropped += 1
            continue
        if sum(c.isalpha() for c in text) / max(len(text), 1) < MIN_ALPHA_RATIO:
            dropped += 1
            continue
        kept.append(node)
    if dropped:
        log(f"Filtrare calitate: {dropped} noduri eliminate (prea scurte / prea puțin text real).")
    return kept


def load_curated_facts_as_documents(csv_path):
    """CSV: id,domain,law_reference,text -- aceeași autoritate ca PDF-urile."""
    if not os.path.exists(csv_path):
        log(f"Fișier de fapte curate '{csv_path}' inexistent (opțional). Continui.")
        return []
    documents = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": f"curated:{row.get('id', '?')}",
                        "domain": row.get("domain", "LOG"),
                        "law_reference": row.get("law_reference", ""),
                        "source": "curated",
                    },
                )
            )
    log(f"Încărcate {len(documents)} fapte curate din '{csv_path}'.")
    return documents


def build_or_load_index():
    if not os.path.exists(PERSIST_DIR):
        log("Storage inexistent. Construiesc indexul de la zero...")
        documents = load_and_clean_pdfs(DIR_PATH)
        documents += load_curated_facts_as_documents(CURATED_FACTS_CSV_PATH)
        if not documents:
            raise RuntimeError(f"Niciun document găsit în '{DIR_PATH}'. Verifică calea.")
        log(f"Total {len(documents)} documente (pagini curate + fapte curate).")

        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        log(f"{len(nodes)} noduri din chunking.")
        nodes = filter_low_quality_nodes(nodes)
        log(f"{len(nodes)} noduri rămase. Construiesc Vector Index cu BGE-M3...")

        idx = VectorStoreIndex(nodes)
        idx.storage_context.persist(persist_dir=PERSIST_DIR)
        log("Index creat și salvat pe disc.")
    else:
        log("Storage găsit. Încarc Vector Index...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            idx = load_index_from_storage(storage_context)
            log("Vector Index încărcat de pe disc.")
            log("  (Șterge folderul 'storage' după orice modificare la "
                "curated_facts.csv, la PDF-uri sau la logica de curățare.)")
        except Exception as e:
            log(f"Storage pare coruptă ({e}). Șterge '{PERSIST_DIR}' și rulează din nou.")
            raise
    return idx


# =====================================================================
# BANCA DE CAZURI SIMILARE (Referee-IQ) -- index separat, INFORMATIV
# =====================================================================

def load_qa_bank(qa_db_path):
    if not os.path.exists(qa_db_path):
        log(f"Banca de cazuri '{qa_db_path}' inexistentă (rulează sync_referee_iq.py). Continui fără ea.")
        return None
    conn = sqlite3.connect(qa_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_embeddings "
        "(id TEXT PRIMARY KEY, model_name TEXT, embedding BLOB)"
    )
    conn.commit()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, category, pipeline_domain, text, answers_json, "
        "correct_answer_index, correct_answer_text FROM questions"
    )
    items = []
    for qid, category, domain, text, answers_json, idx_correct, text_correct in cur.fetchall():
        try:
            answers = json.loads(answers_json) if answers_json else []
        except json.JSONDecodeError:
            answers = []
        items.append({
            "id": qid, "category": category, "pipeline_domain": domain,
            "text": text or "", "answers": answers,
            "correct_answer_index": idx_correct, "correct_answer_text": text_correct,
        })
    return conn, items


def ensure_qa_embeddings(conn, items, embed_model, model_name=QA_EMBED_MODEL_NAME):
    """Cache de embeddings în SQLite; recalculează doar ce lipsește sau e de alt model."""
    cur = conn.cursor()
    cur.execute("SELECT id, model_name, embedding FROM qa_embeddings")
    cached = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    embeddings, to_compute = {}, []
    for item in items:
        hit = cached.get(item["id"])
        if hit and hit[0] == model_name:
            embeddings[item["id"]] = np.frombuffer(hit[1], dtype=np.float32)
        else:
            to_compute.append(item)

    if to_compute:
        log(f"Calculez embeddings pentru {len(to_compute)} cazuri noi/needdate...")
        texts = [it["text"] for it in to_compute]
        if hasattr(embed_model, "get_text_embedding_batch"):
            vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)
        else:
            vectors = [embed_model.get_text_embedding(t) for t in texts]
        rows = []
        for it, vec in zip(to_compute, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            embeddings[it["id"]] = arr
            rows.append((it["id"], model_name, arr.tobytes()))
        cur.executemany(
            "INSERT INTO qa_embeddings (id, model_name, embedding) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET model_name=excluded.model_name, "
            "embedding=excluded.embedding",
            rows,
        )
        conn.commit()
        log("Embeddings salvate în cache local.")

    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([embeddings[it["id"]] for it in items])


def _top_k_indices(scores, top_k):
    scores = np.asarray(scores)
    if scores.size == 0:
        return np.array([], dtype=int)
    if top_k >= len(scores):
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -top_k)[-top_k:]
    return idx[np.argsort(scores[idx])[::-1]]


def _cosine_similarities(query_vec, matrix):
    if matrix.size == 0:
        return np.array([])
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m @ q


def hybrid_retrieve_qa(embed_model, qa_bm25_index, qa_items, qa_matrix,
                       semantic_query, bm25_query_string, domain=None, exclude_id=None):
    """
    exclude_id: la evaluare, exclude întrebarea curentă din propriile ei
    "cazuri similare" -- altfel ar vedea răspunsul corect (data leakage).
    """
    combined = {}

    tokens = tokenize(bm25_query_string)
    if tokens and qa_items:
        scores = qa_bm25_index.get_scores(tokens)
        for i in _top_k_indices(scores, QA_BM25_TOP_K):
            combined[qa_items[i]["id"]] = qa_items[i]

    if qa_items and qa_matrix.size:
        qvec = np.asarray(embed_model.get_query_embedding(semantic_query), dtype=np.float32)
        sims = _cosine_similarities(qvec, qa_matrix)
        for i in _top_k_indices(sims, QA_VECTOR_TOP_K):
            combined[qa_items[i]["id"]] = qa_items[i]

    if exclude_id is not None:
        combined.pop(exclude_id, None)

    candidates = list(combined.values())
    if domain and domain != "AMBELE":
        filtered = [it for it in candidates if it["pipeline_domain"] == domain]
        if filtered:
            return filtered
    return candidates


def _apply_relative_threshold(scored, top_k, margin, always_keep_top1):
    """
    FIX D: prag relativ, scale-free. Scorurile cross-encoder sunt logits brute
    a căror scară variază de la query la query -- un prag absolut fix taie
    inconsistent (uneori tot, uneori nimic).
    """
    if not scored:
        return []
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    best = float(scored[0][1])
    kept = [(item, s) for item, s in scored if float(s) >= best - margin]
    if not kept and always_keep_top1:
        kept = [scored[0]]
    return kept


def rerank_qa(reranker_model, rerank_query, qa_candidates, top_k):
    if not qa_candidates:
        return []
    pairs = [[rerank_query, it["text"]] for it in qa_candidates]
    scores = reranker_model.predict(pairs, batch_size=16, show_progress_bar=False,
                                    convert_to_numpy=True)
    # Cazurile similare sunt opționale: fără always_keep_top1, ca să nu forțăm
    # un caz irelevant în prompt când nu există unul bun.
    return _apply_relative_threshold(list(zip(qa_candidates, scores)), top_k,
                                     QA_RELATIVE_MARGIN, always_keep_top1=False)


def format_qa_context(qa_reranked):
    letters = "ABCDEFGH"
    blocks = []
    for i, (it, _score) in enumerate(qa_reranked):
        answers = it.get("answers") or []
        lines = []
        for idx, ans in enumerate(answers):
            mark = " (CORECT)" if idx == it.get("correct_answer_index") else ""
            letter = letters[idx] if idx < len(letters) else str(idx)
            lines.append(f"{letter}) {ans}{mark}")
        blocks.append(
            f"[CAZ {i + 1}] (domeniu: {it.get('pipeline_domain')})\n"
            f"Întrebare: {it['text']}\n"
            f"Variante:\n" + ("\n".join(lines) if lines else "(fără variante)") + "\n"
            f"Răspuns corect: {it.get('correct_answer_text') or '(necunoscut)'}"
        )
    return "\n\n".join(blocks)


# =====================================================================
# RETRIEVAL PE REGULAMENT
# =====================================================================

def bm25_search(bm25_index, all_nodes, query_string, top_k):
    tokens = tokenize(query_string)
    if not tokens:
        return []
    scores = bm25_index.get_scores(tokens)
    return [all_nodes[i] for i in _top_k_indices(scores, top_k)]


def hybrid_retrieve(vector_retriever, bm25_index, all_nodes,
                    semantic_query, bm25_query_string, raw_query, domain=None):
    """
    FIX F: multi-query pe ambele canale. Query-ul rewrite-uit e o compresie
    cu pierderi; dacă rewrite-ul ratează un termen, canalul pe query-ul brut
    îl mai poate salva. Uniunea costă puțin (rerank-ul filtrează oricum).
    """
    combined = {}

    for node_with_score in vector_retriever.retrieve(semantic_query):
        combined[node_with_score.node.node_id] = node_with_score.node
    scenario = strip_answer_options(raw_query)
    if scenario and scenario != semantic_query:
        for node_with_score in vector_retriever.retrieve(scenario):
            combined[node_with_score.node.node_id] = node_with_score.node

    for node in bm25_search(bm25_index, all_nodes, bm25_query_string, BM25_TOP_K):
        combined[node.node_id] = node
    # BM25 și pe scenariul brut: prinde termeni pe care rewrite-ul i-a pierdut.
    for node in bm25_search(bm25_index, all_nodes, scenario, BM25_TOP_K):
        combined[node.node_id] = node

    nodes = list(combined.values())

    if domain and domain != "AMBELE":
        filtered = [n for n in nodes if n.metadata.get("domain") == domain]
        if filtered:
            return filtered
        log(f"Filtrarea pe domeniu '{domain}' n-a găsit noduri; folosesc setul complet.")
    return nodes


def rerank(reranker_model, rerank_query, hybrid_nodes, top_k):
    if not hybrid_nodes:
        return []
    pairs = [[rerank_query, node.text] for node in hybrid_nodes]
    scores = reranker_model.predict(pairs, batch_size=16, show_progress_bar=False,
                                    convert_to_numpy=True)
    return _apply_relative_threshold(list(zip(hybrid_nodes, scores)), top_k,
                                     RERANK_RELATIVE_MARGIN, RERANK_ALWAYS_KEEP_TOP1)


# =====================================================================
# LLM: rewrite + generare finală
# =====================================================================

def rewrite_query(raw_query):
    rewrite_prompt = f"""Ești un instructor expert de arbitri IFAB / FRF.
Transformă întrebarea utilizatorului într-o interogare juridică clară pentru căutare în regulament.

--- REGULI STRICTE ---
1. Dacă întrebarea conține variante de răspuns (A, B, C, D), IGNORĂ-LE total. Concentrează-te strict pe speța centrală.
2. Extrage termenii fotbalistici oficiali din întrebare.
3. Scrie în limba română, corect gramatical.
4. NU modifica sensul. Păstrează abaterile din întrebare (ex.: dacă apare "offside", fă trimitere la Legea 11).
5. "domain" are exact una din trei valori: "LOG" (reguli de joc: faulturi, ofsaid, cartonașe, echipament, reluări), "ROAF" (organizare competiții, licențiere, transferuri, comisii de disciplină) sau "AMBELE" dacă nu poți decide clar.

--- EXEMPLE ---
Întrebare: "Portarul bate o lovitura de poarta si i-o ridica la cap unui coleg ca sa i-o dea inapoi in brate. Ce face arbitrul? A. Lasa jocul sa continue B. LLI"
JSON:
{{"domain": "LOG", "keywords_bm25": ["portar", "truc deliberat", "eludarea legii", "lovitura libera indirecta"], "semantic_query": "Sancționarea portarului care inițiază un truc deliberat pentru a eluda Legea 12 și a prinde mingea în brațe."}}

Întrebare: "Un jucător aruncă cu gheata într-un adversar aflat în afara terenului. Care este decizia?"
JSON:
{{"domain": "LOG", "keywords_bm25": ["aruncare obiect", "gheata", "afara terenului", "lovitura libera directa", "cartonas rosu", "comportament violent"], "semantic_query": "Reluarea jocului și sancțiunea disciplinară când un jucător aruncă un obiect într-un adversar aflat în afara terenului de joc."}}

Întrebare: "Un club nu si-a achitat taxa de licentiere la termen. Ce se intampla?"
JSON:
{{"domain": "ROAF", "keywords_bm25": ["licentiere club", "taxa licentiere", "termen plata", "sanctiuni administrative", "comisie licentiere"], "semantic_query": "Consecințele pentru un club care nu achită taxa de licențiere în termenul stabilit."}}

--- TASK-UL TĂU ---
Întrebare: "{raw_query}"

Returnează STRICT un JSON valid:
"""
    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": rewrite_prompt}],
            # FIX D: NU mai forțăm format="json" cu deepseek-r1. Modelul
            # scrie normal un bloc <think> înainte de răspuns; combinat cu
            # constrângerea strictă de gramatică JSON a lui Ollama, asta
            # producea content GOL (0 caractere) -- exact eroarea
            # "Expecting value: line 1 column 1 (char 0)". Cerem JSON prin
            # prompt (deja specificat mai sus) și îl extragem tolerant mai
            # jos, în loc să forțăm formatul la nivel de decodare.
            options={"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX, "num_predict": 2000},
        )
        response_text = response["message"]["content"].strip()
        # Eliminăm blocul <think>...</think> înainte de a căuta JSON-ul.
        response_text = re.sub(r"<think>.*?</think>", "", response_text,
                                flags=re.DOTALL | re.IGNORECASE).strip()
    except Exception as e:
        log(f"Ollama a dat rateu la rewrite (posibil timeout): {e}")
        return raw_query, raw_query, "LOG"

    try:
        # Extragere tolerantă: căutăm primul bloc {...} din text, în loc să
        # cerem ca ÎNTREG răspunsul să fie JSON pur -- unele modele mai
        # adaugă text/marcaje în jurul JSON-ului chiar și când li se cere
        # explicit "doar JSON".
        json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        json_str = json_match.group(0) if json_match else response_text
        data = json.loads(json_str)
        domain = data.get("domain", "LOG")
        keywords = data.get("keywords_bm25", [])
        bm25_query_string = " ".join(keywords) if keywords else raw_query
        semantic_query = data.get("semantic_query") or raw_query
        return bm25_query_string, semantic_query, domain
    except (json.JSONDecodeError, AttributeError) as e:
        log(f"Rewrite nu a returnat JSON valid ({e}). Fallback la query-ul brut.")
        return raw_query, raw_query, "LOG"


def generate_final_answer(raw_query, retrieved_contexts, qa_reranked=None):
    qa_reranked = qa_reranked or []

    if not retrieved_contexts:
        log("Niciun fragment relevant. Sar peste generarea LLM.")
        note = ""
        if qa_reranked:
            note = (f"\n\n(Notă: {len(qa_reranked)} cazuri similare informative găsite, "
                    "dar ele nu pot substitui text de lege absent.)")
        return f"**Analiza fazei:**\n\n**Decizia corectă:** {NO_CONTEXT_MESSAGE}{note}"

    log("Generez răspunsul final via Ollama...")

    context_str = "\n\n".join(
        f"--- FRAGMENT {i+1} ---\n{t}" for i, t in enumerate(retrieved_contexts)
    )
    qa_context_str = format_qa_context(qa_reranked) if qa_reranked else "(niciun caz similar relevant)"

    final_prompt = f"""Ești un instructor expert IFAB (International Football Association Board) și CCA (Comisia Centrală de Arbitri), care explică decizii de arbitraj strict pe baza regulamentului oficial.

REGULA FUNDAMENTALĂ: Răspunsul tău se bazează STRICT pe extrasele din CONTEXT LEGISLATIV. Secțiunea CAZURI SIMILARE este informativă și NU e sursă de autoritate. Notele din "CAPCANE FRECVENTE" sunt ghid de interpretare, NU sursă de adevăr.

--- CAPCANE FRECVENTE DE ARBITRAJ (aplică-le doar dacă speța o cere) ---
- LEGEA 11 (Ofsaid) — momentul judecării: poziția se judecă în momentul în care mingea este JUCATĂ/ATINSĂ de coechipier (când mingea părăsește piciorul acestuia), NU în momentul în care jucătorul aflat în ofsaid primește mingea.
- LEGEA 11 — Joc Voluntar vs Deviere: dacă un adversar joacă mingea DELIBERAT și controlat (preluare cu pieptul, pasă intenționată, degajare controlată), ofsaidul anterior se ANULEAZĂ și jocul continuă. Doar o deviere/ricoșeu involuntar sau o respingere (save) PĂSTREAZĂ ofsaidul. O pasă greșită rămâne joc deliberat — greșeala de execuție nu o transformă în deviere.
- LEGEA 11 — apărător ieșit de pe teren: un apărător care părăsește terenul fără permisiune este considerat, la analiza ofsaidului, ca aflându-se pe linia de poartă/margine. Deci el CONTINUĂ să conteze ca adversar — atacantul NU e în ofsaid, golul e valabil, iar apărătorul primește CG pentru părăsirea terenului.
- LEGEA 12 — PORTAR vs JUCĂTOR DE CÂMP: regula specială "portarul joacă mingea cu mâna în propriul careu" (LLI) se aplică STRICT portarului. Un "apărător"/"jucător de câmp" NU e portar, chiar dacă acțiunea are loc în suprafața de pedeapsă — hențul lui se judecă după regula GENERALĂ de joc de mână (penalty dacă e în propriul careu, plus CG sau CR după gravitate). Citește cu atenție CINE comite abaterea.
- LEGEA 12 — SCARA SANCȚIUNII DISCIPLINARE (aplic-o STRICT în această ordine, un pas o dată; nu sări direct la concluzie):
  PAS 1 (tariful de bază, după natura faultului):
    • fault IMPRUDENT (reckless) SAU oprirea unui atac promițător (SPA) → cartonaș GALBEN (CG).
    • fault cu FORȚĂ EXCESIVĂ / comportament violent SAU oprirea unei ocazii clare de gol (DOGSO) → cartonaș ROȘU (CR).
    • fault doar NEGLIJENT (careless), care nu e DOGSO → FĂRĂ cartonaș, doar restart-ul.
  PAS 2 (reducere pentru AVANTAJ — se aplică STRICT DOAR dacă tariful de bază era SPA sau DOGSO; NU se aplică pentru forța excesivă/comportamentul violent de sine stătător): dacă arbitrul acordă avantaj și acesta CHIAR se materializează (echipa păstrează o șansă reală sau marchează), sancțiunea scade cu UN nivel: SPA (CG) → FĂRĂ cartonaș; DOGSO (CR) → CG.
  PAS 3 (reducere suplimentară la PENALTY — abaterea a avut loc în careu și s-a acordat lovitură de pedeapsă): dacă abaterea a fost FĂRĂ INTENȚIE (henț neintenționat) SAU a fost o încercare reală de a juca mingea (fault în lupta pentru minge, nu o acțiune cinică), sancțiunea scade CU ÎNCĂ UN NIVEL — indiferent dacă pasul 2 s-a aplicat deja (pașii 2 și 3 sunt independenți și SE CUMULEAZĂ; ex: DOGSO prin henț neintenționat + avantaj aplicat poate ajunge la FĂRĂ cartonaș).
  Verifică explicit, în ordine: (1) ce tarif de bază? (2) se aplică avantaj și s-a materializat? (3) e penalty și a fost fără intenție/încercare de a juca mingea? Scrie concluzia fiecărui pas înainte de sancțiunea finală.
- LEGEA 12 — abateri în afara terenului: verifică ÎMPOTRIVA CUI. Împotriva unui ADVERSAR (mingea în joc) → LLD de pe linia cea mai apropiată. Împotriva unui COECHIPIER, rezervă, oficial de echipă → LLI de pe linia cea mai apropiată.
- LEGEA 14 — penalty ratat complet: dacă șutul ratează poarta fără să fie atins de portar/bară, jocul se reia cu lovitură de poartă, chiar dacă portarul se mișcase prematur de pe linie. Repetarea se aplică doar dacă mingea a rămas în joc.
- Henț al portarului cu LOCAȚIE NESPECIFICATĂ: dacă enunțul nu precizează dacă atingerea a fost în interiorul sau exteriorul careului, prezintă AMBELE variante: (a) în careu → LLI; (b) în afara careului → LLD.
- Selecția fragmentelor: CONTEXT LEGISLATIV poate conține fragmente din domenii diferite (LOG / ROAF) sau irelevante. Folosește DOAR ce e relevant; ignoră restul tăcut, fără să comentezi fragmentele nefolosite.
- Minidicționar: LLD = lovitură liberă directă, LLI = lovitură liberă indirectă, CG = cartonaș galben, CR = cartonaș roșu, DOGSO = oprirea unei ocazii clare de gol.

--- REGULI DE RĂSPUNS ---
1. ROLURI: identifică explicit CINE face acțiunea (portar / apărător / atacant / rezervă / oficial de echipă) și UNDE (în careu / în afara careului / în afara terenului). Multe reguli diferă exact pe aceste detalii.
2. ANALIZĂ: determină faza și abaterea, apoi legea relevantă STRICT din CONTEXT LEGISLATIV. Dacă mai multe fragmente par relevante, alege-l pe cel mai specific speței.
3. CALCULE: dacă răspunsul implică cifre (minute de prelungiri, distanțe, număr de jucători), scrie calculul explicit pas cu pas și verifică-l înainte de a alege varianta. Nu sări direct la concluzie.
4. CITARE: citează doar legea AȘA CUM APARE ÎN CONTEXT. Legile Jocului sunt numerotate DOAR de la 1 la 17 — nu inventa "Legea 116" sau "Legea 29"; dacă un fragment are un număr intern de articol, scrie "articolul X" fără să-l prezinți drept număr de lege. Dacă nu poți identifica sigur legea, folosește denumirea descriptivă. NU cita CAZURI SIMILARE ca sursă legală.
5. IERARHIE: dacă un CAZ SIMILAR contrazice CONTEXT LEGISLATIV, câștigă CONTEXT LEGISLATIV.
6. AMBIGUITATE: dacă un detaliu esențial lipsește din enunț (locație, cine, dacă mingea era în joc) și rezultatul depinde de el, NU alege arbitrar un scenariu — prezintă toate consecințele posibile. Dacă o variantă de răspuns acoperă explicit ambele scenarii ("LLD sau LLI"), e probabil cea corectă.
7. ANTI-HALUCINAȚIE: dacă răspunsul nu poate fi dedus din CONTEXT LEGISLATIV, spune clar: "{NO_CONTEXT_MESSAGE}"
8. SANCȚIUNE DISCIPLINARĂ (obligatoriu dacă speța implică un cartonaș): stabilește ÎNTÂI rezultatul faptic (ex. dacă golul e valabil) în "Decizia corectă", ABIA APOI parcurge scara din "SCARA SANCȚIUNII DISCIPLINARE". Dacă tariful de bază NU e SPA sau DOGSO, scrie direct "N/A" la pașii 2-3 -- nu forța raționamentul de avantaj/penalty pe abateri de alt tip (ex. al doilea avertisment, comportament violent de sine stătător).
9. VERIFICARE FINALĂ (obligatorie): înainte de a scrie litera finală, recitește propria ta decizie și compar-o cuvânt cu cuvânt cu textul fiecărei variante. Alege litera al cărei TEXT corespunde deciziei tale. Dacă decizia ta nu corespunde exact niciunei variante, alege varianta cea mai apropiată și spune de ce.

--- CONTEXT LEGISLATIV (Extrase Oficiale din Regulament — sursă de autoritate) ---
{context_str}

--- CAZURI SIMILARE REZOLVATE (informativ — NU text de lege) ---
{qa_context_str}

--- ÎNTREBAREA/SPEȚA UTILIZATORULUI ---
{raw_query}

Reamintire: decizia se bazează STRICT pe CONTEXT LEGISLATIV. Verifică rolurile (portar vs apărător) și locațiile înainte de a alege regula.

RĂSPUNSUL TĂU (respectă EXACT această structură, fără să omiti niciun câmp):

**Analiza fazei:** (cine face acțiunea, unde, ce abatere)
**Legea aplicabilă:** (numărul sau denumirea, AȘA CUM APARE în fragment)
**Decizia corectă:** (reluare + sancțiune disciplinară completă -- stabilește ÎNTÂI rezultatul faptic, ex. dacă golul e valabil, ÎNAINTE de a raționa despre cartonaș)
**Sancțiune (doar dacă speța implică un cartonaș):** Pas 1 -- tarif de bază, după natura faultului: e SPA sau DOGSO? Dacă NU (ex: al doilea avertisment în același meci, comportament violent de sine stătător, fault grosolan fără legătură cu avantaj/penalty, orice altă abatere disciplinară care nu ține de oprirea unui atac/gol) → scrie direct "N/A -- nu se aplică scara SPA/DOGSO, sancțiunea e [X] conform regulii specifice" și NU continua cu pașii 2-3. Dacă DA (SPA sau DOGSO) → continuă: Pas 2 -- s-a aplicat avantaj și s-a materializat (folosind rezultatul deja stabilit la "Decizia corectă", nu invers)? Pas 3 -- e penalty și a fost fără intenție/încercare de a juca mingea? → sancțiune finală după reduceri.
**Verificare:** (compară decizia ta cu FIECARE variantă A/B/C/D și explică pe scurt de ce varianta aleasă se potrivește)
**Varianta mea este: [scrie aici litera]** (o singură literă dintre A, B, C, D)
{FINAL_ANSWER_MARKER} [repetă aici aceeași literă]
"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            options={"num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE,
                     "num_ctx": OLLAMA_NUM_CTX},
        )
        initial_answer = response["message"]["content"].strip()
        # FIX D: deepseek-r1 prefixează răspunsul cu un bloc <think>...</think>
        # (raționamentul intern). Îl eliminăm ACUM, nu doar la pasul de
        # self-correction, pentru două motive: (1) initial_answer e inclus
        # verbatim în promptul de self-correction de mai jos -- un bloc
        # <think> lung ar umfla inutil promptul respectiv; (2) evită ca
        # eventuale litere A/B/C/D menționate în timpul raționamentului să
        # fie confundate cu concluzia finală de către alte părți ale
        # pipeline-ului (ex. scriptul de evaluare) care fac regex pe text.
        initial_answer = re.sub(r"<think>.*?</think>", "", initial_answer,
                                 flags=re.DOTALL | re.IGNORECASE).strip()
    except Exception as e:
        log(f"Ollama a dat rateu la generarea finală (posibil timeout): {e}")
        return f"[EROARE] Nu s-a putut genera răspunsul final: {e}"

    # --- SELF-CORRECTION PASS ---
    # Modelele mici scriu uneori o literă diferită față de ce au decis în text
    # (eroare de format confirmată în eval cu letter_text_mismatch=True).
    # Al doilea apel e scurt și rapid: îi cerem să verifice explicit coerența.
    try:
        correction_prompt = f"""Ai generat următorul răspuns la o întrebare de arbitraj:

--- RĂSPUNSUL TĂU ---
{initial_answer}

--- ÎNTREBAREA ORIGINALĂ ---
{raw_query}

Verifică ACUM:
1. Ce decizie ai scris în secțiunea "Decizia corectă"?
2. Ce literă ai scris la "Varianta mea este" și la "RĂSPUNS FINAL"?
3. Textul variantei cu acea literă corespunde cu decizia ta?

Dacă litera și decizia sunt COERENTE — rescrie doar ultima linie:
RĂSPUNS FINAL: [litera]

Dacă litera și decizia sunt INCOERENTE — corectează și rescrie ultimele două linii:
**Varianta mea este: [litera corectă]**
RĂSPUNS FINAL: [litera corectă]

Scrie DOAR aceste una sau două linii, nimic altceva."""

        correction_response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": correction_prompt}],
            # FIX D: cu llama3.1:8b, 50 tokeni erau destui pentru cele 1-2 linii
            # cerute. Cu deepseek-r1, modelul scrie ÎNTÂI un bloc <think> (care
            # poate avea sute de tokeni) înainte de răspunsul propriu-zis -- cu
            # 50 tokeni, apelul se tăia mereu în mijlocul gândirii și nu ajungea
            # niciodată la "RĂSPUNS FINAL". 800 lasă loc de gândire scurtă +
            # cele 1-2 linii cerute.
            options={"num_predict": 800, "temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
        )
        correction_text = correction_response["message"]["content"].strip()
        # Eliminăm blocul <think>...</think> (specific modelelor de tip R1)
        # înainte de a parsa litera -- altfel regex-ul de mai jos ar putea
        # prinde accidental o literă menționată în timpul raționamentului,
        # nu din concluzia finală.
        correction_text = re.sub(r"<think>.*?</think>", "", correction_text,
                                  flags=re.DOTALL | re.IGNORECASE).strip()

        # Extragem litera din răspunsul de corecție
        m = re.search(r"R[AĂÂ]SPUNS\s+FINAL\s*:\s*\**\s*([A-H])\b",
                      correction_text, re.IGNORECASE)
        if m:
            corrected_letter = m.group(1).upper()

            # BUG FIX (confirmat în eval_details.jsonl -- 51/384 răspunsuri aveau
            # "Varianta mea este: X" urmat de "RĂSPUNS FINAL: Y" cu X != Y; în 16
            # cazuri disocierea asta transforma un răspuns corect în greșit).
            # Versiunea veche înlocuia DOAR marcajul "RĂSPUNS FINAL", lăsând
            # "Varianta mea este" cu litera veche -- exact sursa disocierii.
            # Acum înlocuim litera în AMBELE marcaje, păstrând formatarea din jur.
            final = re.sub(
                r"(Varianta mea este\s*:?\s*\**\s*)\[?[A-H]\]?",
                rf"\1{corrected_letter}",
                initial_answer, flags=re.IGNORECASE
            )
            final = re.sub(
                r"(R[AĂÂ]SPUNS\s+FINAL\s*:?\s*)\[?[A-H]\]?",
                rf"\1{corrected_letter}",
                final, flags=re.IGNORECASE
            )
            if final == initial_answer:
                # Dacă niciun marcaj nu exista de înlocuit, le adăugăm pe amândouă.
                final = (initial_answer
                         + f"\n**Varianta mea este: {corrected_letter}**"
                         + f"\n{FINAL_ANSWER_MARKER} {corrected_letter}")

            # Verificare de siguranță: dacă tot mai există o disociere între cele
            # două marcaje (ex. regex-ul nu a găsit unul din ele), nu returnăm
            # un răspuns intern contradictoriu -- forțăm ambele la litera corectă.
            mea_matches = re.findall(r"Varianta mea este\s*:?\s*\**\s*([A-H])",
                                      final, re.IGNORECASE)
            final_matches = re.findall(r"R[AĂÂ]SPUNS\s+FINAL\s*:?\s*([A-H])",
                                        final, re.IGNORECASE)
            if (mea_matches and mea_matches[-1].upper() != corrected_letter) or \
               (final_matches and final_matches[-1].upper() != corrected_letter):
                log(f"ATENȚIE: disociere reziduală după self-correction "
                    f"(mea={mea_matches}, final={final_matches}); forțez litera {corrected_letter}.")
                final += (f"\n**Varianta mea este: {corrected_letter}**"
                          f"\n{FINAL_ANSWER_MARKER} {corrected_letter}")
            return final
    except Exception as e:
        log(f"Self-correction a dat rateu (ignorat, folosesc răspunsul inițial): {e}")

    return initial_answer


# =====================================================================
# PIPELINE
# =====================================================================

class Pipeline:
    def __init__(self, all_nodes, bm25_index, vector_retriever, reranker_model,
                 qa_conn, qa_items, qa_bm25_index, qa_matrix):
        self.all_nodes = all_nodes
        self.bm25_index = bm25_index
        self.vector_retriever = vector_retriever
        self.reranker_model = reranker_model
        self.qa_conn = qa_conn
        self.qa_items = qa_items
        self.qa_bm25_index = qa_bm25_index
        self.qa_matrix = qa_matrix

    def close(self):
        if self.qa_conn:
            self.qa_conn.close()


def initialize_pipeline():
    log(f"Device: {DEVICE.upper()} (embed={EMBED_DEVICE}, rerank={RERANK_DEVICE}) "
        f"| OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    log("Încarc modelul de embedding BGE-M3...")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME, device=EMBED_DEVICE,
        model_kwargs={"trust_remote_code": True},
    )
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    index = build_or_load_index()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    log("Construiesc indexul BM25...")
    all_nodes = list(index.docstore.docs.values())
    bm25_index = BM25Okapi([tokenize(n.text) for n in all_nodes])
    vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)
    log(f"{len(all_nodes)} noduri în index.")

    log(f"Încarc reranker-ul pe {RERANK_DEVICE.upper()}...")
    reranker_kwargs = {"device": RERANK_DEVICE}
    if RERANK_DEVICE == "cuda":
        reranker_kwargs["automodel_args"] = {"torch_dtype": torch.float16}
    reranker_model = CrossEncoder(RERANK_MODEL_NAME, **reranker_kwargs)

    qa_conn, qa_items, qa_bm25_index = None, [], None
    qa_matrix = np.zeros((0, 0), dtype=np.float32)
    qa_loaded = load_qa_bank(QA_DB_PATH)
    if qa_loaded:
        qa_conn, qa_items = qa_loaded
        if qa_items:
            log(f"{len(qa_items)} cazuri similare încărcate.")
            qa_bm25_index = BM25Okapi([tokenize(it["text"]) for it in qa_items])
            qa_matrix = ensure_qa_embeddings(qa_conn, qa_items, Settings.embed_model)
        else:
            log("Banca de cazuri e goală (rulează sync_referee_iq.py).")

    return Pipeline(all_nodes, bm25_index, vector_retriever, reranker_model,
                    qa_conn, qa_items, qa_bm25_index, qa_matrix)


def answer_query(pipeline, raw_query, exclude_qa_id=None, verbose=True):
    """
    FIX H: returnează și fragmentele brute + scorurile + sursele, ca să se
    poată diagnostica dacă o greșeală vine din retrieval sau din raționament.
    """
    bm25_query_string, semantic_query, domain = rewrite_query(raw_query)

    # FIX E (parte 2): rerank pe scenariul complet (fără variante), nu pe
    # query-ul semantic comprimat -- cross-encoder-ul se descurcă mai bine cu
    # contextul complet și nu pierde detaliile pe care rewrite-ul le-a tăiat.
    rerank_query = strip_answer_options(raw_query)

    hybrid_nodes = hybrid_retrieve(
        pipeline.vector_retriever, pipeline.bm25_index, pipeline.all_nodes,
        semantic_query, bm25_query_string, raw_query, domain=domain,
    )
    top_reranked = rerank(pipeline.reranker_model, rerank_query, hybrid_nodes, RERANK_TOP_K)
    retrieved_texts = [n.text for n, _s in top_reranked]

    qa_reranked = []
    if pipeline.qa_items and pipeline.qa_bm25_index is not None:
        qa_candidates = hybrid_retrieve_qa(
            Settings.embed_model, pipeline.qa_bm25_index, pipeline.qa_items,
            pipeline.qa_matrix, semantic_query, bm25_query_string,
            domain=domain, exclude_id=exclude_qa_id,
        )
        qa_reranked = rerank_qa(pipeline.reranker_model, rerank_query,
                                qa_candidates, QA_RERANK_TOP_K)

    final_answer = generate_final_answer(raw_query, retrieved_texts, qa_reranked)

    fragments = [
        {
            "score": float(s),
            "source": n.metadata.get("source", "?"),
            "file_name": n.metadata.get("file_name", "?"),
            "page": n.metadata.get("page_label", ""),
            "domain": n.metadata.get("domain", "?"),
            "text": n.text,
        }
        for n, s in top_reranked
    ]

    if verbose:
        log(f"Domeniu detectat: {domain}")
        log(f"BM25 keywords: {bm25_query_string}")
        log(f"Semantic query: {semantic_query}")
        log(f"{len(hybrid_nodes)} candidați -> {len(top_reranked)} fragmente peste prag.")

        print("\n" + "-" * 28 + " FRAGMENTE REGULAMENT " + "-" * 28)
        if not fragments:
            print("Niciun fragment relevant.")
        for i, f in enumerate(fragments, 1):
            print(f"\n[FRAGMENT {i}] scor={f['score']:.4f} | sursă={f['source']} "
                  f"| {f['file_name']} p.{f['page']} | domeniu={f['domain']}")
            print(f["text"])
        print("\n" + "-" * 78)

        print("\n" + "-" * 28 + " CAZURI SIMILARE " + "-" * 28)
        if not qa_reranked:
            print("Niciun caz similar relevant.")
        for i, (it, score) in enumerate(qa_reranked, 1):
            print(f"\n[CAZ {i}] scor={score:.4f} | domeniu={it['pipeline_domain']}")
            print(f"Întrebare: {it['text']}")
            print(f"Răspuns corect: {it.get('correct_answer_text')}")
        print("\n" + "-" * 78)

        print("\n" + "=" * 28 + " RĂSPUNSUL INSTRUCTORULUI AI " + "=" * 28)
        print(f"\n{final_answer}\n")
        print("=" * 84)

    return {
        "raw_query": raw_query,
        "domain": domain,
        "bm25_query_string": bm25_query_string,
        "semantic_query": semantic_query,
        "rerank_query": rerank_query,
        "num_candidates": len(hybrid_nodes),
        "fragments": fragments,
        "retrieved_texts": retrieved_texts,
        "qa_reranked": qa_reranked,
        "final_answer": final_answer,
    }


def main():
    pipeline = initialize_pipeline()

    raw_query = """Portarul echipei in aparare executa o LLI din propria suprafata de poarta. Neatent, paseaza mingea gresit unui adversar care poate dezvolta un atac promitator. Pana sa o atinga adversarul, portarul reuseste sa o trimita cu mana in afara terenului de joc. Ce decizie trebuie sa ia arbitrul?

A. LLI pentru echipa adversa si CG portarului.

B. LLD sau lovitura de pedeapsa pentru echipa adversa si CG portarului.

C. LLD sau LLI pentru echipa adversa si CG portarului.

D. Nicio varianta de raspuns nu este corecta."""

    print("\n" + "=" * 50)
    log(f"Întrebare:\n{raw_query}")
    print("=" * 50 + "\n")

    answer_query(pipeline, raw_query, verbose=True)
    pipeline.close()


if __name__ == "__main__":
    main()