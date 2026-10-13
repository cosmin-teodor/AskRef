import os
import sys
import io
import torch
import gc
import re
import json
import time
import sqlite3
import numpy as np
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import ollama

# =====================================================================
# FIX CRITIC: Setate ÎNAINTE de a importa alte module
# =====================================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# FIX #2 (documentat): unele combinații de versiuni transformers/torch blochează
# eronat încărcarea checkpoint-urilor .bin oficiale HuggingFace printr-o verificare
# de siguranță prea strictă (check_torch_load_is_safe). Bypass-ul de mai jos e
# necesar DOAR pentru acele combinații de versiuni și DOAR pentru modele din surse
# oficiale/verificate (BAAI/bge-*, în cazul nostru). Dacă faci upgrade la o versiune
# de `transformers` care rezolvă problema nativ, șterge liniile de mai jos.
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# --- 0. FIX PENTRU DIACRITICE (UnicodeEncodeError) ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# --- 1. CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIX #3: embedding + reranker + Ollama pot concura pe același VRAM.
# Dacă întâlnești erori "CUDA out of memory", schimbă EMBED_DEVICE (și/sau
# RERANK_DEVICE) pe "cpu" — sunt mult mai puțin sensibile la VRAM decât
# modelul de generare rulat prin Ollama.
EMBED_DEVICE = DEVICE
RERANK_DEVICE = DEVICE
EMBED_MODEL_NAME = "BAAI/bge-m3"

DIR_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\Documents"
PERSIST_DIR = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\storage"

BM25_TOP_K = 7
VECTOR_TOP_K = 7
RERANK_TOP_K = 3
# FIX #5: prag minim de relevanță la rerank. Calibrează empiric pe câteva
# speze cunoscute (LOG vs ROAF) — scoruri sub prag sunt excluse din context,
# ca să nu forțăm modelul să răspundă pe baza unor fragmente irelevante.
RERANK_MIN_SCORE = 0.15

# --- CONFIG: BANCA DE CAZURI SIMILARE (Referee-IQ), index SEPARAT de regulament ---
# Populată de sync_referee_iq.py. Aici doar o CITIM, nu apelăm API-ul live.
QA_DB_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\referee_iq.db"
QA_BM25_TOP_K = 5
QA_VECTOR_TOP_K = 5
QA_RERANK_TOP_K = 2
QA_MIN_SCORE = 0.20  # prag ceva mai strict decât la regulament: sunt doar informative
# Trebuie să coincidă cu EMBED_MODEL_NAME de mai sus. Dacă schimbi modelul de
# embedding, schimbă și asta -- cache-ul de embeddings se recalculează automat
# când numele modelului nu mai corespunde.
QA_EMBED_MODEL_NAME = EMBED_MODEL_NAME

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT_SECONDS = 120

# FIX #7: client Ollama dedicat, cu timeout explicit, ca să nu rămână
# scriptul agățat la infinit dacă serviciul/modelul nu răspunde.
ollama_client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)

NO_CONTEXT_MESSAGE = "Informația nu se regăsește în regulamentul extras."


def log(msg):
    """Log cu timestamp pentru urmărirea pașilor în consolă."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def infer_domain_from_filename(file_name: str) -> str:
    """
    FIX #4 (parte 1): deducem domeniul (LOG/ROAF) din numele fișierului sursă.
    Presupune că fișierele PDF conțin "LOG" sau "ROAF" în nume
    (ex: 'LOG_2025.pdf', 'ROAF_FRF.pdf'). Ajustează logica dacă numele reale
    diferă.
    """
    name = file_name.lower()
    if "log" in name:
        return "LOG"
    if "roaf" in name:
        return "ROAF"
    return "UNKNOWN"


def build_or_load_index():
    if not os.path.exists(PERSIST_DIR):
        log("Storage directory not found. Parsing PDFs...")
        pdf_parser = PyMuPDFReader()
        file_extractor = {".pdf": pdf_parser}
        documents = SimpleDirectoryReader(
            input_dir=DIR_PATH, file_extractor=file_extractor
        ).load_data()

        # FIX #4 (parte 2): atașăm metadata de domeniu la fiecare document,
        # ca să poată fi folosită mai târziu pentru filtrare la retrieval.
        for doc in documents:
            fname = doc.metadata.get("file_name", "")
            doc.metadata["domain"] = infer_domain_from_filename(fname)

        log(f"Parsed {len(documents)} pages. Building Vector Index cu BGE-M3...")
        idx = VectorStoreIndex.from_documents(documents)
        idx.storage_context.persist(persist_dir=PERSIST_DIR)
        log("Index created and saved to disk.")
    else:
        log("Storage found. Loading Vector Index...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            idx = load_index_from_storage(storage_context)
            log("Vector Index loaded from disk.")
            # Notă: dacă indexul a fost construit ÎNAINTE de a adăuga metadata
            # de domeniu, nodurile nu vor avea 'domain' setat. Șterge folderul
            # PERSIST_DIR și rulează din nou pentru a beneficia de filtrare pe domeniu.
        except Exception as e:
            log(
                f"Storage pare coruptă ({e}). Șterge folderul '{PERSIST_DIR}' "
                "și rulează din nou."
            )
            raise
    return idx


# =====================================================================
# BANCA DE CAZURI SIMILARE (Referee-IQ) — index SEPARAT, tratat ca sursă
# INFORMATIVĂ, nu ca text de lege. Vezi generate_final_answer() pentru
# cum e izolată de CONTEXT-ul legislativ în prompt.
# =====================================================================

def ensure_qa_schema(conn):
    """Tabel de cache pentru embeddings, separat de tabelul 'questions' scris de sync_referee_iq.py."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_embeddings (
            id TEXT PRIMARY KEY,
            model_name TEXT,
            embedding BLOB
        )
        """
    )
    conn.commit()


def load_qa_bank(qa_db_path):
    """Încarcă toate cazurile din baza locală sincronizată. Returnează None dacă nu există încă."""
    if not os.path.exists(qa_db_path):
        log(
            f"Baza de cazuri similare '{qa_db_path}' nu există. "
            "Rulează sync_referee_iq.py mai întâi. Continui FĂRĂ cazuri similare."
        )
        return None

    conn = sqlite3.connect(qa_db_path)
    ensure_qa_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, category, pipeline_domain, text, answers_json, "
        "correct_answer_index, correct_answer_text FROM questions"
    )
    rows = cur.fetchall()

    items = []
    for qid, category, pipeline_domain, text, answers_json, correct_idx, correct_text in rows:
        try:
            answers = json.loads(answers_json) if answers_json else []
        except json.JSONDecodeError:
            answers = []
        items.append(
            {
                "id": qid,
                "category": category,
                "pipeline_domain": pipeline_domain,
                "text": text or "",
                "answers": answers,
                "correct_answer_index": correct_idx,
                "correct_answer_text": correct_text,
            }
        )
    return conn, items


def build_qa_bm25(items):
    corpus_tokens = [tokenize(it["text"]) for it in items]
    return BM25Okapi(corpus_tokens)


def ensure_qa_embeddings(conn, items, embed_model, model_name=QA_EMBED_MODEL_NAME):
    """
    Calculează embeddings o singură dată per caz și le cache-uiește în SQLite
    (tabelul qa_embeddings). La rulările următoare, doar cazurile noi/needdate
    (sau cele calculate cu alt model) sunt recalculate.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, model_name, embedding FROM qa_embeddings")
    cached = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    embeddings = {}
    to_compute = []
    for item in items:
        qid = item["id"]
        if qid in cached:
            cached_model, blob = cached[qid]
            if cached_model == model_name:
                embeddings[qid] = np.frombuffer(blob, dtype=np.float32)
                continue
        to_compute.append(item)

    if to_compute:
        log(f"Calculez embeddings pentru {len(to_compute)} cazuri noi/needdate din banca Referee-IQ...")
        texts = [it["text"] for it in to_compute]
        if hasattr(embed_model, "get_text_embedding_batch"):
            vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)
        else:
            vectors = [embed_model.get_text_embedding(t) for t in texts]

        rows_to_upsert = []
        for it, vec in zip(to_compute, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            embeddings[it["id"]] = arr
            rows_to_upsert.append((it["id"], model_name, arr.tobytes()))

        cur.executemany(
            """
            INSERT INTO qa_embeddings (id, model_name, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model_name = excluded.model_name,
                embedding = excluded.embedding
            """,
            rows_to_upsert,
        )
        conn.commit()
        log("Embeddings pentru cazuri similare salvate în cache local.")

    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([embeddings[it["id"]] for it in items])


def qa_bm25_search(qa_bm25_index, items, query_string, top_k):
    query_tokens = tokenize(query_string)
    if not query_tokens or not items:
        return []
    scores = np.asarray(qa_bm25_index.get_scores(query_tokens))
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [items[i] for i in top_indices]


def _cosine_similarities(query_vec, matrix):
    if matrix.size == 0:
        return np.array([])
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m_norm @ q_norm


def qa_vector_search(embed_model, query_text, items, qa_matrix, top_k):
    if not items or qa_matrix.size == 0:
        return []
    query_vec = np.asarray(embed_model.get_query_embedding(query_text), dtype=np.float32)
    sims = _cosine_similarities(query_vec, qa_matrix)
    if top_k >= len(sims):
        top_indices = np.argsort(sims)[::-1]
    else:
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    return [items[i] for i in top_indices]


def hybrid_retrieve_qa(
    embed_model, qa_bm25_index, qa_items, qa_matrix,
    semantic_query, bm25_query_string, domain=None, exclude_id=None,
):
    """
    Hybrid BM25 + vector pe banca de cazuri similare, cu filtrare opțională
    pe domeniu (pipeline_domain, deja mapat LJ->LOG de sync_referee_iq.py).

    exclude_id: dacă faci un eval harness care rulează chiar întrebările din
    referee_iq.db prin acest pipeline, exclude ID-ul întrebării curente ca să
    eviți data leakage (răspunsul corect apărând ca "informativ" în contextul
    propriei sale spețe).
    """
    bm25_hits = qa_bm25_search(qa_bm25_index, qa_items, bm25_query_string, QA_BM25_TOP_K)
    vector_hits = qa_vector_search(embed_model, semantic_query, qa_items, qa_matrix, QA_VECTOR_TOP_K)

    combined = {}
    for it in bm25_hits:
        combined[it["id"]] = it
    for it in vector_hits:
        combined[it["id"]] = it

    if exclude_id is not None:
        combined.pop(exclude_id, None)

    candidates = list(combined.values())

    if domain and domain != "AMBELE":
        filtered = [it for it in candidates if it["pipeline_domain"] == domain]
        if filtered:
            return filtered

    return candidates


def rerank_qa(reranker_model, semantic_query, qa_candidates, top_k, min_score=QA_MIN_SCORE):
    if not qa_candidates:
        return []
    pairs = [[semantic_query, it["text"]] for it in qa_candidates]
    scores = reranker_model.predict(
        pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True
    )
    scored = sorted(zip(qa_candidates, scores), key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]
    return [(it, s) for it, s in top_scored if s >= min_score]


def format_qa_context(qa_reranked):
    """Formatează cazurile similare într-un bloc de text separat pentru prompt."""
    letters = "ABCDEFGH"
    blocks = []
    for i, (it, score) in enumerate(qa_reranked):
        answers = it.get("answers") or []
        lettered = []
        for idx, ans in enumerate(answers):
            marker = " (CORECT)" if idx == it.get("correct_answer_index") else ""
            letter = letters[idx] if idx < len(letters) else str(idx)
            lettered.append(f"{letter}) {ans}{marker}")
        variante_str = "\n".join(lettered) if lettered else "(fără variante înregistrate)"
        correct_line = f"Răspuns corect: {it.get('correct_answer_text') or '(necunoscut)'}"
        blocks.append(
            f"[CAZ {i + 1}] (domeniu: {it.get('pipeline_domain')})\n"
            f"Întrebare: {it['text']}\n"
            f"Variante:\n{variante_str}\n"
            f"{correct_line}"
        )
    return "\n\n".join(blocks)


def rewrite_query(raw_query):
    """Query rewriting via Ollama cu Few-Shot Prompting pentru precizie maximă."""
    rewrite_prompt = f"""Ești un instructor expert de arbitri IFAB / FRF. 
Transformă întrebarea utilizatorului într-o interogare juridică clară.

--- REGULI STRICTE ---
1. Dacă întrebarea conține variante de răspuns (A, B, C, D), IGNORĂ-LE total. Concentrează-te strict pe speța centrală.
2. Extrage termenii fotbalistici din întrebare.
3. Vorbește limba română corect gramatical și logic.
4. Nu modifica query ul semantic decat daca este nevoie mare. Păstreaza abaterile din întrebare(ex.: daca apare cuvantul offside trebuie sa faci trimitere la Legea 11).
5. "domain" trebuie să fie una din trei valori: "LOG" (Legile Jocului - reguli aplicate pe teren: faulturi, ofsaid, cartonașe, echipament), "ROAF" (organizare competiții, licențiere, transferuri, comisii de disciplină) sau "AMBELE" dacă nu poți decide clar din întrebare.

--- EXEMPLE ---
Întrebare: "Portarul bate o lovitura de poarta si i-o ridica la cap unui coleg ca sa i-o dea inapoi in brate. Ce face arbitrul? A. Lasa jocul sa continue B. LLI"
JSON:
{{
  "domain": "LOG",
  "keywords_bm25": ["portar", "truc deliberat", "eludarea legii", "lovitura libera indirecta"],
  "semantic_query": "Sancționarea portarului care inițiază un truc deliberat pentru a eluda Legea 12 și a prinde mingea în brațe."
}}

Întrebare: "Un jucător aruncă cu gheata într-un adversar aflat în afara terenului. Care este decizia?"
JSON:
{{
  "domain": "LOG",
  "keywords_bm25": ["aruncare obiect", "gheata", "afara terenului", "lovitura libera directa", "cartonas rosu", "comportament violent"],
  "semantic_query": "Reluarea jocului și sancțiunea disciplinară când un jucător aruncă un obiect într-un adversar aflat în afara terenului de joc."
}}

Întrebare: "Un club nu și-a achitat taxa de licențiere la termenul stabilit de comisie. Ce se întâmplă?"
JSON:
{{
  "domain": "ROAF",
  "keywords_bm25": ["licențiere club", "taxă licențiere", "termen plată", "sancțiuni administrative", "comisie licențiere"],
  "semantic_query": "Consecințele pentru un club care nu achită taxa de licențiere în termenul stabilit de comisia de licențiere."
}}

--- TASK-UL TĂU ---
Întrebare: "{raw_query}"

Returnează STRICT un JSON valid:
"""
    try:
        # FIX #6: temperature=0 la rewriting, pentru determinism maxim
        # (e un pas de extracție/normalizare, nu de generare creativă).
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": rewrite_prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        response_text = response["message"]["content"].strip()
    except Exception as e:
        log(f"Ollama a dat rateu la rewrite (posibil timeout): {e}")
        return raw_query, raw_query, "LOG"

    try:
        query_data = json.loads(response_text)
        domain = query_data.get("domain", "LOG")
        keywords_list = query_data.get("keywords_bm25", [])
        bm25_query_string = " ".join(keywords_list) if keywords_list else raw_query
        semantic_query = query_data.get("semantic_query", raw_query)
        return bm25_query_string, semantic_query, domain
    except json.JSONDecodeError as e:
        log(f"Local LLM nu a returnat JSON valid ({e}). Fallback la query-ul brut.")
        return raw_query, raw_query, "LOG"


def generate_final_answer(raw_query, retrieved_contexts, qa_reranked=None):
    """
    Generează răspunsul final, folosind:
    - CONTEXT LEGISLATIV (retrieved_contexts): sursă de autoritate, din regulament.
    - CAZURI SIMILARE (qa_reranked): informativ, NU text de lege — vezi ierarhia
      explicită din prompt.
    """
    qa_reranked = qa_reranked or []

    # FIX #5 (parte 2): dacă nu a rămas niciun fragment legislativ relevant,
    # nu mai apelăm deloc LLM-ul pentru decizie — cazurile similare NU pot
    # substitui absența textului de lege.
    if not retrieved_contexts:
        log("Niciun fragment relevant găsit în regulament (sub pragul de scor). Sar peste generarea LLM.")
        note = ""
        if qa_reranked:
            note = (
                f"\n\n(Notă: au fost găsite {len(qa_reranked)} cazuri similare informative, "
                "dar acestea nu pot substitui text de lege absent din regulamentul extras.)"
            )
        return f"**Analiza fazei:**\n\n**Decizia corectă:** {NO_CONTEXT_MESSAGE}{note}"

    log("Generating final answer via Ollama...")

    context_str = "\n\n".join(
        [f"--- FRAGMENT {i+1} ---\n{text}" for i, text in enumerate(retrieved_contexts)]
    )
    qa_context_str = format_qa_context(qa_reranked) if qa_reranked else "(niciun caz similar relevant găsit)"

    final_prompt = f"""Ești un instructor expert IFAB (International Football Association Board) și CCA (Comisia Centrală de Arbitri), care explică decizii de arbitraj strict pe baza regulamentului oficial.

REGULA FUNDAMENTALĂ: Răspunsul tău trebuie să se bazeze STRICT pe extrasele din CONTEXT LEGISLATIV de mai jos. Secțiunea CAZURI SIMILARE este informativă și NU este o sursă de autoritate. Notele din "CAPCANE FRECVENTE" sunt doar un ghid de interpretare — NU o sursă de adevăr în sine.

--- CAPCANE FRECVENTE DE ARBITRAJ (ghid de interpretare, aplică-le doar dacă speța o cere) ---
- LEGEA 11 (Offside): distinge clar între "Joc Voluntar / Deliberate Play" (anulează offside-ul; ex: pasă, preluare controlată) și "Deviere/Save" (păstrează offside-ul; ex: ricoșeu involuntar, respingere portar).
- LEGEA 12 (Portarul): dacă portarul joacă mingea cu mâna după o pasă deliberată cu piciorul de la un coechipier, verifică în CONTEXT LEGISLATIV dacă se confirmă Lovitura Liberă Indirectă drept sancțiune.
- Selecția fragmentelor: fragmentele din CONTEXT LEGISLATIV pot proveni din domenii diferite (LOG - reguli de joc, sau ROAF - organizare/disciplină). Folosește DOAR fragmentele relevante pentru natura speței; ignoră restul, fără să le menționezi explicit în răspuns.
- Dacă speța e despre o reluare a jocului (aut, lovitură de poartă, corner), ignoră fragmentele despre penalty (Legea 14), decât dacă a existat clar un fault în suprafața de pedeapsă.
- Atenție la situațiile care menționează că se oprește un atac promițător sau o situație clara de gol. Se acorda advertisment sau eliminiare dacă nu se acordă avantaj sau lovitură de pedeapsă
- Henț al portarului cu mâna, LOCAȚIE NESPECIFICATĂ: dacă enunțul NU precizează explicit dacă portarul a atins mingea în interiorul sau în exteriorul propriei suprafețe de pedeapsă, NU presupune automat o singură variantă. Ia în calcul AMBELE scenarii posibile și menționează-le pe amândouă în decizie: (a) în interiorul careului -> LLI (Legea 12, joc de mână al portarului în propriul careu); (b) în exteriorul careului -> LLD (henț tratat ca la orice jucător de câmp).
- Minidicționar: LLD = lovitură liberă directă, LLI = lovitură liberă indirectă, CG = cartonaș galben, CR = cartonaș roșu

--- REGULI DE RĂSPUNS ---
1. ANALIZĂ: identifică faza de joc și abaterea, apoi determină LEGEA/articolul relevant STRICT din fragmentele din CONTEXT LEGISLATIV. Dacă mai multe fragmente par relevante, prioritizează cel mai specific speței.
2. FORMAT: dacă întrebarea are variante (A, B, C, D), identifică litera corectă explicit într-un câmp separat. Dacă nu are variante, explică decizia în 2-4 propoziții.
3. CITARE: citează doar numărul/numele legii AȘA CUM APARE ÎN CONTEXT LEGISLATIV (ex: "Conform Legii 12..."). NU cita sau invoca un articol care nu apare explicit acolo. NU cita niciodată CAZURI SIMILARE ca sursă legală.
4. IERARHIE: dacă un CAZ SIMILAR pare să contrazică CONTEXT LEGISLATIV, CONTEXT LEGISLATIV are ÎNTOTDEAUNA prioritate. Folosește CAZURI SIMILARE doar ca ghid de raționament (cum s-a aplicat regula într-o speță asemănătoare), niciodată ca înlocuitor al textului de lege.
5. AMBIGUITATE: înainte să decizi, verifică explicit dacă un detaliu esențial pentru rezultat (ex: locația exactă a acțiunii față de suprafața de pedeapsă, dacă mingea era sau nu în joc, dacă exista sau nu un coechipier în poziție de ofsaid activ) LIPSEȘTE din enunț. Dacă lipsește și CONTEXT LEGISLATIV permite mai multe rezultate posibile în funcție de acel detaliu, NU alege arbitrar un singur scenariu — prezintă TOATE consecințele posibile, clar separate, în loc de o singură decizie fermă.
6. LIMITĂ ANTI-HALUCINAȚIE: dacă răspunsul nu poate fi dedus din CONTEXT LEGISLATIV — chiar dacă crezi că știi regula din altă sursă, sau dacă doar CAZURILE SIMILARE par să o confirme — spune clar: "{NO_CONTEXT_MESSAGE}" Nu completa din cunoștințe generale.

--- CONTEXT LEGISLATIV (Extrase Oficiale din Regulament — sursă de autoritate) ---
{context_str}

--- CAZURI SIMILARE REZOLVATE (informativ — NU text de lege, doar ghid de raționament) ---
{qa_context_str}

--- ÎNTREBAREA/SPEȚA UTILIZATORULUI ---
{raw_query}

Reamintire: decizia se bazează STRICT pe CONTEXT LEGISLATIV. CAZURILE SIMILARE sunt doar un ghid, nu o sursă de citat.

RĂSPUNSUL TĂU STRUCTURAT:
**Analiza fazei:**
**Legea aplicabila (din CONTEXT LEGISLATIV):** daca nu o știi nu mai spune
**Decizia corectă:**
**Varianta corectă (doar dacă întrebarea are variante A/B/C/D):**
"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            options={"num_predict": 400, "temperature": 0.1},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log(f"Ollama a dat rateu la generarea finală (posibil timeout): {e}")
        return f"[EROARE] Nu s-a putut genera răspunsul final: {e}"


def bm25_search(bm25_index, all_nodes, query_string, top_k):
    query_tokens = tokenize(query_string)
    if not query_tokens:
        return []
    scores = np.asarray(bm25_index.get_scores(query_tokens))
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [all_nodes[i] for i in top_indices]


def hybrid_retrieve(vector_retriever, bm25_index, all_nodes, semantic_query, bm25_query_string, domain=None):
    vector_nodes = vector_retriever.retrieve(semantic_query)
    bm25_nodes = bm25_search(bm25_index, all_nodes, bm25_query_string, BM25_TOP_K)

    combined = {}
    for n in vector_nodes:
        combined[n.node.node_id] = n.node
    for n in bm25_nodes:
        combined[n.node_id] = n
    nodes = list(combined.values())

    # FIX #4 (parte 3): filtrăm pe domeniu dacă avem metadata disponibilă.
    # Dacă filtrarea ar elimina TOT (ex: metadata lipsă din index vechi,
    # sau domeniu detectat greșit), cădem înapoi pe lista nefiltrată,
    # ca să nu pierdem recall din cauza unui filtru prea agresiv.
    if domain and domain != "AMBELE":
        filtered = [n for n in nodes if n.metadata.get("domain") == domain]
        if filtered:
            return filtered
        log(f"Filtrarea pe domeniu '{domain}' nu a găsit noduri; folosesc setul complet.")

    return nodes


def rerank(reranker_model, semantic_query, hybrid_nodes, top_k, min_score=RERANK_MIN_SCORE):
    if not hybrid_nodes:
        return []

    pairs = [[semantic_query, node.text] for node in hybrid_nodes]

    scores = reranker_model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    scored = sorted(zip(hybrid_nodes, scores), key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]

    # FIX #5 (parte 3): eliminăm fragmentele sub pragul minim de relevanță,
    # chiar dacă erau în top_k ca poziție.
    return [(n, s) for n, s in top_scored if s >= min_score]


class Pipeline:
    """Container simplu cu toate componentele inițializate o singură dată (index, retrievere, reranker, bancă QA)."""

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
    """
    Încarcă tot ce e necesar O SINGURĂ DATĂ: modele, index de regulament,
    BM25, reranker și banca de cazuri similare. Reutilizabil atât din main()
    (o singură întrebare), cât și dintr-un script de evaluare (multe întrebări).
    """
    log(f"Hardware Acceleration activată pe: {DEVICE.upper()} (embed={EMBED_DEVICE}, rerank={RERANK_DEVICE})")
    log("Loading BGE-M3 Embedding Model...")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME, device=EMBED_DEVICE, model_kwargs={"trust_remote_code": True}
    )
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=768, chunk_overlap=128)

    index = build_or_load_index()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    log("Extracting text nodes for BM25...")
    all_nodes = list(index.docstore.docs.values())
    corpus_tokens = [tokenize(node.text) for node in all_nodes]
    bm25_index = BM25Okapi(corpus_tokens)
    vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)

    log(f"Loading Reranker model on {RERANK_DEVICE.upper()}...")
    # FIX #1: folosim RERANK_DEVICE (derivat din disponibilitatea reală CUDA),
    # nu mai hardcodăm 'cuda' -- altfel crapă pe orice mașină fără GPU.
    reranker_kwargs = {"device": RERANK_DEVICE}
    if RERANK_DEVICE == "cuda":
        reranker_kwargs["automodel_args"] = {"torch_dtype": torch.float16}
    reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", **reranker_kwargs)

    # --- Încărcare bancă de cazuri similare (Referee-IQ), index SEPARAT ---
    qa_conn = None
    qa_items = []
    qa_bm25_index = None
    qa_matrix = np.zeros((0, 0), dtype=np.float32)

    qa_loaded = load_qa_bank(QA_DB_PATH)
    if qa_loaded:
        qa_conn, qa_items = qa_loaded
        if qa_items:
            log(f"Am încărcat {len(qa_items)} cazuri similare din '{QA_DB_PATH}'.")
            qa_bm25_index = build_qa_bm25(qa_items)
            qa_matrix = ensure_qa_embeddings(qa_conn, qa_items, Settings.embed_model)
        else:
            log("Baza de cazuri similare există, dar e goală (rulează sync_referee_iq.py).")

    return Pipeline(all_nodes, bm25_index, vector_retriever, reranker_model,
                     qa_conn, qa_items, qa_bm25_index, qa_matrix)


def answer_query(pipeline, raw_query, exclude_qa_id=None, verbose=True):
    """
    Rulează UN singur query prin tot pipeline-ul: rewrite -> hybrid retrieve
    (regulament) -> rerank -> hybrid retrieve (cazuri similare) -> rerank ->
    generare răspuns final. Returnează un dict cu toate detaliile intermediare.

    exclude_qa_id: exclude acest id din banca de cazuri similare la retrieval
    (folosit de scriptul de evaluare, ca să nu "vadă" propriul răspuns ca
    "informativ" atunci când testează chiar acea întrebare din referee_iq.db).
    """
    bm25_query_string, semantic_query, domain = rewrite_query(raw_query)

    hybrid_nodes = hybrid_retrieve(
        pipeline.vector_retriever, pipeline.bm25_index, pipeline.all_nodes,
        semantic_query, bm25_query_string, domain=domain,
    )
    top_reranked_nodes = rerank(pipeline.reranker_model, semantic_query, hybrid_nodes, RERANK_TOP_K)
    retrieved_texts = [n.text for n, s in top_reranked_nodes]

    qa_reranked = []
    if pipeline.qa_items and pipeline.qa_bm25_index is not None:
        qa_candidates = hybrid_retrieve_qa(
            Settings.embed_model, pipeline.qa_bm25_index, pipeline.qa_items, pipeline.qa_matrix,
            semantic_query, bm25_query_string, domain=domain, exclude_id=exclude_qa_id,
        )
        qa_reranked = rerank_qa(pipeline.reranker_model, semantic_query, qa_candidates, QA_RERANK_TOP_K)

    final_answer = generate_final_answer(raw_query, retrieved_texts, qa_reranked)

    if verbose:
        print("\n" + "-" * 30 + " TOP FRAGMENTE REGULAMENT (peste prag) " + "-" * 30)
        if not top_reranked_nodes:
            print("Niciun fragment nu a trecut de pragul minim de relevanță.")
        for i, (node, score) in enumerate(top_reranked_nodes):
            print(f"\n[MATCH {i + 1}] | Scor: {score:.4f}")
            print(f"{node.text}\n")
        print("-" * 90 + "\n")

        print("\n" + "-" * 30 + " CAZURI SIMILARE (peste prag) " + "-" * 30)
        if not qa_reranked:
            print("Niciun caz similar relevant găsit.")
        for i, (it, score) in enumerate(qa_reranked):
            print(f"\n[CAZ {i + 1}] | Scor: {score:.4f} | Domeniu: {it['pipeline_domain']}")
            print(f"Întrebare: {it['text']}")
            print(f"Răspuns corect: {it.get('correct_answer_text')}")
        print("-" * 90 + "\n")

        print("=" * 30 + " RĂSPUNSUL INSTRUCTORULUI AI " + "=" * 30)
        print(f"\n{final_answer}\n")
        print("=" * 86)

    return {
        "raw_query": raw_query,
        "domain": domain,
        "bm25_query_string": bm25_query_string,
        "semantic_query": semantic_query,
        "retrieved_texts": retrieved_texts,
        "reranked_nodes": top_reranked_nodes,
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
    log(f"Raw User Query:\n{raw_query}")
    print("=" * 50 + "\n")

    answer_query(pipeline, raw_query, verbose=True)

    pipeline.close()


if __name__ == "__main__":
    main()
