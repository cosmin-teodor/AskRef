"""
Rulează întrebările din referee_iq.db prin pipeline-ul complet (v5) și
compară litera aleasă de model cu răspunsul corect din baza de date.

NOU față de versiunea anterioară:
  - salvează FRAGMENTELE RECUPERATE (text + scor + sursă) pentru fiecare
    întrebare => se poate diagnostica retrieval vs raționament
  - CLASIFICARE AUTOMATĂ a cauzei: dacă niciun fragment nu conține termenii
    din răspunsul corect => "RETRIEVAL"; dacă fragmentele îl conțineau dar
    litera e greșită => "RAȚIONAMENT"
  - detectează RĂSPUNSURI TRUNCHIATE (modelul tăiat la mijlocul frazei)
  - detectează NEPOTRIVIRE literă/text (modelul raționează corect dar
    scrie altă literă)
  - două fișiere: eval_report.csv (sumar, ușor de deschis în Excel) și
    eval_details.jsonl (tot, inclusiv fragmentele brute)

Rulează din același folder cu rag_hybrid_search_v5.py.
"""

import os
import re
import csv
import json
import time
import sqlite3
import difflib

from rag_hybrid_search_v5 import (
    initialize_pipeline,
    answer_query,
    log,
    QA_DB_PATH,
    FINAL_ANSWER_MARKER,
    NO_CONTEXT_MESSAGE,
)

# =====================================================================
# CONFIG
# =====================================================================
EVAL_LIMIT = 15        # None = toate; pune 20-30 pentru test rapid
EVAL_CATEGORIES = ("LJ", "ROAF")  # sau doar ("LJ",)

BASE_DIR = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM"
REPORT_CSV_PATH = os.path.join(BASE_DIR, "eval_report.csv")
DETAILS_JSONL_PATH = os.path.join(BASE_DIR, "eval_details.jsonl")

LETTERS = "ABCDEFGH"

CSV_FIELDS = [
    "id", "category", "question", "correct_letter", "correct_text",
    "predicted_letter", "is_correct", "failure_type",
    "answer_looks_truncated", "letter_text_mismatch",
    "domain_detected", "num_candidates", "num_fragments",
    "top_fragment_score", "fragment_sources", "error",
]

# Cuvinte prea comune ca să conteze la potrivirea răspuns-corect / fragmente
STOPWORDS = {
    "care", "este", "unui", "unei", "unde", "cand", "când", "daca", "dacă",
    "pentru", "catre", "către", "dupa", "după", "fara", "fără", "sau", "si",
    "și", "din", "cu", "la", "in", "în", "pe", "de", "se", "un", "o", "al",
    "ale", "lui", "nu", "mai", "ca", "că", "cel", "cea", "acest", "acesta",
    "toate", "raspunsurile", "răspunsurile", "varianta", "nicio", "corecta",
    "corectă", "jocul", "arbitrul", "echipei", "echipa", "jucator", "jucător",
}


def normalize(text):
    text = (text or "").lower()
    replacements = {"ă": "a", "â": "a", "î": "i", "ș": "s", "ş": "s",
                    "ț": "t", "ţ": "t"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def content_tokens(text):
    return {t for t in re.findall(r"\w+", normalize(text))
            if len(t) > 3 and t not in STOPWORDS}


def load_all_questions(db_path, categories):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in categories)
    cur.execute(
        f"SELECT id, category, pipeline_domain, text, answers_json, "
        f"correct_answer_index, correct_answer_text FROM questions "
        f"WHERE category IN ({placeholders})",
        categories,
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for qid, category, domain, text, answers_json, idx_correct, text_correct in rows:
        try:
            answers = json.loads(answers_json) if answers_json else []
        except json.JSONDecodeError:
            answers = []
        items.append({
            "id": qid, "category": category, "pipeline_domain": domain,
            "text": text or "", "answers": answers,
            "correct_answer_index": idx_correct, "correct_answer_text": text_correct,
        })
    return items


def build_raw_query(item):
    lines = [item["text"], ""]
    for idx, ans in enumerate(item["answers"]):
        letter = LETTERS[idx] if idx < len(LETTERS) else str(idx)
        lines.append(f"{letter}. {ans}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_selected_letter(answer_text):
    """
    Parsare robustă bazată pe ce produce efectiv llama3.1:8b în practică.
    Ordinea de prioritate: marcaj strict > câmp dedicat > ultima literă din text.

    Variante observate în eval:
      "RĂSPUNS FINAL: C"
      "RĂSPUNS FINAL: **C**"
      "**RĂSPUNSUL TĂU:** D"
      "**Varianta mea este: C**"
      "Varianta corectă: C"
      "Varianta corectă (doar dacă...): C. Lovitura..."
      "Răspunsul meu este: A"
      "răspuns final: b"  (fără diacritice, lowercase)
    """
    if not answer_text:
        return None

    # 1. Marcajul principal din prompt: "RĂSPUNS FINAL: X"
    # (și variante fără diacritice / lowercase pe care llama le produce uneori)
    patterns_strict = [
        r"R[AĂÂ]SPUNS\s+FINAL\s*:\s*\**\s*([A-H])\b",
        r"raspuns\s+final\s*:\s*\**\s*([A-H])\b",
    ]
    for pat in patterns_strict:
        m = re.search(pat, answer_text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2. Câmpul "Varianta mea este:" (nou în v5)
    m = re.search(r"Varianta\s+mea\s+este\s*:\s*\[?\**\s*([A-H])\b",
                  answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. Diverse forme de "răspunsul meu / al meu" observate în output real
    patterns_response = [
        r"R[AĂÂ]SPUNSUL\s+(MEU|T[AĂÂ]U|CORECT)\s*:\s*\**\s*([A-H])\b",
        r"r[aăâ]spunsul\s+(meu|t[aăâ]u|corect)\s*:\s*\**\s*([A-H])\b",
        r"R[AĂÂ]SPUNS(?:UL)?\s+(?:FINAL\s+)?(?:MEU|AL\s+MEU)\s*(?:ESTE|:)\s*\**\s*([A-H])\b",
    ]
    for pat in patterns_response:
        m = re.search(pat, answer_text, re.IGNORECASE)
        if m:
            # grupul cu litera poate fi 1 sau 2 depinzând de pattern
            letter_group = m.lastindex
            return m.group(letter_group).upper()

    # 4. "Varianta corectă (...): X" sau "Varianta corectă: X. Textul..."
    m = re.search(r"Varianta\s+corect[ăa][^:]*:\s*\**\s*([A-H])\b",
                  answer_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    # 5. "Răspunsul meu este: X" / "Răspuns: X"
    m = re.search(r"R[aăâ]spuns(?:ul\s+meu\s+este)?\s*:\s*\**\s*([A-H])\b",
                  answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 6. Fallback: ultima literă urmată de '.' sau ')' din tot textul.
    # Ultimele 400 de caractere: modelul pune concluzia la final.
    tail = answer_text[-400:]
    fallback = re.findall(r"\b([A-H])[\.)\s]", tail)
    if fallback:
        return fallback[-1].upper()

    # 7. Ultimă resursă: orice literă izolată în ultimele 100 de caractere
    last = answer_text[-100:].strip()
    m = re.search(r"\b([A-H])\b", last)
    if m:
        return m.group(1).upper()

    return None


def letter_implied_by_text(answer_text, answers):
    """
    Care variantă se potrivește cel mai bine cu ce a SCRIS modelul în
    secțiunea de decizie? Folosit ca să detectăm cazul în care raționamentul
    e corect dar litera finală e greșită (eroare de format, nu de fond).
    """
    decision = answer_text
    match = re.search(r"Decizia corect[ăa][^:]*:(.{0,600})", answer_text,
                      re.IGNORECASE | re.DOTALL)
    if match:
        decision = match.group(1)
    decision_norm = normalize(decision)

    best_letter, best_ratio = None, 0.0
    for idx, ans in enumerate(answers):
        ratio = difflib.SequenceMatcher(None, normalize(ans), decision_norm).ratio()
        tokens = content_tokens(ans)
        overlap = (len(tokens & content_tokens(decision)) / len(tokens)) if tokens else 0.0
        combined = max(ratio, overlap)
        if combined > best_ratio:
            best_ratio, best_letter = combined, (
                LETTERS[idx] if idx < len(LETTERS) else str(idx))
    return (best_letter, best_ratio) if best_ratio >= 0.5 else (None, best_ratio)


def looks_truncated(answer_text):
    """Răspuns tăiat de limita de tokeni: nu se termină cu punctuație de final."""
    stripped = (answer_text or "").strip()
    if not stripped:
        return False
    if FINAL_ANSWER_MARKER.lower() in stripped.lower():
        return False
    return not stripped.endswith((".", "!", "?", ")", "\"", "”", ":"))


def classify_failure(fragments, correct_text, predicted_letter, answer_text):
    """
    Cauza cea mai probabilă a greșelii:
      NO_ANSWER        - nu s-a putut extrage o literă
      TRUNCATED        - răspuns tăiat de limita de tokeni
      NO_CONTEXT       - modelul a spus explicit că nu găsește informația
      RETRIEVAL        - fragmentele NU conțineau răspunsul corect
      FORMAT           - fragmentele îl conțineau, raționamentul pare corect,
                         dar litera scrisă e alta
      RAȚIONAMENT      - fragmentele îl conțineau, dar decizia e greșită
    """
    if looks_truncated(answer_text):
        return "TRUNCATED"
    if predicted_letter is None:
        return "NO_ANSWER"
    if NO_CONTEXT_MESSAGE.lower() in (answer_text or "").lower():
        return "NO_CONTEXT"

    correct_tokens = content_tokens(correct_text)
    if not correct_tokens:
        return "RAȚIONAMENT"

    all_fragment_text = " ".join(f.get("text", "") for f in fragments)
    fragment_tokens = content_tokens(all_fragment_text)
    coverage = len(correct_tokens & fragment_tokens) / len(correct_tokens)

    if coverage < 0.5:
        return "RETRIEVAL"
    return "RAȚIONAMENT"


def load_already_processed_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def append_csv(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def main():
    log("Încarc întrebările din referee_iq.db...")
    items = load_all_questions(QA_DB_PATH, EVAL_CATEGORIES)
    log(f"Total în bază: {len(items)}")

    processed = load_already_processed_ids(REPORT_CSV_PATH)
    if processed:
        log(f"Raport parțial existent: {len(processed)} deja procesate, le sar.")
        items = [it for it in items if it["id"] not in processed]

    if EVAL_LIMIT is not None:
        items = items[:EVAL_LIMIT]

    total = len(items)
    log(f"De rulat acum: {total}")
    if total == 0:
        log("Nimic de evaluat.")
        return

    pipeline = initialize_pipeline()

    correct_count, error_count = 0, 0
    wrong_results = []
    per_category = {}
    failure_counts = {}
    start = time.time()

    for i, item in enumerate(items, start=1):
        qid, category = item["id"], item["category"]
        per_category.setdefault(category, [0, 0])
        per_category[category][1] += 1

        raw_query = build_raw_query(item)
        idx_correct = item["correct_answer_index"]
        correct_letter = (LETTERS[idx_correct]
                          if isinstance(idx_correct, int) and 0 <= idx_correct < len(LETTERS)
                          else None)

        error_msg, predicted_letter, final_answer = "", None, ""
        domain_detected, fragments, num_candidates = "", [], 0

        try:
            result = answer_query(pipeline, raw_query, exclude_qa_id=qid, verbose=False)
            final_answer = result["final_answer"]
            domain_detected = result["domain"]
            fragments = result["fragments"]
            num_candidates = result["num_candidates"]
            predicted_letter = extract_selected_letter(final_answer)
        except Exception as e:
            error_msg = str(e)
            log(f"[{i}/{total}] EROARE la id={qid}: {e}")

        is_correct = (predicted_letter is not None
                      and correct_letter is not None
                      and predicted_letter == correct_letter)

        implied_letter, implied_ratio = letter_implied_by_text(
            final_answer, item["answers"])
        letter_text_mismatch = bool(
            implied_letter and predicted_letter and implied_letter != predicted_letter)

        failure_type = ""
        if error_msg:
            failure_type = "EXCEPTION"
            error_count += 1
        elif is_correct:
            correct_count += 1
            per_category[category][0] += 1
        else:
            failure_type = classify_failure(
                fragments, item["correct_answer_text"], predicted_letter, final_answer)
            if letter_text_mismatch and failure_type == "RAȚIONAMENT":
                failure_type = "FORMAT"
            wrong_results.append({
                "item": item,
                "correct_letter": correct_letter,
                "predicted_letter": predicted_letter,
                "implied_letter": implied_letter,
                "final_answer": final_answer,
                "fragments": fragments,
                "failure_type": failure_type,
                "domain_detected": domain_detected,
            })

        if failure_type:
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        append_csv(REPORT_CSV_PATH, {
            "id": qid,
            "category": category,
            "question": item["text"],
            "correct_letter": correct_letter,
            "correct_text": item["correct_answer_text"],
            "predicted_letter": predicted_letter,
            "is_correct": is_correct,
            "failure_type": failure_type,
            "answer_looks_truncated": looks_truncated(final_answer),
            "letter_text_mismatch": letter_text_mismatch,
            "domain_detected": domain_detected,
            "num_candidates": num_candidates,
            "num_fragments": len(fragments),
            "top_fragment_score": (round(fragments[0]["score"], 4) if fragments else ""),
            "fragment_sources": ";".join(
                f"{f['source']}:{f['file_name']}p{f['page']}" for f in fragments),
            "error": error_msg,
        })

        append_jsonl(DETAILS_JSONL_PATH, {
            "id": qid,
            "category": category,
            "question": item["text"],
            "answers": item["answers"],
            "correct_letter": correct_letter,
            "predicted_letter": predicted_letter,
            "implied_letter": implied_letter,
            "implied_ratio": round(implied_ratio, 3),
            "is_correct": is_correct,
            "failure_type": failure_type,
            "domain_detected": domain_detected,
            "fragments": fragments,
            "final_answer": final_answer,
            "error": error_msg,
        })

        elapsed = time.time() - start
        remaining = (elapsed / i) * (total - i)
        status = "OK" if is_correct else (failure_type or "GREȘIT")
        log(f"[{i}/{total}] {qid[:8]} {category} -> {status} "
            f"(pred={predicted_letter}, corect={correct_letter}, "
            f"frag={len(fragments)}) | rămas ~{remaining/60:.1f} min")

    pipeline.close()

    # =====================================================================
    # SUMAR
    # =====================================================================
    print("\n" + "=" * 36 + " SUMAR EVALUARE " + "=" * 36)
    evaluated = correct_count + len(wrong_results) + error_count
    if evaluated:
        print(f"Total evaluate: {evaluated}")
        print(f"Corecte: {correct_count} ({100.0*correct_count/evaluated:.1f}%)")
        print(f"Greșite: {len(wrong_results)}")
        print(f"Excepții: {error_count}")

    print("\nAcuratețe pe categorie:")
    for category, (ok, tot) in per_category.items():
        print(f"  {category}: {ok}/{tot} ({100.0*ok/tot if tot else 0:.1f}%)")

    print("\nCAUZA GREȘELILOR (asta îți spune unde să investești):")
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count}")
    print("\n  RETRIEVAL   -> fragmentul corect nu a ajuns în context "
          "(crește TOP_K, ajustează chunking, adaugă în curated_facts.csv)")
    print("  RAȚIONAMENT -> contextul era bun, modelul a decis greșit "
          "(limita modelului -> prompt sau model mai capabil)")
    print("  FORMAT      -> raționament corect, literă greșită "
          "(problemă de formatare a răspunsului)")
    print("  TRUNCATED   -> răspuns tăiat (crește OLLAMA_NUM_PREDICT)")

    # =====================================================================
    # DETALII GREȘELI
    # =====================================================================
    print("\n" + "=" * 30 + f" DETALII GREȘELI ({len(wrong_results)}) " + "=" * 30)
    if not wrong_results:
        print("Nicio greșeală.")
    for i, w in enumerate(wrong_results, 1):
        item = w["item"]
        print(f"\n{'-' * 88}")
        print(f"[GREȘIT #{i}] id={item['id']} | {item['category']} "
              f"| cauză={w['failure_type']} | domeniu={w['domain_detected']}")
        print(f"Întrebare:\n{item['text']}")
        print("\nVariante:")
        for idx, ans in enumerate(item["answers"]):
            letter = LETTERS[idx] if idx < len(LETTERS) else str(idx)
            mark = ""
            if letter == w["correct_letter"]:
                mark += "  <-- CORECT"
            if letter == w["predicted_letter"]:
                mark += "  <-- ALES"
            if letter == w["implied_letter"] and letter != w["predicted_letter"]:
                mark += "  <-- (sugerat de propriul text al modelului)"
            print(f"  {letter}. {ans}{mark}")

        print(f"\nFragmente recuperate ({len(w['fragments'])}):")
        if not w["fragments"]:
            print("  (niciunul)")
        for j, f in enumerate(w["fragments"], 1):
            preview = f["text"][:300].replace("\n", " ")
            print(f"  [{j}] scor={f['score']:.3f} sursă={f['source']} "
                  f"{f['file_name']} p.{f['page']}")
            print(f"      {preview}...")

        print(f"\nRăspunsul modelului:\n{w['final_answer']}")

    print("\n" + "=" * 88)
    log(f"Sumar CSV: {REPORT_CSV_PATH}")
    log(f"Detalii complete (cu fragmente): {DETAILS_JSONL_PATH}")


if __name__ == "__main__":
    main()
