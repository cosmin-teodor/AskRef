"""
Diagnostic: parcurge PDF-urile din DIR_PATH și identifică paginile care
conțin imagini/diagrame ȘI au text extras foarte puțin -- candidate clare
pentru transcriere MANUALĂ în curated_facts.csv, pentru că extragerea
automată de text nu poate "vedea" conținutul unei diagrame.

Nu modifică nimic -- doar afișează un raport, ca să știi exact unde merită
timp manual, în loc să răsfoiești tot documentul.
"""

import os
import fitz  # PyMuPDF

DIR_PATH = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\Documents"

# Sub acest număr de caractere de text extras pe pagină, considerăm
# pagina "săracă în text" -- candidat pentru inspecție manuală dacă
# are și imagini.
LOW_TEXT_THRESHOLD_CHARS = 150


def analyze_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    flagged_pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        images = page.get_images(full=True)

        if images and len(text) < LOW_TEXT_THRESHOLD_CHARS:
            flagged_pages.append(
                {
                    "page": page_num,
                    "num_images": len(images),
                    "text_chars": len(text),
                    "text_preview": text[:100].replace("\n", " "),
                }
            )

    total_pages = doc.page_count
    doc.close()
    return flagged_pages, total_pages


def main():
    pdf_files = [f for f in sorted(os.listdir(DIR_PATH)) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Niciun PDF găsit în {DIR_PATH}")
        return

    print("=" * 90)
    print("RAPORT: pagini cu imagini/diagrame și text puțin extras (candidate pentru transcriere manuală)")
    print("=" * 90)

    total_flagged = 0
    for fname in pdf_files:
        full_path = os.path.join(DIR_PATH, fname)
        flagged, total_pages = analyze_pdf(full_path)
        total_flagged += len(flagged)

        print(f"\n--- {fname} ---")
        if not flagged:
            print("  Nicio pagină suspectă (toate paginile cu imagini au text suficient extras).")
            continue

        for f in flagged:
            print(
                f"  Pagina {f['page']}: {f['num_images']} imagine(i), "
                f"{f['text_chars']} caractere text extras"
            )
            print(f"    Preview text: \"{f['text_preview']}...\"")

    print("\n" + "=" * 90)
    print(f"Total pagini de verificat manual: {total_flagged}")
    print("Pentru fiecare, deschide PDF-ul la acea pagină, vezi ce arată diagrama,")
    print("și dacă e o regulă importantă, adaug-o ca rând nou în curated_facts.csv.")
    print("=" * 90)


if __name__ == "__main__":
    main()