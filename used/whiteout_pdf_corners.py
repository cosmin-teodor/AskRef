"""
Redactează (elimină conținutul + colorează alb) colțul stânga-sus și
colțul dreapta-jos pe fiecare pagină din PDF-urile dintr-un folder.

IMPORTANT: folosește page.add_redact_annot() + apply_redactions(), care
elimină EFECTIV textul/imaginile din zonă (nu doar le acoperă vizual cu
un dreptunghi). Asta contează dacă extragi text din PDF ulterior (ex:
pipeline-ul RAG) -- o simplă suprapunere vizuală NU ar elimina textul
"ascuns" dedesubt la extragere.

Nu suprascrie fișierele originale -- salvează copii curate într-un folder nou.
"""

import os
import fitz  # PyMuPDF

# =====================================================================
# CONFIG
# =====================================================================
INPUT_DIR = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\Documents"
OUTPUT_DIR = r"C:\Users\cosmy\Documents\AskRef\AskRef_LLM\Documents_curatate"

# Dimensiunea colțurilor, ca procent din lățimea/înălțimea paginii.
# Ajustează dacă logo-ul/numărul de pagină e mai mare sau mai mic decât
# acoperă implicit -- deschide o pagină și estimează vizual ce procent
# din lățime/înălțime ocupă zona cu zgomot (număr pagină, logo etc).
CORNER_WIDTH_RATIO = 1
CORNER_HEIGHT_RATIO = 0.1


def whiteout_corners(input_path, output_path,
                      corner_width_ratio=CORNER_WIDTH_RATIO,
                      corner_height_ratio=CORNER_HEIGHT_RATIO):
    doc = fitz.open(input_path)

    for page in doc:
        rect = page.rect
        w, h = rect.width, rect.height

        top_left = fitz.Rect(0, 0, w * corner_width_ratio, h * corner_height_ratio)
        bottom_right = fitz.Rect(
            w * (1 - corner_width_ratio), h * (1 - corner_height_ratio), w, h
        )

        for zone in (top_left, bottom_right):
            # fill=(1,1,1) = alb. Redacția elimină conținutul din zonă,
            # nu doar îl acoperă -- apare gol/alb și la extragerea de text.
            page.add_redact_annot(zone, fill=(1, 1, 1))

        page.apply_redactions()

    doc.save(output_path)
    doc.close()


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Folder negăsit: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Niciun PDF găsit în {INPUT_DIR}")
        return

    for fname in pdf_files:
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)
        print(f"Procesez: {fname} ...")
        whiteout_corners(input_path, output_path)
        print(f"  Salvat: {output_path}")

    print(f"\nGata. {len(pdf_files)} fișiere curate salvate în: {OUTPUT_DIR}")
    print("Verifică vizual câteva pagini -- dacă zona albă nu acoperă exact")
    print("ce trebuie (prea mult / prea puțin), ajustează CORNER_WIDTH_RATIO")
    print("și CORNER_HEIGHT_RATIO de mai sus și rulează din nou.")


if __name__ == "__main__":
    main()
