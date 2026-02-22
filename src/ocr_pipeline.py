import os
import cv2
import numpy as np
import easyocr
from pdf2image import convert_from_path
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Initialise EasyOCR (English) ───────────────────────────────────────────
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("Model loaded.")

# ── Image preprocessing ────────────────────────────────────────────────────
def preprocess_image(pil_image):
    """Convert PIL image to cleaned grayscale numpy array for OCR."""
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Binarise using Otsu's threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ── OCR a single PDF ───────────────────────────────────────────────────────
def process_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")
    pages = convert_from_path(str(pdf_path), dpi=200, poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin")
    all_text = []

    for i, page in enumerate(pages):
        print(f"  Page {i+1}/{len(pages)}...")
        img = preprocess_image(page)
        results = reader.readtext(img, detail=0, paragraph=True)
        page_text = "\n".join(results)
        all_text.append(f"--- Page {i+1} ---\n{page_text}")

    full_text = "\n\n".join(all_text)

    # Save extracted text
    output_file = PROCESSED_DIR / (pdf_path.stem + ".txt")
    output_file.write_text(full_text, encoding="utf-8")
    print(f"  Saved to {output_file}")
    return full_text

# ── Run pipeline on all PDFs in data/raw ──────────────────────────────────
if __name__ == "__main__":
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/")
    else:
        print(f"Found {len(pdf_files)} PDF(s)")
        for pdf in pdf_files:
            process_pdf(pdf)
        print("\nAll done! Check data/processed/ for results.")