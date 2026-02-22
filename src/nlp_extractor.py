import re
import json
import spacy
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load spaCy model ───────────────────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded.")

# ── Extraction functions ───────────────────────────────────────────────────
def extract_reference(text):
    match = re.search(r'\b\d{2}/\d{4,5}/[A-Z]{2,}\b', text)
    return match.group(0) if match else None

def extract_date(text):
    match = re.search(r'\b\d{1,2}\s+\w+\s+\d{4}\b', text)
    return match.group(0) if match else None

def extract_address(text):
    match = re.search(r'At:\s*\n(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    # fallback: use spaCy to find location entities
    doc = nlp(text[:2000])
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    return locs[0] if locs else None

def extract_decision(text):
    patterns = [
        r'(APPROVED|REFUSED|GRANTED|DISMISSED|ALLOWED|PART REFUSAL[,\s]+PART APPROVAL)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip().upper()
    return None

def extract_applicant(text):
    doc = nlp(text[:2000])
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return people[0] if people else None

def confidence_score(value, method="regex"):
    if value is None:
        return 0.0
    if method == "regex":
        return 1.0
    if method == "spacy":
        return 0.6
    return 0.5

def extract_all(text, filename):
    reference = extract_reference(text)
    date      = extract_date(text)
    address   = extract_address(text)
    decision  = extract_decision(text)
    applicant = extract_applicant(text)

    # Address uses spaCy fallback so lower confidence
    address_method = "regex" if re.search(r'At:\s*\n', text) else "spacy"

    return {
        "filename":             filename,
        "reference":            reference,
        "reference_confidence": confidence_score(reference),
        "date":                 date,
        "date_confidence":      confidence_score(date),
        "address":              address,
        "address_confidence":   confidence_score(address, address_method),
        "decision":             decision,
        "decision_confidence":  confidence_score(decision),
        "applicant":            applicant,
        "applicant_confidence": confidence_score(applicant, "spacy"),
    }

# ── Run extractor on all processed text files ──────────────────────────────
if __name__ == "__main__":
    txt_files = list(PROCESSED_DIR.glob("*.txt"))
    if not txt_files:
        print("No text files found in data/processed/")
    else:
        results = []
        for txt_file in txt_files:
            print(f"Extracting from: {txt_file.name}")
            text = txt_file.read_text(encoding="utf-8")
            record = extract_all(text, txt_file.name)
            results.append(record)
            print(f"  → {record}")

        # Save to JSON
        json_path = OUTPUT_DIR / "extracted_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save to CSV
        csv_path = OUTPUT_DIR / "extracted_data.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

        print(f"\nResults saved to {json_path} and {csv_path}")