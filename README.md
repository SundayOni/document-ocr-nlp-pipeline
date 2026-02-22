# Document OCR & NLP Pipeline

An end-to-end pipeline that extracts and structures information from scanned planning and local authority documents using OCR and NLP techniques.

## Project Overview

This pipeline takes PDF planning documents as input, extracts text using OCR, and uses NLP to identify and structure key fields such as reference numbers, dates, addresses, and decisions. Each extracted field is assigned a confidence score to flag uncertain extractions for human review.

This mirrors real-world data extraction challenges faced by organisations migrating document-heavy workflows onto digital registers.

## Pipeline Architecture
```
PDF Documents → Image Conversion → Preprocessing → OCR (EasyOCR) → NLP Extraction → Structured Output (JSON/CSV)
```

## Features

- Converts PDF pages to images using pdf2image and Poppler
- Preprocesses images with OpenCV (grayscale, binarisation) for improved OCR accuracy
- Extracts text using EasyOCR
- Identifies key fields using regex patterns and spaCy NER:
  - Planning reference number
  - Decision date
  - Site address
  - Decision outcome
  - Applicant name
- Assigns confidence scores (0.0–1.0) to each extracted field
- Outputs structured data to JSON and CSV

## Tech Stack

- Python 3.11
- EasyOCR
- OpenCV
- pdf2image / Poppler
- spaCy (en_core_web_sm)
- pandas

## Project Structure
```
document-ocr-nlp-pipeline/
├── data/
│   ├── raw/          # Input PDF documents
│   └── processed/    # Extracted raw text files
├── outputs/          # Structured JSON and CSV outputs
├── notebooks/        # Jupyter walkthrough notebook
├── src/
│   ├── ocr_pipeline.py    # PDF to text extraction
│   └── nlp_extractor.py   # NLP field extraction with confidence scoring
└── README.md
```

## Setup
```bash
git clone https://github.com/SundayOni/document-ocr-nlp-pipeline.git
cd document-ocr-nlp-pipeline
python -m venv venv
venv\Scripts\activate
pip install easyocr opencv-python-headless pdf2image spacy pandas
python -m spacy download en_core_web_sm
```

Also install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and update the `poppler_path` in `ocr_pipeline.py`.

## Usage
```bash
# Step 1: Extract text from PDFs
python src/ocr_pipeline.py

# Step 2: Extract structured fields
python src/nlp_extractor.py
```

Results are saved to `outputs/extracted_data.json` and `outputs/extracted_data.csv`.

## Sample Output
```json
{
  "filename": "SPLIT_DECISION_TREE_WORKS-6460222.txt",
  "reference": "25/07499/TR",
  "reference_confidence": 1.0,
  "date": "17 February 2026",
  "date_confidence": 1.0,
  "address": "Land Next To 29, Moor Drive, Headingley, Leeds,",
  "address_confidence": 1.0,
  "decision": "PART REFUSAL, PART APPROVAL",
  "decision_confidence": 1.0,
  "applicant": "Leeds LS2",
  "applicant_confidence": 0.6
}
```

## Known Limitations

- Applicant extraction relies on spaCy NER which can misidentify location names as person names. Confidence scores flag these cases.
- OCR quality depends on the scan quality of the source PDF.
- Currently optimised for UK planning document formats.

## Future Improvements

- GPU support for faster OCR processing
- Fine-tuned NER model for planning document entities
- Cloud deployment as a containerised batch pipeline (e.g. AWS Lambda or Docker)