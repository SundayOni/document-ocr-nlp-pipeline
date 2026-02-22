import json

notebook = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.9"
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Document OCR & NLP Pipeline — Walkthrough\n", "\n", "An end-to-end pipeline that extracts and structures information from scanned UK planning documents using OCR and NLP."],
   "id": "1"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Setup & Imports"],
   "id": "2"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "import re\n",
    "import cv2\n",
    "import numpy as np\n",
    "import easyocr\n",
    "import spacy\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from pdf2image import convert_from_path\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "POPPLER_PATH = r'C:\\poppler\\poppler-25.12.0\\Library\\bin'\n",
    "RAW_DIR = Path('../data/raw')\n",
    "PROCESSED_DIR = Path('../data/processed')\n",
    "print('All imports successful')"
   ],
   "id": "3"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Load a Sample PDF"],
   "id": "4"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pdf_files = list(RAW_DIR.glob('*.pdf'))\n",
    "print(f'Found {len(pdf_files)} PDF(s):')\n",
    "for f in pdf_files:\n",
    "    print(f'  {f.name}')\n",
    "sample_pdf = [f for f in pdf_files if 'DECISION' in f.name.upper()][0]\n",
    "print(f'\\nUsing: {sample_pdf.name}')"
   ],
   "id": "5"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Convert PDF to Images"],
   "id": "6"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pages = convert_from_path(str(sample_pdf), dpi=200, poppler_path=POPPLER_PATH)\n",
    "print(f'Converted {len(pages)} page(s)')\n",
    "plt.figure(figsize=(10, 14))\n",
    "plt.imshow(pages[0])\n",
    "plt.title('Page 1 - Original')\n",
    "plt.axis('off')\n",
    "plt.show()"
   ],
   "id": "7"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Preprocess Image for OCR"],
   "id": "8"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_image(pil_image):\n",
    "    img = np.array(pil_image)\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\n",
    "    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
    "    return binary\n",
    "\n",
    "processed = preprocess_image(pages[0])\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 10))\n",
    "axes[0].imshow(pages[0])\n",
    "axes[0].set_title('Original')\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(processed, cmap='gray')\n",
    "axes[1].set_title('Preprocessed (Binarised)')\n",
    "axes[1].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ],
   "id": "9"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5. Extract Text with EasyOCR"],
   "id": "10"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "reader = easyocr.Reader(['en'], gpu=False)\n",
    "results = reader.readtext(processed, detail=0, paragraph=True)\n",
    "raw_text = '\\n'.join(results)\n",
    "print(raw_text)"
   ],
   "id": "11"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 6. Extract Structured Fields with NLP"],
   "id": "12"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "nlp = spacy.load('en_core_web_sm')\n",
    "txt_file = PROCESSED_DIR / (sample_pdf.stem + '.txt')\n",
    "full_text = txt_file.read_text(encoding='utf-8')\n",
    "\n",
    "def extract_reference(text):\n",
    "    match = re.search(r'\\b\\d{2}/\\d{4,5}/[A-Z]{2,}\\b', text)\n",
    "    return match.group(0) if match else None\n",
    "\n",
    "def extract_date(text):\n",
    "    match = re.search(r'\\b\\d{1,2}\\s+\\w+\\s+\\d{4}\\b', text)\n",
    "    return match.group(0) if match else None\n",
    "\n",
    "def extract_address(text):\n",
    "    match = re.search(r'At:\\s*\\n(.+?)(?:\\n|$)', text)\n",
    "    if match:\n",
    "        return match.group(1).strip()\n",
    "    doc = nlp(text[:2000])\n",
    "    locs = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC', 'FAC')]\n",
    "    return locs[0] if locs else None\n",
    "\n",
    "def extract_decision(text):\n",
    "    match = re.search(r'(APPROVED|REFUSED|GRANTED|DISMISSED|ALLOWED|PART REFUSAL[,\\s]+PART APPROVAL)', text, re.IGNORECASE)\n",
    "    return match.group(0).strip().upper() if match else None\n",
    "\n",
    "def extract_applicant(text):\n",
    "    doc = nlp(text[:2000])\n",
    "    people = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']\n",
    "    return people[0] if people else None\n",
    "\n",
    "extracted = {\n",
    "    'reference': extract_reference(full_text),\n",
    "    'date':      extract_date(full_text),\n",
    "    'address':   extract_address(full_text),\n",
    "    'decision':  extract_decision(full_text),\n",
    "    'applicant': extract_applicant(full_text),\n",
    "}\n",
    "for k, v in extracted.items():\n",
    "    print(f'{k:12}: {v}')"
   ],
   "id": "13"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 7. View Final Structured Output"],
   "id": "14"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../outputs/extracted_data.csv')\n",
    "df"
   ],
   "id": "15"
  }
 ]
}

with open('notebooks/pipeline_walkthrough.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Notebook created successfully')