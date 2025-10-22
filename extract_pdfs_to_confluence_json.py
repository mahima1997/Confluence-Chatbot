#!/usr/bin/env python3
"""
extract_pdfs_to_confluence_json.py

Extract text, tables and images from PDFs in source_data/data/ and produce Confluence-like JSON payloads.

Outputs:
- extracted_page_html.json files -> extracted_json/
- extracted images -> extracted_content/extracted_images/
"""

import os
import json
import traceback
from pathlib import Path

# ---------- CONFIG ----------
SOURCE_DATA_DIR = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot/source_data/data")
OUT_DIR = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot/extracted_content/extracted_images")
OUT_JSON_DIR = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot/extracted_json")

# Create output directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------

def try_import(name):
    """Try to import a module, return None if not available"""
    try:
        return __import__(name)
    except Exception:
        return None

def image_contains_text(image_path, min_text_length=10):
    """
    Check if an image contains text using OCR
    Returns True if text is found, False otherwise
    """
    try:
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np

        # Load image with PIL for OCR
        pil_image = Image.open(image_path)

        # Convert to grayscale for better OCR results
        gray_image = pil_image.convert('L')

        # Try OCR with different PSM modes for better results
        text_content = ""

        # Try different PSM modes (Page Segmentation Modes)
        for psm in [3, 6, 8, 11, 13]:  # Common modes for different text layouts
            try:
                config = f'--psm {psm}'
                text = pytesseract.image_to_string(gray_image, config=config)
                if len(text.strip()) > len(text_content):
                    text_content = text.strip()
            except:
                continue

        # Clean up the extracted text
        text_content = text_content.strip()

        # Check if we found meaningful text
        if len(text_content) >= min_text_length:
            # Additional check: look for actual words (not just symbols/numbers)
            words = [word for word in text_content.split() if len(word) > 2]
            if len(words) >= 1:  # At least one word with 3+ characters
                return True, text_content[:200]  # Return first 200 chars as preview

        return False, ""

    except Exception as e:
        print(f"Warning: OCR failed for {image_path}: {str(e)}")
        return False, ""

def clean_text_remove_tables(text, tables_for_page):
    """
    Clean text by removing table-like content when structured tables are available
    """
    if not tables_for_page:
        return text

    lines = text.split('\n')
    cleaned_lines = []
    i = 0

    # Expanded table content detection patterns
    table_keywords = [
        'sr', 'no', 'reasons', 'descriptions', 'detailed', 'cause',
        'testing', 'budget', 'campaign', 'rejection', 'optimization',
        'specific', 'additional', 'higher', 'incorrect', 'manually',
        'national', 'newly', 'launched', 'flight', 'inventory',
        'recommendation', 'version', 'error', 'deprecat', 'category',
        'confidence', 'q4', 'q1', 'q2', 'q3', '2024', '2025', '2026',
        'dependencies', 'team', 'work', 'holistic', 'optimization',
        'across', 'levers', 'mvp', 'infight', 'audience', 'tcom',
        'cmp', 'integration', 'search', 'expansion', 'test', 'completion',
        'measurement', 'timeline', 'sept', 'currently', 'results',
        'planned', 'ppr', 'release', 'change', 'scope', 'resource',
        'different', 'project', 'hence', 'delivered', 'post',
        'completion', 'requirements', 'gtm', 'psg', 'read',
        'creative', 'weightage', 'mandatory', 'mag', 'awareness',
        'consideration', 'deployment', 'cross', 'channel', 'low',
        'adoption', 'improvement', 'all', 'channels', 'custom',
        'based', 'media', 'brief', 'filters', 'rejection', 'reason',
        'enhancement', 'lines', 'ending', 'early', 'others',
        'deprioritized', 'did', 'not', 'pick', 'up', 'delivery',
        'explainability', 'genai', 'product', 'support', 'audience',
        'roa', 'prediction', 'mpr', 'allocation', 'existing', 'pool',
        'conversion', 'catalog', 'custom', 'segments', 'upsell',
        'kiosk', 'radeus', 'roadmap', 'revisited', 'track',
        'self', 'service', 'campaign', 'phase', 'low', 'obb',
        'audience', 'reach', 'ctr', 'dev', 'obb', 'strategy',
        'social', 'self', 'requirement', 'alignment', 'maria',
        'measurement', 'median', 'candel', 'requirement'
    ]

    while i < len(lines):
        line = lines[i].strip()
        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this line looks like table content - be very aggressive
        is_table_like = False

        # Pattern 1: Short lines with table keywords (very sensitive detection)
        if len(line) < 200:  # Increased from 80 to catch more fragments
            line_lower = line.lower()
            keyword_matches = sum(1 for keyword in table_keywords if keyword in line_lower)
            if keyword_matches >= 1:
                is_table_like = True

        # Pattern 2: Lines that start with numbers or contain table-like patterns
        if (line and (line[0].isdigit() or any(char in line for char in ['|', '\t', '•'])) and
            len(line.split()) <= 8):
            is_table_like = True

        # Pattern 3: Lines that look like fragmented table headers or data
        if (len(line) < 100 and
            any(keyword in line.lower() for keyword in ['q4', 'q1', 'q2', 'q3', '2024', '2025', '2026'])):
            is_table_like = True

        # If this looks like table content, skip it and look ahead
        if is_table_like:
            # Skip this line and look ahead to find the end of table content
            j = i + 1
            skip_count = 0
            while j < len(lines) and skip_count < 5:  # Skip up to 5 lines to catch fragments
                next_line = lines[j].strip()
                # Stop if we find a long line (likely real content) or empty line followed by content
                if (len(next_line) > 200 or  # Increased threshold for real content
                    (not next_line and j + 1 < len(lines) and len(lines[j + 1].strip()) > 150)):
                    break
                # Also stop if next line looks like real paragraph content (longer than table fragments)
                if (next_line and len(next_line) > 150 and
                    not any(keyword in next_line.lower() for keyword in table_keywords)):
                    break
                j += 1
                skip_count += 1
            i = j
        else:
            cleaned_lines.append(lines[i])
            i += 1

    return '\n'.join(cleaned_lines).strip()

def extract_pdf_content(pdf_path, output_dir, json_output_path):
    """
    Extract text, tables and images from a single PDF file
    Returns a dictionary with extracted content
    """
    pdfplumber = try_import("pdfplumber")
    fitz = try_import("fitz")      # PyMuPDF
    PyPDF2 = try_import("PyPDF2")

    extracted_text = []
    tables_html = []
    image_filenames = []

    print(f"Processing: {pdf_path}")

    # 1) Extract text and tables with pdfplumber if available
    if pdfplumber:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    extracted_text.append({"page": i+1, "text": txt})

                    # table extraction (HTML format)
                    try:
                        tbls = page.extract_tables()
                    except Exception:
                        tbls = []

                    # Clean text for this page if tables are present
                    page_tables = [tbl for tbl in tbls if tbl]  # Filter out empty tables
                    if page_tables:
                        cleaned_text = clean_text_remove_tables(txt, page_tables)
                        extracted_text[-1]["text"] = cleaned_text

                    for t_idx, tbl in enumerate(tbls):
                        if not tbl:
                            continue
                        header = tbl[0]
                        body = tbl[1:] if len(tbl) > 1 else []
                        html = '<table class="confluenceTable"><thead><tr>'
                        for cell in header:
                            html += f"<th>{(cell or '').strip()}</th>"
                        html += "</tr></thead><tbody>"
                        for row in body:
                            html += "<tr>"
                            for cell in row:
                                html += f"<td>{(cell or '').strip()}</td>"
                            html += "</tr>"
                        html += "</tbody></table>"
                        tables_html.append({"page": i+1, "table_index": t_idx, "html": html})
        except Exception:
            traceback.print_exc()
    else:
        # fallback to PyPDF2 for text
        if PyPDF2:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(pdf_path)
                for i, page in enumerate(reader.pages):
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    extracted_text.append({"page": i+1, "text": txt})
            except Exception:
                traceback.print_exc()
        else:
            extracted_text.append({"page": 1, "text": ""})

    # 2) Extract images using fitz (PyMuPDF) if available
    if fitz:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    fname = output_dir / f"{Path(pdf_path).stem}_page{page_index+1}_img{img_index+1}.{ext}"

                    # Save image first
                    with open(fname, "wb") as f:
                        f.write(image_bytes)

                    # Check if image contains text
                    has_text, text_preview = image_contains_text(fname)

                    if has_text:
                        image_filenames.append(str(fname))
                        print(f"  ✓ Image with text: {os.path.basename(fname)}")
                        if text_preview:
                            print(f"    Text preview: {text_preview}")
                    else:
                        print(f"  ✗ Image without text (filtered): {os.path.basename(fname)}")
                        # Optionally delete the file since it doesn't contain text
                        try:
                            os.remove(fname)
                        except:
                            pass
        except Exception:
            traceback.print_exc()

    # 3) If no extracted images and pdfplumber can only provide bbox info, create placeholders
    # Note: For bbox-only images, we can't perform OCR, so we'll include them but mark as unfiltered
    if not image_filenames and pdfplumber:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    imgs = page.images or []
                    for j, im in enumerate(imgs):
                        fname = output_dir / f"{Path(pdf_path).stem}_page{i+1}_image{j+1}_bbox.txt"
                        with open(fname, "w") as f:
                            # Convert the bbox data to a simple dictionary format
                            # Extract basic properties that are serializable
                            bbox_data = {
                                'name': getattr(im, 'name', f'image_{j+1}'),
                                'x0': float(im.get('x0', 0)),
                                'y0': float(im.get('y0', 0)),
                                'x1': float(im.get('x1', 0)),
                                'y1': float(im.get('y1', 0)),
                                'width': float(im.get('width', 0)),
                                'height': float(im.get('height', 0)),
                                'page': i + 1
                            }
                            f.write(json.dumps(bbox_data))
                        # For bbox-only images, include them as they might contain text
                        # (we can't perform OCR on bbox data alone)
                        image_filenames.append(str(fname))
                        print(f"  ✓ Bbox image (unfiltered): {os.path.basename(fname)}")
        except Exception:
            traceback.print_exc()

    return {
        'extracted_text': extracted_text,
        'tables_html': tables_html,
        'image_filenames': image_filenames
    }

def build_confluence_json(pdf_path, extracted_content):
    """
    Build Confluence-like JSON payload from extracted content
    """
    title = Path(pdf_path).stem

    # Build HTML content
    html_parts = []
    html_parts.append(f"<h1>{title}</h1>")

    # Add extracted text
    for p in extracted_content['extracted_text']:
        page_num = p["page"]
        text = (p["text"] or "").strip()
        if text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                html_parts.append(f"<h2>Page {page_num} text</h2>")
                # limit number of lines to avoid huge output; adjust as needed
                for ln in lines[:500]:
                    # Escape or sanitize if necessary (here we simply insert)
                    html_parts.append(f"<p>{ln}</p>")

    # add tables
    if extracted_content['tables_html']:
        for t in extracted_content['tables_html']:
            html_parts.append(f"<h3>Table (page {t['page']})</h3>")
            html_parts.append(t["html"])

    # add extracted images as <ac:image> references (filename used as attachment name)
    for fname in extracted_content['image_filenames']:
        basename = os.path.basename(fname)
        html_parts.append("<p>Extracted image:</p>")
        html_parts.append(f'<ac:image><ri:attachment ri:filename="{basename}" /></ac:image>')

    storage_value = "\n".join(html_parts)

    # Metadata counts (pdf_export style)
    metadata = {
        "pdf_export": {
            "original_filename": os.path.basename(pdf_path),
            "pages_in_pdf": len(extracted_content['extracted_text']) if extracted_content['extracted_text'] else None,
            "tables_detected": len(extracted_content['tables_html']),
            "visual_elements": {
                "images": len(extracted_content['image_filenames']),
                "diagrams": 0,
                "charts": 0,
                "total_pages": len(extracted_content['extracted_text']) if extracted_content['extracted_text'] else None
            },
            "has_visuals": len(extracted_content['image_filenames']) > 0
        }
    }

    payload = {
        "type": "page",
        "title": title,
        "space": {"key": "LOCAL"},
        "body": {
            "storage": {
                "value": storage_value,
                "representation": "storage"
            }
        },
        "metadata": metadata,
        "attachments": [{"filename": os.path.basename(p), "path": p} for p in extracted_content['image_filenames']]
    }

    return payload

def main():
    """Main function to process all PDFs"""
    # Find all PDF files in source_data/data/
    pdf_files = list(SOURCE_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {SOURCE_DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    # Process each PDF
    for pdf_path in pdf_files:
        try:
            # Extract content
            extracted_content = extract_pdf_content(pdf_path, OUT_DIR, None)

            # Build JSON payload
            json_payload = build_confluence_json(pdf_path, extracted_content)

            # Save JSON file
            json_filename = f"{Path(pdf_path).stem}_extracted.json"
            json_output_path = OUT_JSON_DIR / json_filename

            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)

            print(f"✓ Processed {pdf_path.name}")
            print(f"  - JSON saved to: {json_output_path}")
            print(f"  - Extracted images: {len(extracted_content['image_filenames'])}")
            print(f"  - Tables detected: {len(extracted_content['tables_html'])}")
            print()

        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}")
            traceback.print_exc()
            print()

    print("Extraction complete!")
    print(f"JSON files saved to: {OUT_JSON_DIR}")
    print(f"Extracted images saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
