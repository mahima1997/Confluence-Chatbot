#!/usr/bin/env python3
import os
import json
import PyPDF2
from pathlib import Path

def extract_pdf_text(pdf_path):
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"

            return text_content.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def update_json_with_content(json_file_path, pdf_file_path, pdf_filename):
    """Update JSON file with extracted PDF content"""
    try:
        # Read the existing JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Extract text from PDF
        pdf_text = extract_pdf_text(pdf_file_path)

        if pdf_text:
            # Update the body content with extracted text
            json_data['body']['storage']['value'] = f"<p>{pdf_text}</p>"
            json_data['body']['atlas_doc_format']['value'] = f"<p>{pdf_text}</p>"

            # Update metadata with file size and content info
            file_size = os.path.getsize(pdf_file_path)
            json_data['metadata']['pdf_export']['file_size'] = str(file_size)
            json_data['metadata']['pdf_export']['content_extracted'] = True
            json_data['metadata']['pdf_export']['pages_in_pdf'] = len(PyPDF2.PdfReader(pdf_file_path).pages)

            # Write back to JSON file
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, indent=2, ensure_ascii=False)

            print(f"Updated {json_file_path} with content from {pdf_filename}")
            return True
        else:
            print(f"No content extracted from {pdf_filename}")
            return False

    except Exception as e:
        print(f"Error updating {json_file_path}: {e}")
        return False

def main():
    """Main function to process all PDF files"""
    data_dir = Path("data")
    api_data_dir = Path("api_data/pages")

    if not data_dir.exists():
        print(f"Data directory {data_dir} not found")
        return

    if not api_data_dir.exists():
        print(f"API data directory {api_data_dir} not found")
        return

    # Process each PDF file
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        # Extract page ID from filename
        filename = pdf_file.name
        if '_' in filename:
            # Format: "Title_pageId-timestamp.pdf"
            parts = filename.replace('.pdf', '').split('_')
            if len(parts) >= 2:
                # Get the last part which should contain the page ID and timestamp
                last_part = parts[-1]
                if '-' in last_part:
                    page_id = last_part.split('-')[0]
                else:
                    print(f"Could not extract page ID from {filename}")
                    continue

                # Find corresponding JSON file
                json_file = api_data_dir / f"{page_id}.json"
                if json_file.exists():
                    update_json_with_content(json_file, pdf_file, filename)
                else:
                    print(f"No JSON file found for page ID {page_id}")
            else:
                print(f"Unexpected filename format: {filename}")
        else:
            print(f"Unexpected filename format: {filename}")

if __name__ == "__main__":
    main()
