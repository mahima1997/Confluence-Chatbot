#!/usr/bin/env python3
"""
Analyze PDFs for visual elements (diagrams, charts, images)
"""

import fitz  # PyMuPDF
from pathlib import Path
import json

def analyze_pdf_visuals(pdf_path):
    """Analyze a PDF for visual elements"""
    visuals = {
        'images': 0,
        'diagrams': 0,
        'charts': 0,
        'total_pages': 0
    }

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Get images on the page
            image_list = page.get_images(full=True)
            visuals['images'] += len(image_list)

            # Look for text that might indicate diagrams/charts
            page_text = page.get_text()

            # Simple heuristics for detecting diagrams/charts
            diagram_keywords = [
                'diagram', 'chart', 'graph', 'figure', 'illustration',
                'flowchart', 'process', 'timeline', 'gantt', 'pie chart',
                'bar chart', 'line graph', 'scatter plot'
            ]

            text_lower = page_text.lower()
            for keyword in diagram_keywords:
                if keyword in text_lower:
                    visuals['diagrams'] += 1
                    break  # Count each page only once for diagrams

        visuals['total_pages'] = len(doc)
        doc.close()

    except Exception as e:
        print(f"Error analyzing {pdf_path}: {e}")

    return visuals

def update_metadata_with_visuals(api_data_path: str = "api_data"):
    """Update JSON files with visual element information"""
    data_dir = Path("data")
    api_data_dir = Path(api_data_path) / "pages"

    if not data_dir.exists() or not api_data_dir.exists():
        print("Required directories not found")
        return

    # Process each PDF file
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"Analyzing {len(pdf_files)} PDF files for visual elements...")

    for pdf_file in pdf_files:
        # Extract page ID from filename
        filename = pdf_file.name
        if '_' in filename:
            parts = filename.replace('.pdf', '').split('_')
            if len(parts) >= 2:
                last_part = parts[-1]
                if '-' in last_part:
                    page_id = last_part.split('-')[0]
                else:
                    print(f"Could not extract page ID from {filename}")
                    continue

                # Find corresponding JSON file by reading the manifest
                json_file = None

                # Read manifest to find the correct JSON filename
                manifest_file = Path(api_data_path) / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest_data = json.load(f)

                    # Find the page with matching original filename
                    for page in manifest_data.get('pages', []):
                        if page.get('original_filename') == filename:
                            json_filename = page.get('api_file', '').replace('pages/', '')
                            json_file = api_data_dir / json_filename
                            break

                if json_file and json_file.exists():
                    visuals = analyze_pdf_visuals(pdf_file)

                    # Update JSON metadata
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                        if 'metadata' not in json_data:
                            json_data['metadata'] = {}
                        if 'pdf_export' not in json_data['metadata']:
                            json_data['metadata']['pdf_export'] = {}

                        # Add visual elements info
                        json_data['metadata']['pdf_export']['visual_elements'] = visuals
                        json_data['metadata']['pdf_export']['has_visuals'] = (
                            visuals['images'] > 0 or visuals['diagrams'] > 0
                        )

                        # Write back to JSON file
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)

                        print(f"Updated {json_file.name}: {visuals['images']} images, {visuals['diagrams']} potential diagrams")

                    except Exception as e:
                        print(f"Error updating {json_file}: {e}")
                else:
                    print(f"No JSON file found for {filename}")
            else:
                print(f"Unexpected filename format: {filename}")
        else:
            print(f"Unexpected filename format: {filename}")

def main():
    """Main function"""
    update_metadata_with_visuals()
    print("Visual elements analysis complete!")

if __name__ == "__main__":
    main()
