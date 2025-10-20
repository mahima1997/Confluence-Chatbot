#!/usr/bin/env python3
import os
import json
import PyPDF2
import camelot
import re
from pathlib import Path

def extract_text_with_layout(pdf_path):
    """Extract text content from a PDF file while preserving layout"""
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

def extract_tables(pdf_path):
    """Extract tables from PDF using Camelot"""
    try:
        # Extract tables from PDF
        tables = camelot.read_pdf(pdf_path, pages='all')

        if len(tables) == 0:
            return []

        extracted_tables = []
        for i, table in enumerate(tables):
            # Convert table to HTML format that Confluence can understand
            df = table.df

            # Create HTML table
            html_table = '<table class="confluenceTable">'
            html_table += '<thead><tr>'

            # Add headers (first row)
            for col in df.columns:
                html_table += f'<th class="confluenceTh">{str(col)}</th>'
            html_table += '</tr></thead>'

            html_table += '<tbody>'
            # Add data rows (skip first row as it's headers)
            for _, row in df.iterrows():
                html_table += '<tr>'
                for cell in row:
                    html_table += f'<td class="confluenceTd">{str(cell)}</td>'
                html_table += '</tr>'
            html_table += '</tbody></table>'

            extracted_tables.append({
                'table_number': i + 1,
                'html_content': html_table,
                'dataframe': df.to_dict('records')
            })

        return extracted_tables

    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
        return []

def detect_table_like_content(text_content):
    """Detect sections that look like tables based on patterns"""
    lines = text_content.split('\n')
    table_candidates = []

    # Look for lines with multiple columns (separated by multiple spaces or pipes)
    for line in lines:
        # Count spaces (indicating possible columns)
        space_count = line.count(' ')
        # Check for pipe separators
        pipe_count = line.count('|')

        # If line has many spaces or pipes, it might be a table row
        if space_count > 10 or pipe_count > 2:
            table_candidates.append(line)

    return table_candidates

def create_table_from_text_lines(text_lines):
    """Convert text lines to HTML table"""
    if not text_lines:
        return None

    html_table = '<table class="confluenceTable"><tbody>'

    for line in text_lines:
        # Clean up the line
        line = line.strip()

        # Try different separators
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        else:
            # Split by multiple spaces (assuming columns are space-separated)
            cells = re.split(r'\s{2,}', line)
            cells = [cell.strip() for cell in cells if cell.strip()]

        if len(cells) > 1:  # Only create table rows if we have multiple cells
            html_table += '<tr>'
            for cell in cells:
                # Check if this looks like a header
                is_header = any(keyword in cell.lower() for keyword in
                              ['priority', 'product', 'initiative', 'timeframe', 'level', 'effort', 'business value'])
                if is_header:
                    html_table += f'<th class="confluenceTh">{cell}</th>'
                else:
                    html_table += f'<td class="confluenceTd">{cell}</td>'
            html_table += '</tr>'

    html_table += '</tbody></table>'
    return html_table if '<tr>' in html_table else None

def update_json_with_tables(json_file_path, pdf_file_path, pdf_filename):
    """Update JSON file with table-aware content"""
    try:
        # Read the existing JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Extract regular text content
        text_content = extract_text_with_layout(pdf_file_path)

        # Extract tables using Camelot
        tables = extract_tables(pdf_file_path)

        print(f"Processing {pdf_filename}:")
        print(f"  - Text content length: {len(text_content)} characters")
        print(f"  - Tables found with Camelot: {len(tables)}")

        # If we found tables with Camelot, use them
        if tables:
            content_parts = []

            # Split text by table locations (this is a simplified approach)
            remaining_text = text_content

            for table in tables:
                # Add text before table
                text_part = f"<p>{remaining_text}</p>" if remaining_text.strip() else ""
                if text_part:
                    content_parts.append(text_part)

                # Add table
                content_parts.append(table['html_content'])

                # This is simplified - in reality we'd need better text-table separation
                remaining_text = ""

            # Add any remaining text
            if remaining_text.strip():
                content_parts.append(f"<p>{remaining_text}</p>")

            final_content = ''.join(content_parts)
        else:
            # No tables found, check for table-like content in text
            table_candidates = detect_table_like_content(text_content)

            if table_candidates:
                print(f"  - Found {len(table_candidates)} potential table lines")

                # Try to create a table from the candidates
                html_table = create_table_from_text_lines(table_candidates[:20])  # Limit to first 20 lines

                if html_table:
                    # Replace table-like content with HTML table
                    table_text = '\n'.join(table_candidates[:20])
                    final_content = text_content.replace(table_text, html_table)

                    # Wrap remaining content in paragraphs
                    other_parts = text_content.replace(table_text, '').strip()
                    if other_parts:
                        final_content = html_table + f"<p>{other_parts}</p>"
                    else:
                        final_content = html_table
                else:
                    final_content = f"<p>{text_content}</p>"
            else:
                final_content = f"<p>{text_content}</p>"

        # Update the body content
        json_data['body']['storage']['value'] = final_content
        json_data['body']['atlas_doc_format']['value'] = final_content

        # Update metadata
        file_size = os.path.getsize(pdf_file_path)
        json_data['metadata']['pdf_export']['file_size'] = str(file_size)
        json_data['metadata']['pdf_export']['content_extracted'] = True
        json_data['metadata']['pdf_export']['tables_detected'] = len(tables)
        json_data['metadata']['pdf_export']['pages_in_pdf'] = len(PyPDF2.PdfReader(pdf_file_path).pages)

        # Write back to JSON file
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=2, ensure_ascii=False)

        print(f"✅ Updated {json_file_path} with table-aware content")
        return True

    except Exception as e:
        print(f"❌ Error updating {json_file_path}: {e}")
        return False

def main():
    """Main function to process all PDF files"""
    data_dir = Path("data")
    api_data_dir = Path("api_data/pages")

    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} not found")
        return

    if not api_data_dir.exists():
        print(f"❌ API data directory {api_data_dir} not found")
        return

    # Process each PDF file
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"🔍 Found {len(pdf_files)} PDF files to process")

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
                    print(f"⚠️  Could not extract page ID from {filename}")
                    continue

                json_file = api_data_dir / f"{page_id}.json"
                if json_file.exists():
                    update_json_with_tables(json_file, pdf_file, filename)
                else:
                    print(f"⚠️  No JSON file found for page ID {page_id}")
            else:
                print(f"⚠️  Unexpected filename format: {filename}")
        else:
            print(f"⚠️  Unexpected filename format: {filename}")

if __name__ == "__main__":
    main()
