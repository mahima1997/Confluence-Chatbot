#!/usr/bin/env python3
import os
import json
import PyPDF2
import camelot
import pandas as pd
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
            html_table = '<table>'
            html_table += '<thead><tr>'

            # Add headers (first row)
            for col in df.columns:
                html_table += f'<th>{str(col)}</th>'
            html_table += '</tr></thead>'

            html_table += '<tbody>'
            # Add data rows (skip first row as it's headers)
            for _, row in df.iterrows():
                html_table += '<tr>'
                for cell in row:
                    html_table += f'<td>{str(cell)}</td>'
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

def detect_table_sections(text_content):
    """Try to identify table-like sections in text content"""
    lines = text_content.split('\n')
    table_sections = []
    current_table = []

    for line in lines:
        # Look for patterns that might indicate table structure
        if any(pattern in line.lower() for pattern in ['priority', 'product', 'initiative', 'timeframe', 'level of effort']):
            if current_table and len(current_table) > 3:
                table_sections.append('\n'.join(current_table))
            current_table = [line]
        elif current_table:
            current_table.append(line)

    if current_table and len(current_table) > 3:
        table_sections.append('\n'.join(current_table))

    return table_sections

def create_confluence_table_html(text_table):
    """Convert text-based table to HTML format"""
    lines = text_table.strip().split('\n')
    if len(lines) < 2:
        return text_table

    # Try to detect columns by looking for consistent spacing or separators
    html_table = '<table><tbody>'

    for line in lines:
        if '|' in line or '\t' in line:
            # Use pipe or tab as column separator
            cells = [cell.strip() for cell in line.replace('|', '\t').split('\t') if cell.strip()]
        else:
            # Try to split by multiple spaces
            cells = [cell.strip() for cell in line.split() if cell.strip()]

        if cells:
            html_table += '<tr>'
            for cell in cells:
                # Check if this looks like a header (contains keywords)
                is_header = any(keyword in cell.lower() for keyword in
                              ['priority', 'product', 'initiative', 'timeframe', 'level', 'effort'])
                tag = 'th' if is_header else 'td'
                html_table += f'<{tag}>{cell}</{tag}>'
            html_table += '</tr>'

    html_table += '</tbody></table>'
    return html_table

def update_json_with_improved_content(json_file_path, pdf_file_path, pdf_filename):
    """Update JSON file with improved PDF content handling"""
    try:
        # Read the existing JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Extract regular text content
        text_content = extract_text_with_layout(pdf_file_path)

        # Extract tables using Camelot
        tables = extract_tables(pdf_file_path)

        # If no tables found with Camelot, try to detect table-like sections in text
        if not tables:
            table_sections = detect_table_sections(text_content)
            for section in table_sections:
                html_table = create_confluence_table_html(section)
                if html_table != section:  # If we successfully converted to table
                    text_content = text_content.replace(section, html_table)

        # Build improved content with tables
        content_parts = []

        if tables:
            # Add text before first table
            text_before_first_table = text_content.split(tables[0]['html_content'])[0] if tables else text_content
            if text_before_first_table.strip():
                content_parts.append(f"<p>{text_before_first_table.strip()}</p>")

            # Add tables
            for i, table in enumerate(tables):
                content_parts.append(table['html_content'])

                # Add text between tables
                if i < len(tables) - 1:
                    text_between = text_content.split(table['html_content'])[1].split(tables[i+1]['html_content'])[0]
                    if text_between.strip():
                        content_parts.append(f"<p>{text_between.strip()}</p>")

            # Add text after last table
            text_after_last_table = text_content.split(tables[-1]['html_content'])[1]
            if text_after_last_table.strip():
                content_parts.append(f"<p>{text_after_last_table.strip()}</p>")
        else:
            # No tables found, use original text content
            content_parts.append(f"<p>{text_content}</p>")

        # Combine all parts
        final_content = ''.join(content_parts)

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

        print(f"Updated {json_file_path} with improved content from {pdf_filename}")
        print(f"  - Tables detected: {len(tables)}")
        return True

    except Exception as e:
        print(f"Error updating {json_file_path}: {e}")
        return False

def main():
    """Main function to process all PDF files with improved table handling"""
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
    print(f"Found {len(pdf_files)} PDF files to process with improved table handling")

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
                    update_json_with_improved_content(json_file, pdf_file, filename)
                else:
                    print(f"No JSON file found for page ID {page_id}")
            else:
                print(f"Unexpected filename format: {filename}")
        else:
            print(f"Unexpected filename format: {filename}")

if __name__ == "__main__":
    main()
