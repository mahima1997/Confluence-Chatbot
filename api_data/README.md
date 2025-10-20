# Confluence API Data Conversion

This directory contains manually downloaded Confluence pages converted to the format that would be returned by the Confluence REST API.

## Structure

```
api_data/
├── manifest.json          # Overview of all converted documents
├── pages/                 # Individual page JSON files
│   ├── Copy of Optimus 2025.json
│   ├── Optimus Best Practices.json
│   ├── Optimus Impact in FY24.json
│   ├── Optimus Recommendation Rejection.json
│   ├── Optimus Resources.json
│   └── Roadmap - Optimus.json
└── attachments/           # (Empty - no attachments in this export)
```

## Conversion Details

- **Source**: 6 PDF files manually downloaded from Confluence
- **Target Format**: Confluence REST API v2 JSON responses
- **Space**: OPTIMUS
- **Export Date**: 2025-01-25T22:40:17.000Z

## Page Information

1. **Copy of Optimus 2025** (Original: Copy of Optimus 2025_736e298495384451b5c606dda3d3e3f3-191025-2239-1698.pdf)
2. **Optimus Best Practices** (Original: Optimus Best Practices_0fa63a6a2b8248f4a6f0d2ade05d5882-191025-2240-1704.pdf)
3. **Optimus Impact in FY'24** (Original: Optimus Impact in FY'24_70029109679049d19945849c1664e085-191025-2240-1702.pdf)
4. **Optimus Recommendation Rejection** (Original: Optimus Recommendation Rejection_105ba7c854034b278806e34c248a0d88-191025-2240-1706.pdf)
5. **Optimus Resources** (Original: Optimus Resources_be22766d4e034e6887a914b7e48daa5a-191025-2239-1700.pdf)
6. **Roadmap - Optimus** (Original: Roadmap - Optimus_0d4a5e89693946cf9cada0991de75641-191025-2240-1708.pdf)

## Usage

Each JSON file contains the complete Confluence page metadata and structure as would be returned by the Confluence REST API. The `manifest.json` file provides an overview of all converted documents.

## Notes

- Original PDF content is referenced in the `metadata.pdf_export` section
- Page IDs were extracted from the original filenames
- Timestamps were derived from the filename timestamps
- All pages are marked as "current" status
- Space information is standardized to "OPTIMUS"
