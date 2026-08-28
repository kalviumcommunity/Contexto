# Document Intake Results

The document loader was tested using a small mixed-format sample corpus.

## Sample Corpus

- `sample.txt` — Plain text
- `sample.md` — Markdown
- `sample.html` — HTML
- `sample.pdf` — PDF
- `sample.xyz` — Unsupported format used to test graceful failure

## Test Output

```text
OK sample.html: 76 chars | 'Contexto Archive This document contains archived media resea'
OK sample.md: 126 chars | '# Contexto Research Notes\n\nContexto stores historical resear'
OK sample.pdf: 78 chars | 'Contexto PDF Archive\nThis PDF contains historical media rese'
OK sample.txt: 93 chars | 'Contexto helps journalists retrieve accurate historical info'
SKIP sample.xyz: unsupported format: .xyz