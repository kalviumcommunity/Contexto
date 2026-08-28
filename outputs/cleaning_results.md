# Text Cleaning Results

The text cleaning pipeline was tested using the Contexto sample corpus.

## Cleaning Operations

The same cleaning function is applied to every loaded document.

- Unicode normalization using NFKC
- Common encoding artifact cleanup
- Line-ending normalization
- Removal of page-number boilerplate
- Removal of repeated archive headers
- Whitespace and tab normalization
- Reduction of excessive blank lines

## Before / After Example

### Before

CONTEXTO MEDIA ARCHIVE

Contexto helps journalists retrieve accurate historical information.   This   document contains research notes.

Page 3 of 12

The journalist's archive contains historical research and interview notes.

CONTEXTO MEDIA ARCHIVE

The organisationâ€™s records provide useful context for current reporting.

Page 4 of 12

This document demonstrates text cleaning for the RAG pipeline.

### After

Contexto helps journalists retrieve accurate historical information. This document contains research notes.

The journalist's archive contains historical research and interview notes.

The organisation's records provide useful context for current reporting.

This document demonstrates text cleaning for the RAG pipeline.

## Corpus Results

```text
sample.html: 76 -> 76 chars
sample.md: 126 -> 125 chars
sample.pdf: 78 -> 77 chars
sample.txt: 405 -> 321 chars