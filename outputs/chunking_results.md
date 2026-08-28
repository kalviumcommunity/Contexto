# Document Chunking Results

## Strategies Compared

Two chunking strategies were tested on the Contexto sample corpus:

1. Fixed-size chunking with 150-character chunks and 30-character overlap.
2. Paragraph-based chunking using paragraph boundaries.

## Comparison Results

| Document | Fixed-size | Paragraph |
|---|---:|---:|
| sample.html | 1 chunk, avg 76 chars | 1 chunk, avg 76 chars |
| sample.md | 2 chunks, avg 65 chars | 2 chunks, avg 61 chars |
| sample.pdf | 1 chunk, avg 77 chars | 1 chunk, avg 77 chars |
| sample.txt | 3 chunks, avg 127 chars | 4 chunks, avg 78 chars |

## Fixed-size Example

```text
Contexto helps journalists retrieve accurate historical information. This document contains research notes.

The journalist's archive contains histori...