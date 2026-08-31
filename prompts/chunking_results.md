# Token-Aware Chunking with Overlap

This note demonstrates the token-based chunking approach that keeps retrieval boundaries meaningful. The sample code uses `tiktoken` and the `cl100k_base` encoding so the chunk size is expressed in tokens instead of characters.

Run it with:

```bash
python prompts/token_chunker.py
```

## Why tokens matter more than characters

A character count can be misleading because dense prose and spaced-out prose do not consume the same number of model tokens. A 500-character block of dense text can exceed the available budget, while a 500-character block with many spaces and short words may be far below it. Token-aware chunking keeps the chunk width aligned with the model's actual context budget.

## Overlap preserves boundary context

The chunker advances by `size - overlap` tokens instead of a full `size` tokens. That means adjacent chunks share a small slice of context and the retriever is less likely to cut a sentence or fact in half.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def token_chunks(text, size=400, overlap=60):
    toks = enc.encode(text)
    out, i = [], 0
    while i < len(toks):
        out.append(enc.decode(toks[i:i+size]))
        i += size - overlap
    return out
```

## Cost trade-off

Overlap increases the number of repeated tokens and therefore increases indexing cost, embedding cost, and storage. Too little overlap risks losing boundary context; too much overlap adds duplication without much retrieval gain. A common starting point is roughly 300 to 500 tokens per chunk with about 10% to 15% overlap.

## Context-window coupling

Chunk size is not independent from retrieval settings. The actual context sent to the model is roughly:

- retrieved chunk text
- prompt instructions
- any chat history
- the model's maximum context window

If chunk size grows, fewer chunks fit in the same retrieved context budget. The right chunk size must therefore be chosen jointly with retrieval depth (`top-k`) and the model's context limit.

## Sample output

```text
Tokenizer: cl100k_base
overlap 0: 1 chunks
overlap 60: 1 chunks

Boundary preview:

Chunk 1 (80 tokens)
Contexto is a newsroom research assistant built to help journalists find grounded answers from archived reports, interviews, and notes. A strong retrieval system keeps the answer faithful to the supplied sources instead of guessing from general knowledge.

Chunk 2 (80 tokens)
The chunking step matters because a document is often too large to fit into one retrieval window, and a hard cut can split a key fact in half. When text is split by tokens rather than characters, the model can work with a more predictable context budget.
```

The exact chunk count depends on the text length and the chosen overlap. The key principle is consistent: the model sees token-aware chunks, and the overlapping edges preserve the facts that sit near the boundary.

## Recommendation

For a production RAG system, a sensible starting configuration is a chunk size around 400 tokens with 10% to 15% overlap. That range is large enough to hold a meaningful fact or paragraph while small enough to stay within common context budgets. As the corpus and retrieval depth change, re-tune chunk size and overlap together rather than changing one in isolation.
