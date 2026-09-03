# Contexto — Media Research & Attribution Assistant

Contexto is a Retrieval-Augmented Generation (RAG) application designed to help journalists quickly retrieve accurate historical context from articles, interview transcripts, and archived footage notes.

## Project Goal

The application will allow journalists to ask natural-language questions, retrieve relevant historical information, and inspect the sources used to support an answer.

## Current Assignment

This stage establishes a clean, isolated, reproducible, and secure development workspace for the RAG application.

## Project Structure

```text
Contexto/
├── data/
│   └── .gitkeep
├── src/
│   ├── .gitkeep
│   └── app.py
├── prompts/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3
- pip
- Git

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_FORK_URL>
cd Contexto
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then add the required local configuration values to `.env`.

Required variables:

```text
OPENAI_BASE_URL
OPENAI_API_KEY
CHAT_MODEL
EMBED_MODEL
```

### 6. Run the application

```bash
python src/app.py
```

## Prompt templates

The reusable grounded answer prompt is defined in `prompts/answer.py`. Callers
render it with runtime values instead of embedding prompt text in business logic:

```python
from prompts.answer import ANSWER, render

message = render(ANSWER, context=retrieved_chunks, question=user_question)
```

Update the grounding or citation rules in `prompts/answer.py` to change them for
the application and prompt experiment together.

## Verification

The workspace was successfully tested with the virtual environment activated.

The following command completed successfully:

```bash
python src/app.py
```

The application confirmed that all required environment variables were loaded and reported:

```text
Workspace setup successful!
```

## Security

- `.env` is excluded from Git.
- `.venv/` is excluded from Git.
- Local files inside `data/` are excluded from Git.
- Generated files inside `outputs/` are excluded from Git.
- `.env.example` contains variable names without real secrets.
- No real API keys or private documents should be committed.

## Next Development Stages

Future stages will implement document processing, chunking, embeddings, vector search, RAG answer generation, source attribution, and the journalist-facing interface...

## Retrieval Relevance Tuning

Run the deterministic offline retrieval experiment:

```bash
python -m src.retrieval_tuning
python -m unittest tests.test_retrieval_tuning
```

The experiment compares chunk size, `k`, metadata filtering, and minimum score
thresholds across three test queries. It reports source hit rate and top-1 hit
rate in `outputs/retrieval_tuning_results.md`. The current sample results choose
`baseline_k3` (chunk size 40, `k=3`, no filter, minimum score 0.0): it achieves
100% on both metrics, while the filtered and strict settings achieve 67%.

The corpus and queries are intentionally small and deterministic so the result
can be reproduced without API credentials. Production rollout should rerun the
same evaluation with representative queries and manually review the retrieved
chunks as well as source-level hits.

## Chunk Re-Ranking

Run the two-stage candidate retrieval and re-ranking demo:

```bash
python -m src.reranking
python -m unittest tests.test_reranking
```

It retrieves 10 candidates, scores them again, and keeps the final 3. The
before-and-after ordering, scores, metadata, selected text, and latency/cost
trade-off are recorded in `outputs/reranking_results.md`.

## Document Loading

The loader accepts PDF, TXT, Markdown, and HTML files. It returns a common document shape with the source filename preserved:

```python
{"source": "sample_article.txt", "text": "..."}
```

Run it against the included sample corpus:

```bash
python src/document_loader.py
```

Each successful file prints its extracted character count and a short sample. Unsupported, missing, corrupt, or unreadable files are reported as `SKIP` entries while the remaining corpus continues loading. PDF extraction uses `pypdf`; HTML tags are removed with Beautiful Soup.


Workflow Established
Created separate feature branches for each team member to avoid direct changes to main.
Used GitHub Issues to track tasks, assign responsibilities, and document requirements.
Followed a Pull Request-based workflow for merging changes into main.
Established code review so that changes are reviewed and approved by at least one teammate before merging.
Adopted Conventional Commits such as feat:, fix:, docs:, refactor:, and test: for clear and consistent commit history.
Linked Pull Requests with their corresponding Issues using Closes #<issue-number>.
Kept main as the stable branch containing reviewed and approved changes.
Team Workflow
Issue → Feature Branch → Changes → Commit → Push → Pull Request → Code Review → Approval → Merge