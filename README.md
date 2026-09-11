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

## Evaluate answer quality

End-to-end evaluation utilities live in `prompts/evaluation.py`. Each example
declares expected answer points and source references, while the answer result
provides `answer`, `sources`, and `retrieved_chunks`:

```python
from prompts.evaluation import evaluate_test_set

summary = evaluate_test_set(test_set, answer_fn)
```

The summary reports average correctness, grounding, citation accuracy, and the
individual examples that fail at least one dimension. Run the evaluation tests
with `python -m pytest -q`.

## Run the backend API

The FastAPI backend is defined in `src/api.py`. Set `OPENAI_API_KEY` and
`VECTOR_DB_URL`, then start it with:

```bash
uvicorn src.api:app --reload
```

Send a question to `POST /query`:

```bash
curl -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"question":"What evidence is required for project submission?"}'
```

The response contains an `answer`, normalized `sources`, and a `status` such
as `answered` or `refused_weak_context`.

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