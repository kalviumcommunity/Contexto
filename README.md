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