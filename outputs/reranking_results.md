# Chunk Re-Ranking Results

**Query:** What evidence is required for project submission?  
**Candidate set:** 9 chunks retrieved (up to 10 requested)  
**Final context:** top 3 chunks after re-ranking

The first stage uses the existing vector-style lexical score. The second stage combines query coverage, coverage of direct task terms, and a phrase match for `project submissions`.

## Before Re-Ranking: Initial Candidate Order

| Rank | Vector score | Re-rank score | Source | Metadata | Text |
| ---: | ---: | ---: | --- | --- | --- |
| 1 | 0.2857 | n/a | `submission-rubric.md` | doc_type=rubric, chunk_index=0 | Project submissions must include evidence: a working demonstration, test results, and a |
| 2 | 0.1429 | n/a | `campus-guide.md` | doc_type=guide, chunk_index=0 | The cafeteria menu changes every Monday morning. The weekly menu is posted |
| 3 | 0.0000 | n/a | `account-guide.md` | doc_type=guide, chunk_index=0 | Learners can reset a forgotten password from the account settings page. Choose |
| 4 | 0.0000 | n/a | `account-guide.md` | doc_type=guide, chunk_index=1 | Reset password, verify the account email, and follow the link before it |
| 5 | 0.0000 | n/a | `account-guide.md` | doc_type=guide, chunk_index=2 | expires. |
| 6 | 0.0000 | n/a | `campus-guide.md` | doc_type=guide, chunk_index=1 | near the entrance and on the campus services page. |
| 7 | 0.0000 | n/a | `news-brief.txt` | doc_type=article, chunk_index=0 | The student newspaper reported on a busy campus week and interviewed several |
| 8 | 0.0000 | n/a | `news-brief.txt` | doc_type=article, chunk_index=1 | learners about their daily routines. |
| 9 | 0.0000 | n/a | `submission-rubric.md` | doc_type=rubric, chunk_index=1 | short explanation of design decisions. |

## After Re-Ranking: Final Selected Context

| Rank | Vector score | Re-rank score | Source | Metadata | Text |
| ---: | ---: | ---: | --- | --- | --- |
| 1 | 0.2857 | 0.5821 | `submission-rubric.md` | doc_type=rubric, chunk_index=0 | Project submissions must include evidence: a working demonstration, test results, and a |
| 2 | 0.1429 | 0.0786 | `campus-guide.md` | doc_type=guide, chunk_index=0 | The cafeteria menu changes every Monday morning. The weekly menu is posted |
| 3 | 0.0000 | 0.0000 | `account-guide.md` | doc_type=guide, chunk_index=0 | Learners can reset a forgotten password from the account settings page. Choose |

## Trade-Off

Requesting up to 10 candidates instead of 3 gives the second stage more opportunities to recover a precise chunk, but it increases candidate transfer and scoring work. This offline scorer performs one cheap pass over 9 chunks; an LLM or cross-encoder would add model calls, latency, and per-candidate cost. The measured benefit here is precision-oriented ordering, while the candidate count keeps the extra work bounded.
