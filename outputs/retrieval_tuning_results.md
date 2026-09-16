# Retrieval Tuning Results

This deterministic experiment evaluates three queries against a small metadata-aware corpus. Relevance is measured by source hit rate and top-1 hit rate.

## Test Queries

| Query | Expected source |
| --- | --- |
| How can a learner reset their password? | `account-guide.md` |
| When does the cafeteria menu change? | `campus-guide.md` |
| What evidence is required for project submission? | `submission-rubric.md` |

## Compared Settings

| Setting | Chunk size | k | Filter | Min score | Hit rate | Top-1 hit rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `baseline_k3` | 40 | 3 | none | 0.00 | 100% | 100% |
| `filtered_k3` | 40 | 3 | guide | 0.00 | 67% | 67% |
| `strict_k5` | 40 | 5 | none | 0.50 | 67% | 67% |

## Query-Level Results

### `baseline_k3`

| Query | Returned sources | Hit |
| --- | --- | :---: |
| How can a learner reset their password? | `account-guide.md`, `news-brief.txt`, `submission-rubric.md` | yes |
| When does the cafeteria menu change? | `campus-guide.md`, `account-guide.md`, `news-brief.txt` | yes |
| What evidence is required for project submission? | `submission-rubric.md`, `campus-guide.md`, `account-guide.md` | yes |

### `filtered_k3`

| Query | Returned sources | Hit |
| --- | --- | :---: |
| How can a learner reset their password? | `account-guide.md`, `campus-guide.md` | yes |
| When does the cafeteria menu change? | `campus-guide.md`, `account-guide.md` | yes |
| What evidence is required for project submission? | `campus-guide.md`, `account-guide.md` | no |

### `strict_k5`

| Query | Returned sources | Hit |
| --- | --- | :---: |
| How can a learner reset their password? | `account-guide.md` | yes |
| When does the cafeteria menu change? | `campus-guide.md` | yes |
| What evidence is required for project submission? | none | no |

## Decision

Choose `baseline_k3`: it achieved a 100% source hit rate and 100% top-1 hit rate on all three queries. Its guide filter was not used, which avoids dropping the rubric source needed by the third query. This is a small offline benchmark, so it should be rerun with production queries before rollout.
