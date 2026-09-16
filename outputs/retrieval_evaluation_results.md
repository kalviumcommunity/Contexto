# Retrieval Evaluation Results

Labels were manually defined for 3 queries using stable `12`-word chunk IDs. Metrics are macro-averaged recall@k and precision@k.

## Labelled Query Set

| Query | Relevant chunk IDs |
| --- | --- |
| How can a learner reset their password? | `account-guide.md:0`, `account-guide.md:1` |
| When does the cafeteria menu change? | `campus-guide.md:0`, `campus-guide.md:1` |
| What evidence is required for project submission? | `submission-rubric.md:0`, `submission-rubric.md:1` |

## Aggregate Metrics

| k | Recall@k | Precision@k |
| ---: | ---: | ---: |
| 3 | 67% | 44% |
| 5 | 83% | 33% |
| 10 | 100% | 22% |

## Query-Level Results

### k=3

| Query | Retrieved IDs | Hits | Recall | Precision |
| --- | --- | --- | ---: | ---: |
| How can a learner reset their password? | `account-guide.md:0`, `account-guide.md:1`, `news-brief.txt:0` | `account-guide.md:0`, `account-guide.md:1` | 100% | 67% |
| When does the cafeteria menu change? | `campus-guide.md:0`, `account-guide.md:0`, `account-guide.md:1` | `campus-guide.md:0` | 50% | 33% |
| What evidence is required for project submission? | `submission-rubric.md:0`, `campus-guide.md:0`, `account-guide.md:0` | `submission-rubric.md:0` | 50% | 33% |

### k=5

| Query | Retrieved IDs | Hits | Recall | Precision |
| --- | --- | --- | ---: | ---: |
| How can a learner reset their password? | `account-guide.md:0`, `account-guide.md:1`, `news-brief.txt:0`, `news-brief.txt:1`, `submission-rubric.md:0` | `account-guide.md:0`, `account-guide.md:1` | 100% | 40% |
| When does the cafeteria menu change? | `campus-guide.md:0`, `account-guide.md:0`, `account-guide.md:1`, `campus-guide.md:1`, `news-brief.txt:0` | `campus-guide.md:0`, `campus-guide.md:1` | 100% | 40% |
| What evidence is required for project submission? | `submission-rubric.md:0`, `campus-guide.md:0`, `account-guide.md:0`, `account-guide.md:1`, `account-guide.md:2` | `submission-rubric.md:0` | 50% | 20% |

### k=10

| Query | Retrieved IDs | Hits | Recall | Precision |
| --- | --- | --- | ---: | ---: |
| How can a learner reset their password? | `account-guide.md:0`, `account-guide.md:1`, `news-brief.txt:0`, `news-brief.txt:1`, `submission-rubric.md:0`, `account-guide.md:2`, `campus-guide.md:0`, `campus-guide.md:1`, `submission-rubric.md:1` | `account-guide.md:0`, `account-guide.md:1` | 100% | 22% |
| When does the cafeteria menu change? | `campus-guide.md:0`, `account-guide.md:0`, `account-guide.md:1`, `campus-guide.md:1`, `news-brief.txt:0`, `account-guide.md:2`, `news-brief.txt:1`, `submission-rubric.md:0`, `submission-rubric.md:1` | `campus-guide.md:0`, `campus-guide.md:1` | 100% | 22% |
| What evidence is required for project submission? | `submission-rubric.md:0`, `campus-guide.md:0`, `account-guide.md:0`, `account-guide.md:1`, `account-guide.md:2`, `campus-guide.md:1`, `news-brief.txt:0`, `news-brief.txt:1`, `submission-rubric.md:1` | `submission-rubric.md:0`, `submission-rubric.md:1` | 100% | 22% |

## Failure Analysis

- **k=3, When does the cafeteria menu change?**: missing `campus-guide.md:1`. Likely cause: chunking or small-k limit: the second relevant chunk ranked below the cutoff.
- **k=3, What evidence is required for project submission?**: missing `submission-rubric.md:1`. Likely cause: chunking or small-k limit: the second relevant chunk ranked below the cutoff.
- **k=5, What evidence is required for project submission?**: missing `submission-rubric.md:1`. Likely cause: chunking or small-k limit: the second relevant chunk ranked below the cutoff.

## Next Improvement

Choose k=10 for recall-sensitive retrieval: it reaches 100% recall@k and 22% precision@k on this set. The k=3 failures show that the second relevant chunk is pushed below the cutoff; the next experiment should test smaller semantic chunks or re-ranking while monitoring precision and context cost.
