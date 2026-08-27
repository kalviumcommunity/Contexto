# Token Counts and Cost Estimate

Generated with `python prompts/token_cost_estimator.py` using the `cl100k_base` tokenizer.
The rates are illustrative: `$0.0005` per 1K input tokens and `$0.0015` per 1K output tokens.
Always replace them with the active provider's pricing before budgeting.

## Sample results

| Sample | Input chars | Input words | Input tokens | Output chars | Output words | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short question | 26 | 5 | 6 | 48 | 8 | 10 |
| Research paragraph | 175 | 22 | 27 | 104 | 12 | 15 |
| Full README document | 3,083 | 435 | 695 | 115 | 18 | 20 |

The three samples show the expected relationship: longer text has more tokens, but the ratios differ. The short question averages about 4.3 characters per token, while the README averages about 4.4 and includes Markdown structure and code formatting that tokenize differently from ordinary prose. Character length and token count are useful correlated measurements, not interchangeable ones.

## Cost estimate

Using `input_tokens / 1000 * $0.0005 + output_tokens / 1000 * $0.0015`:

- Short question: `$0.000018`
- Research paragraph: `$0.000036`
- Full README document: `$0.000378`
- All three samples: `$0.000432`

The script prints the exact values for the checked-out project files. Run it with:

```bash
python prompts/token_cost_estimator.py
```