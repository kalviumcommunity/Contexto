# Prompt Comparison

## Task 1: Separate roles

The experiment sends the staff question in a `user` message and keeps the assistant's behavior in a `system` message. See `prompt_experiment.py` for the exact request sent to the model.

## Task 2: Chosen system prompt

The chosen system prompt gives Contexto a role, limits answers to supplied or retrieved information, forbids invented facts, sets a 60-word limit and professional tone, and defines an explicit fallback: `I don't know based on the available information.`

## Task 3: Same task, two variations

Both variations ask about the refund window and use the same system message. The only difference is the user prompt.

### Vague

**Input:** `Explain our refund policy.`

**Example output:** `Our refund policy allows customers to request refunds within the applicable window. The exact window and eligibility may depend on the purchase and other conditions.`

This is readable, but it is broad and does not answer the specific question in a stable format.

### Clear and constrained

**Input:** `Answer this staff question in one sentence: What is the refund window? Use only the supplied information and do not guess.`

**Example output:** `I don't know based on the available information.`

The second prompt is better because it names the exact task, requires one sentence, and reinforces the no-guessing fallback. With retrieved policy text added later, the same prompt will produce a concise grounded answer.

## Task 4: Parameter comparison

The script also sends the exact same `TEMPERATURE_PROMPT` twice with the same model. The only changed request parameter is `temperature`:

- `temperature=0.0` favors focused, repeatable answers.
- `temperature=1.0` permits more variation and can increase unsupported embellishment in a grounded task.

Both calls use the RAG guardrails in `prompt_experiment.py`:

- `max_tokens=300` caps output length and output-token cost.
- `stop=["\\n\\nUser:"]` prevents generation from continuing into a later user turn.

`top_p` is an alternative nucleus-sampling control. Tune it instead of `temperature`, rather than changing both at once. For a grounded assistant, start with `temperature=0.0` to `0.2`, keep a task-sized `max_tokens` limit, and use `stop` only when the sequence matches the expected output format.

## Task 5: Structured output

The final request asks for JSON only in this fixed shape:

```json
{"answer": "string", "source": "string"}
```

It also sends `response_format={"type": "json_object"}` and `temperature=0.0`. The response is parsed with `json.loads()` and validated before the application uses either field. If parsing or validation fails, the script retries once with an explicit JSON-only reminder; a second failure raises a clear error instead of guessing where the answer or citation ends.

## Run the live comparison

```bash
python prompts/prompt_experiment.py
```

Configure `OPENAI_API_KEY` and `CHAT_MODEL` in `.env` first. The script never prints the key. It runs the original prompt comparison and then the same-prompt temperature comparison.