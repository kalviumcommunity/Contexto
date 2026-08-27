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

## Run the live comparison

```bash
python prompts/prompt_experiment.py
```

Configure `OPENAI_API_KEY` and `CHAT_MODEL` in `.env` first. The script never prints the key.