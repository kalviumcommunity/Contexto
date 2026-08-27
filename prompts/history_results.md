# Context Window History Management

The sample uses the same `cl100k_base` tokenizer as the token-cost exercise and a deliberately small 100-token request budget. Before each fake API request, the conversation counts all message content. When the count exceeds the budget, it removes the oldest complete user/assistant turn and always retains the system message.

Run it with:

```bash
python prompts/history_manager.py
```

## Sample run

```text
Tokenizer: cl100k_base
Token budget per request: 100
turn | naive tokens | request tokens | removed old turns
   1 |           39 |             39 |                  0
   2 |           65 |             65 |                  0
   3 |           91 |             91 |                  0
   4 |          117 |             91 |                  1
   5 |          143 |             91 |                  1
   6 |          169 |             91 |                  1
   7 |          195 |             91 |                  1
   8 |          221 |             91 |                  1
System preserved: True
Final history tokens: 97
```

Without trimming, turn 4 would send 117 tokens and exceed the 100-token limit. The manager removes the oldest complete turn before the request, so every request remains at or below the budget. The trade-off is explicit: older conversational detail is discarded while the system instructions and recent turns remain available.