# Streaming Responses & Citation Display

## Implemented Features

- FastAPI streaming endpoint at 
- Server-sent event format with , , , and  events
- Streamlit UI progressively displays answer tokens
- Visible citation markers
- Document names and chunk IDs displayed
- Expandable source evidence with source text
- Loading state while streaming
- Partial answers are preserved if streaming stops
- Error message and retry instruction are displayed

## Sample Question

How can I reset my password?

## Citation

[1] account-guide.md — chunk-1

Source text:
Learners can reset a forgotten password from the account settings page by choosing Reset password and following the reset link.

## Streaming Behavior

The backend emits the answer progressively as token events. The Streamlit interface updates the answer area as events arrive.

## Error Handling

If the stream cannot start or is interrupted, the UI displays:

The answer stopped streaming. Please try again.

Previously received answer text and citations remain visible.

## Local Verification

 and  compiled successfully with .
