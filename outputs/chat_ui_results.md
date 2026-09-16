# Chat Interface & Query UI

## Implemented Features
- Question input
- Ask button
- Loading spinner
- Grounded answer display
- Retrieved source names and chunk IDs
- API error handling
- Empty-question validation
- Configurable RAG API URL

## API Integration
The frontend sends POST requests to  with a JSON question and displays the returned answer, sources, and status.

## Sample Question
How can I reset my password?

## Error Handling
When the RAG backend is unavailable, the UI displays a clear connection error instead of failing silently.

## Local Run
The Streamlit interface was successfully launched locally at .
