# EzeeChatBot (Minimal RAG Chatbot API)

Simple backend API where users can upload content (text or URL) and ask questions based only on that content.



## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```


Create `.env`:
```
GEMINI_API_KEY=your_api_key
```


Run:
```bash
python main.py
```


## API

### POST /upload
```json
{ "content": "your text" }
```
### POST /chat
```json
{ "bot_id": "...", "user_message": "..." }
```


### GET /stats/{bot_id}


## Chunking Strategy
Text is split using sentence boundaries instead of fixed size.

* Multiple sentences per chunk
* Small overlap between chunks
* Keeps context intact for better retrieval

## Hallucination Handling
If answer is not found in the content, the bot responds:
"I could not find this in the knowledge base."

## Stack
FastAPI, FAISS, sentence-transformers, Gemini API
