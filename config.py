import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vectordb")
    
    EMBEDDING_COST_PER_1M = float(os.getenv("EMBEDDING_COST_PER_1M", "0.00"))
    LLM_INPUT_COST_PER_1M = float(os.getenv("LLM_INPUT_COST_PER_1M", "0.00"))
    LLM_OUTPUT_COST_PER_1M = float(os.getenv("LLM_OUTPUT_COST_PER_1M", "0.00"))
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided knowledge base.

IMPORTANT RULES:
1. Answer questions using ONLY the information from the Context below
2. If the answer is not in the Context, say: "I don't have that information in my knowledge base."
3. Do NOT make up information or use knowledge outside the provided Context
4. Be concise but thorough in your answers
5. If you're unsure, admit it rather than guessing

Context from knowledge base:
{context}

Conversation History:
{history}

User Question: {question}

Answer:"""

config = Config()
