from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class UploadRequest(BaseModel):
    content: Optional[str] = Field(None, description="Direct text content to upload")
    url: Optional[str] = Field(None, description="URL to fetch content from")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Your knowledge base text here...",
                "url": "https://example.com/article"
            }
        }

class UploadResponse(BaseModel):
    bot_id: str
    chunks_created: int
    message: str

class Message(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: List[Message] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "bot_abc123",
                "user_message": "What is the main topic?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"}
                ]
            }
        }

class BotStats(BaseModel):
    bot_id: str
    total_messages: int
    average_response_latency_ms: float
    estimated_token_cost_usd: float
    unanswered_questions_count: int
    created_at: str
    last_accessed: str

class ChatMetadata(BaseModel):
    bot_id: str
    timestamp: datetime
    response_time_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    was_answered: bool
