from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime

from models import UploadRequest, UploadResponse, ChatRequest, BotStats, ChatMetadata
from chunker import IntelligentChunker
from vector_store import vector_store  # Now uses FAISS - fast and easy to install!
from llm_client import llm_client
from stats_tracker import stats_tracker
from content_fetcher import content_fetcher
from config import config

# Initialize FastAPI app
app = FastAPI(
    title="EzeeChatBot API",
    description="A minimal RAG chatbot API for grounded question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chunker = IntelligentChunker()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "EzeeChatBot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload knowledge base",
            "chat": "POST /chat - Chat with a bot",
            "stats": "GET /stats/{bot_id} - Get bot statistics"
        }
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_content(request: UploadRequest):
    """
    Upload content to create a new chatbot
    
    Accepts either direct text content or a URL to fetch content from.
    Returns a bot_id that can be used for chat interactions.
    """
    try:
        # Validate that either content or url is provided
        if not request.content and not request.url:
            raise HTTPException(
                status_code=400,
                detail="Either 'content' or 'url' must be provided"
            )
        
        # Get the content
        if request.url:
            content = content_fetcher.fetch_from_url(request.url)
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch content from URL: {request.url}"
                )
            source_metadata = {"source": "url", "url": request.url}
        else:
            content = request.content
            source_metadata = {"source": "direct_input"}
        
        # Validate content length
        if len(content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Content is too short. Please provide at least 50 characters."
            )
        
        # Chunk the content
        chunks = chunker.chunk_text(content, source_metadata)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to create chunks from content"
            )
        
        # Create bot and store embeddings
        bot_id, embedding_tokens = vector_store.create_bot(chunks)
        
        # Initialize stats tracking
        stats_tracker.initialize_bot(bot_id)
        stats_tracker.record_embedding_tokens(bot_id, embedding_tokens)
        
        return UploadResponse(
            bot_id=bot_id,
            chunks_created=len(chunks),
            message=f"Successfully created bot with {len(chunks)} knowledge chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with a bot using its knowledge base
    
    Streams the response back to the client in real-time.
    The bot will only answer based on its uploaded knowledge base.
    """
    try:
        # Verify bot exists
        if not vector_store.bot_exists(request.bot_id):
            raise HTTPException(
                status_code=404,
                detail=f"Bot {request.bot_id} not found. Please upload content first."
            )
        
        # Start timing
        start_time = time.time()
        
        # Retrieve relevant context
        context_chunks = vector_store.query_bot(request.bot_id, request.user_message)
        
        # Check if we have relevant context
        if not context_chunks or all(not chunk.strip() for chunk in context_chunks):
            # No relevant context found - return graceful fallback
            fallback_message = (
                "I don't have that information in my knowledge base. "
                "I can only answer questions based on the content that was uploaded to me."
            )
            
            # Record this as an unanswered question
            response_time = (time.time() - start_time) * 1000
            metadata = ChatMetadata(
                bot_id=request.bot_id,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                input_tokens=llm_client.count_tokens(request.user_message),
                output_tokens=llm_client.count_tokens(fallback_message),
                cost_usd=0.0,
                was_answered=False
            )
            stats_tracker.record_chat(metadata)
            
            # Return as streaming response for consistency
            async def fallback_stream():
                yield fallback_message
            
            return StreamingResponse(
                fallback_stream(),
                media_type="text/plain"
            )
        
        # Generate streaming response
        async def generate():
            full_response = []
            
            async for chunk in llm_client.generate_response_stream(
                request.user_message,
                context_chunks,
                request.conversation_history
            ):
                full_response.append(chunk)
                yield chunk
            
            # After streaming completes, record stats
            response_time = (time.time() - start_time) * 1000
            complete_response = "".join(full_response)
            
            # Estimate tokens (we don't have exact count in streaming mode)
            input_tokens = llm_client.count_tokens(
                request.user_message + " ".join(context_chunks)
            )
            output_tokens = llm_client.count_tokens(complete_response)
            
            # Check if this was an unanswered question
            was_answered = not _is_unanswered(complete_response)
            
            metadata = ChatMetadata(
                bot_id=request.bot_id,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                was_answered=was_answered
            )
            stats_tracker.record_chat(metadata)
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats/{bot_id}", response_model=BotStats)
async def get_stats(bot_id: str):
    """
    Get statistics for a specific bot
    
    Returns:
    - Total messages served
    - Average response latency in milliseconds
    - Estimated token cost in USD
    - Count of unanswered questions
    """
    try:
        # Check if bot exists
        if not vector_store.bot_exists(bot_id):
            raise HTTPException(
                status_code=404,
                detail=f"Bot {bot_id} not found"
            )
        
        stats = stats_tracker.get_bot_stats(bot_id)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def _is_unanswered(response: str) -> bool:
    """Helper to check if response indicates inability to answer"""
    unanswered_phrases = [
        "don't have that information",
        "not in my knowledge base",
        "cannot find",
        "don't know",
        "not available in the context"
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in unanswered_phrases)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
