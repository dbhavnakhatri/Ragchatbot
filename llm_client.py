import google.generativeai as genai
from typing import List, AsyncGenerator, Tuple
from config import config
from models import Message

class LLMClient:
    """
    Handles interaction with Google Gemini LLM API
    Focuses on grounded responses and prevents hallucination
    """
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=config.LLM_MODEL,
            generation_config={
                "temperature": 0.3,  # Lower temperature for more grounded responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
    
    async def generate_response_stream(
        self, 
        user_message: str, 
        context_chunks: List[str], 
        conversation_history: List[Message]
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response based on context
        
        Args:
            user_message: The user's question
            context_chunks: Relevant chunks from knowledge base
            conversation_history: Previous conversation messages
            
        Yields:
            Chunks of the response text
        """
        # Build the prompt
        context_text = self._format_context(context_chunks)
        history_text = self._format_history(conversation_history)
        
        prompt = config.SYSTEM_PROMPT.format(
            context=context_text,
            history=history_text,
            question=user_message
        )
        
        # Stream response from Gemini
        response = self.model.generate_content(prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def generate_response_sync(
        self,
        user_message: str,
        context_chunks: List[str],
        conversation_history: List[Message]
    ) -> Tuple[str, int, int]:
        """
        Generate a complete response (non-streaming)
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        context_text = self._format_context(context_chunks)
        history_text = self._format_history(conversation_history)
        
        prompt = config.SYSTEM_PROMPT.format(
            context=context_text,
            history=history_text,
            question=user_message
        )
        
        response = self.model.generate_content(prompt)
        
        response_text = response.text
        
        # Estimate tokens (Gemini API provides usage metadata)
        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        except:
            # Fallback estimation if usage metadata not available
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response_text)
        
        return response_text, input_tokens, output_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        try:
            # Use Gemini's token counting
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Rough estimate if counting fails
            return len(text.split())
    
    def _format_context(self, chunks: List[str]) -> str:
        """Format context chunks for the prompt"""
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[Context {i}]:\n{chunk}\n")
        
        return "\n".join(formatted)
    
    def _format_history(self, history: List[Message]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history[-5:]:  # Only use last 5 messages to avoid context overflow
            role = msg.role.capitalize()
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _is_unanswered_response(self, response: str) -> bool:
        """
        Check if the response indicates the bot couldn't answer
        """
        unanswered_phrases = [
            "don't have that information",
            "not in my knowledge base",
            "cannot find",
            "don't know",
            "not available in the context",
            "no information about",
            "not mentioned in the knowledge base"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in unanswered_phrases)

# Global instance
llm_client = LLMClient()
