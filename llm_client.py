import google.generativeai as genai
from typing import List, AsyncGenerator, Tuple
from config import config
from models import Message

class LLMClient:
    
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=config.LLM_MODEL,
            generation_config={
                "temperature": 0.3,
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
        context_text = self._format_context(context_chunks)
        history_text = self._format_history(conversation_history)
        
        prompt = config.SYSTEM_PROMPT.format(
            context=context_text,
            history=history_text,
            question=user_message
        )
        
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
        context_text = self._format_context(context_chunks)
        history_text = self._format_history(conversation_history)
        
        prompt = config.SYSTEM_PROMPT.format(
            context=context_text,
            history=history_text,
            question=user_message
        )
        
        response = self.model.generate_content(prompt)
        
        response_text = response.text
        
        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        except:
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response_text)
        
        return response_text, input_tokens, output_tokens
    
    def count_tokens(self, text: str) -> int:
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            return len(text.split())
    
    def _format_context(self, chunks: List[str]) -> str:
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[Context {i}]:\n{chunk}\n")
        
        return "\n".join(formatted)
    
    def _format_history(self, history: List[Message]) -> str:
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history[-5:]:
            role = msg.role.capitalize()
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _is_unanswered_response(self, response: str) -> bool:
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
