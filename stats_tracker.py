import json
from typing import Dict
from datetime import datetime
from models import ChatMetadata, BotStats
from config import config
import os

class StatsTracker:
    """
    Tracks usage statistics for each bot:
    - Total messages
    - Response latency
    - Token costs
    - Unanswered questions
    """
    
    def __init__(self, stats_file: str = None):
        self.stats_file = stats_file or f"{config.VECTOR_DB_PATH}/stats.json"
        self.stats: Dict[str, Dict] = self._load_stats()
    
    def initialize_bot(self, bot_id: str):
        """Initialize stats for a new bot"""
        if bot_id not in self.stats:
            self.stats[bot_id] = {
                "bot_id": bot_id,
                "total_messages": 0,
                "total_response_time_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_embedding_tokens": 0,
                "unanswered_questions": 0,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat()
            }
            self._save_stats()
    
    def record_chat(self, metadata: ChatMetadata):
        """Record a chat interaction"""
        bot_id = metadata.bot_id
        
        if bot_id not in self.stats:
            self.initialize_bot(bot_id)
        
        stats = self.stats[bot_id]
        
        # Update counters
        stats["total_messages"] += 1
        stats["total_response_time_ms"] += metadata.response_time_ms
        stats["total_input_tokens"] += metadata.input_tokens
        stats["total_output_tokens"] += metadata.output_tokens
        stats["last_accessed"] = datetime.utcnow().isoformat()
        
        if not metadata.was_answered:
            stats["unanswered_questions"] += 1
        
        self._save_stats()
    
    def record_embedding_tokens(self, bot_id: str, token_count: int):
        """Record tokens used for embeddings during upload"""
        if bot_id not in self.stats:
            self.initialize_bot(bot_id)
        
        self.stats[bot_id]["total_embedding_tokens"] += token_count
        self._save_stats()
    
    def get_bot_stats(self, bot_id: str) -> BotStats:
        """Get statistics for a specific bot"""
        if bot_id not in self.stats:
            # Return empty stats if bot doesn't exist
            return BotStats(
                bot_id=bot_id,
                total_messages=0,
                average_response_latency_ms=0.0,
                estimated_token_cost_usd=0.0,
                unanswered_questions_count=0,
                created_at="",
                last_accessed=""
            )
        
        stats = self.stats[bot_id]
        
        # Calculate average latency
        avg_latency = 0.0
        if stats["total_messages"] > 0:
            avg_latency = stats["total_response_time_ms"] / stats["total_messages"]
        
        # Calculate total cost
        embedding_cost = (stats["total_embedding_tokens"] / 1_000_000) * config.EMBEDDING_COST_PER_1M
        input_cost = (stats["total_input_tokens"] / 1_000_000) * config.LLM_INPUT_COST_PER_1M
        output_cost = (stats["total_output_tokens"] / 1_000_000) * config.LLM_OUTPUT_COST_PER_1M
        total_cost = embedding_cost + input_cost + output_cost
        
        return BotStats(
            bot_id=bot_id,
            total_messages=stats["total_messages"],
            average_response_latency_ms=round(avg_latency, 2),
            estimated_token_cost_usd=round(total_cost, 4),
            unanswered_questions_count=stats["unanswered_questions"],
            created_at=stats["created_at"],
            last_accessed=stats["last_accessed"]
        )
    
    def _load_stats(self) -> Dict:
        """Load stats from disk"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load stats: {e}")
        
        return {}
    
    def _save_stats(self):
        """Save stats to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save stats: {e}")

# Global instance
stats_tracker = StatsTracker()
