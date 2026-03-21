"""
FAISS-based vector store for fast similarity search
Drop-in replacement for ChromaDB - no C++ compilation needed!
"""
import uuid
import json
import os
from typing import List, Dict
from datetime import datetime
import numpy as np
import faiss
from config import config
import google.generativeai as genai
import pickle


class VectorStore:
    """
    Manages vector embeddings and similarity search using FAISS.
    Ensures multi-bot isolation by using separate indices per bot.
    Uses Google Gemini API for embeddings.
    """
    
    def __init__(self):
        self.storage_path = config.VECTOR_DB_PATH
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Configure Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._bot_metadata = {}
        self._bot_indices = {}  # {bot_id: faiss.Index}
        self._bot_data = {}     # {bot_id: {'texts': list, 'metadatas': list}}
        
        # Load existing data
        self._load_all_bots()
    
    def create_bot(self, chunks: List[Dict]) -> tuple[str, int]:
        """
        Create a new bot with its own knowledge base using FAISS
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            Tuple of (bot_id, number of tokens used for embeddings)
        """
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"
        
        # Prepare data for insertion
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {k: str(v) for k, v in chunk.items() if k != "text"}
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings_list, token_count = self._generate_embeddings(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]  # Get embedding dimension
        index = faiss.IndexFlatIP(dimension)    # Inner Product index (for cosine similarity)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add vectors to FAISS index
        index.add(embeddings_array)
        
        # Store bot data
        self._bot_indices[bot_id] = index
        self._bot_data[bot_id] = {
            'texts': texts,
            'metadatas': metadatas,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store bot metadata
        self._bot_metadata[bot_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks),
            "embedding_tokens": token_count
        }
        
        # Persist to disk
        self._save_bot(bot_id)
        self._save_bot_metadata()
        
        return bot_id, token_count
    
    def query_bot(self, bot_id: str, query_text: str, top_k: int = 4) -> List[str]:
        """
        Retrieve most relevant chunks for a query using FAISS
        
        Args:
            bot_id: The bot to query
            query_text: The user's question
            top_k: Number of top results to return
            
        Returns:
            List of relevant text chunks
        """
        if bot_id not in self._bot_indices:
            return []
        
        # Generate query embedding
        query_embedding_list, _ = self._generate_embeddings([query_text])
        query_embedding = np.array(query_embedding_list).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search using FAISS (very fast!)
        distances, indices = self._bot_indices[bot_id].search(query_embedding, top_k)
        
        # Return top texts
        texts = self._bot_data[bot_id]['texts']
        return [texts[i] for i in indices[0] if i < len(texts)]
    
    def bot_exists(self, bot_id: str) -> bool:
        """Check if a bot exists"""
        return bot_id in self._bot_indices
    
    def get_bot_metadata(self, bot_id: str) -> Dict:
        """Get metadata for a specific bot"""
        self._load_bot_metadata()
        return self._bot_metadata.get(bot_id, {})
    
    def _generate_embeddings(self, texts: List[str]) -> tuple[List[List[float]], int]:
        """
        Generate embeddings using Google Gemini API
        
        Returns:
            Tuple of (embeddings list, estimated tokens used)
        """
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            try:
                # Generate embedding using Gemini
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
                # Estimate tokens
                total_tokens += len(text.split())
                
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Fallback to zero embedding if error
                embeddings.append([0.0] * 768)  # Default embedding dimension
        
        return embeddings, total_tokens
    
    def _save_bot(self, bot_id: str):
        """Save FAISS index and bot data to disk"""
        try:
            # Save FAISS index
            index_path = os.path.join(self.storage_path, f"{bot_id}.index")
            faiss.write_index(self._bot_indices[bot_id], index_path)
            
            # Save texts and metadata as pickle
            data_path = os.path.join(self.storage_path, f"{bot_id}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(self._bot_data[bot_id], f)
        except Exception as e:
            print(f"Warning: Could not save bot {bot_id}: {e}")
    
    def _load_bot(self, bot_id: str):
        """Load FAISS index and bot data from disk"""
        try:
            index_path = os.path.join(self.storage_path, f"{bot_id}.index")
            data_path = os.path.join(self.storage_path, f"{bot_id}.pkl")
            
            if os.path.exists(index_path) and os.path.exists(data_path):
                # Load FAISS index
                self._bot_indices[bot_id] = faiss.read_index(index_path)
                
                # Load texts and metadata
                with open(data_path, 'rb') as f:
                    self._bot_data[bot_id] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load bot {bot_id}: {e}")
    
    def _load_all_bots(self):
        """Load all bots from disk on startup"""
        try:
            if os.path.exists(self.storage_path):
                for filename in os.listdir(self.storage_path):
                    if filename.endswith('.index'):
                        bot_id = filename[:-6]  # Remove .index extension
                        self._load_bot(bot_id)
        except Exception as e:
            print(f"Warning: Could not load bots: {e}")
        
        # Load metadata
        self._load_bot_metadata()
    
    def _save_bot_metadata(self):
        """Persist bot metadata to disk"""
        try:
            metadata_path = os.path.join(self.storage_path, "bot_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self._bot_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save bot metadata: {e}")
    
    def _load_bot_metadata(self):
        """Load bot metadata from disk"""
        try:
            metadata_path = os.path.join(self.storage_path, "bot_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self._bot_metadata = json.load(f)
        except FileNotFoundError:
            self._bot_metadata = {}
        except Exception as e:
            print(f"Warning: Could not load bot metadata: {e}")
            self._bot_metadata = {}

# Global instance
vector_store = VectorStore()
