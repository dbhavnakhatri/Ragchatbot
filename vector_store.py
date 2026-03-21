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
    
    def __init__(self):
        self.storage_path = config.VECTOR_DB_PATH
        os.makedirs(self.storage_path, exist_ok=True)
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._bot_metadata = {}
        self._bot_indices = {}
        self._bot_data = {}
        
        self._load_all_bots()
    
    def create_bot(self, chunks: List[Dict]) -> tuple[str, int]:
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {k: str(v) for k, v in chunk.items() if k != "text"}
            metadatas.append(metadata)
        
        embeddings_list, token_count = self._generate_embeddings(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings_array)
        
        index.add(embeddings_array)
        
        self._bot_indices[bot_id] = index
        self._bot_data[bot_id] = {
            'texts': texts,
            'metadatas': metadatas,
            'created_at': datetime.utcnow().isoformat()
        }
        
        self._bot_metadata[bot_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks),
            "embedding_tokens": token_count
        }
        
        self._save_bot(bot_id)
        self._save_bot_metadata()
        
        return bot_id, token_count
    
    def query_bot(self, bot_id: str, query_text: str, top_k: int = 4) -> List[str]:
        if bot_id not in self._bot_indices:
            return []
        
        query_embedding_list, _ = self._generate_embeddings([query_text])
        query_embedding = np.array(query_embedding_list).astype('float32')
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self._bot_indices[bot_id].search(query_embedding, top_k)
        
        texts = self._bot_data[bot_id]['texts']
        return [texts[i] for i in indices[0] if i < len(texts)]
    
    def bot_exists(self, bot_id: str) -> bool:
        return bot_id in self._bot_indices
    
    def get_bot_metadata(self, bot_id: str) -> Dict:
        self._load_bot_metadata()
        return self._bot_metadata.get(bot_id, {})
    
    def _generate_embeddings(self, texts: List[str]) -> tuple[List[List[float]], int]:
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
                total_tokens += len(text.split())
                
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)
        
        return embeddings, total_tokens
    
    def _save_bot(self, bot_id: str):
        try:
            index_path = os.path.join(self.storage_path, f"{bot_id}.index")
            faiss.write_index(self._bot_indices[bot_id], index_path)
            
            data_path = os.path.join(self.storage_path, f"{bot_id}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(self._bot_data[bot_id], f)
        except Exception as e:
            print(f"Warning: Could not save bot {bot_id}: {e}")
    
    def _load_bot(self, bot_id: str):
        try:
            index_path = os.path.join(self.storage_path, f"{bot_id}.index")
            data_path = os.path.join(self.storage_path, f"{bot_id}.pkl")
            
            if os.path.exists(index_path) and os.path.exists(data_path):
                self._bot_indices[bot_id] = faiss.read_index(index_path)
                
                with open(data_path, 'rb') as f:
                    self._bot_data[bot_id] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load bot {bot_id}: {e}")
    
    def _load_all_bots(self):
        try:
            if os.path.exists(self.storage_path):
                for filename in os.listdir(self.storage_path):
                    if filename.endswith('.index'):
                        bot_id = filename[:-6]
                        self._load_bot(bot_id)
        except Exception as e:
            print(f"Warning: Could not load bots: {e}")
        
        self._load_bot_metadata()
    
    def _save_bot_metadata(self):
        try:
            metadata_path = os.path.join(self.storage_path, "bot_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self._bot_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save bot metadata: {e}")
    
    def _load_bot_metadata(self):
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
