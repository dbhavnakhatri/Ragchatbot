import re
from typing import List, Dict
from config import config

class IntelligentChunker:
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, source_metadata: Dict = None) -> List[Dict]:
        text = self._clean_text(text)
        
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        " ".join(current_chunk), 
                        len(chunks), 
                        source_metadata
                    ))
                    current_chunk = []
                    current_length = 0
                
                sub_chunks = self._split_long_sentence(sentence)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        sub, 
                        len(chunks), 
                        source_metadata
                    ))
                continue
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk_dict(
                    " ".join(current_chunk), 
                    len(chunks), 
                    source_metadata
                ))
                
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                " ".join(current_chunk), 
                len(chunks), 
                source_metadata
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        parts = re.split(r'([,;])', sentence)
        
        chunks = []
        current = ""
        
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                current = part
        
        if current:
            chunks.append(current.strip())
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), self.chunk_size):
                    final_chunks.append(chunk[i:i + self.chunk_size])
        
        return final_chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        overlap = []
        current_length = 0
        
        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_size:
                overlap.insert(0, sentence)
                current_length += len(sentence)
            else:
                break
        
        return overlap
    
    def _create_chunk_dict(self, text: str, index: int, source_metadata: Dict = None) -> Dict:
        chunk = {
            "text": text,
            "chunk_index": index,
            "char_count": len(text),
        }
        
        if source_metadata:
            chunk.update(source_metadata)
        
        return chunk
