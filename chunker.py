import re
from typing import List, Dict
from config import config

class IntelligentChunker:
    """
    Implements a semantic chunking strategy that:
    1. Respects sentence boundaries (doesn't cut mid-sentence)
    2. Uses overlapping chunks for context continuity
    3. Preserves paragraph structure when possible
    4. Adds metadata for better retrieval
    """
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, source_metadata: Dict = None) -> List[Dict]:
        """
        Chunk text intelligently with semantic boundaries
        
        Args:
            text: The text to chunk
            source_metadata: Optional metadata about the source
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean and normalize the text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        # Build chunks respecting sentence boundaries
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If single sentence exceeds chunk_size, split it carefully
            if sentence_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        " ".join(current_chunk), 
                        len(chunks), 
                        source_metadata
                    ))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by punctuation
                sub_chunks = self._split_long_sentence(sentence)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        sub, 
                        len(chunks), 
                        source_metadata
                    ))
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk_dict(
                    " ".join(current_chunk), 
                    len(chunks), 
                    source_metadata
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                " ".join(current_chunk), 
                len(chunks), 
                source_metadata
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and normalize text"""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex that handles common cases:
        - Mr., Dr., etc.
        - Abbreviations
        - Decimal numbers
        """
        # This pattern splits on .!? followed by space and capital letter
        # while avoiding common abbreviations
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence at commas, semicolons, or by character limit"""
        # Try splitting at natural breaks
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
        
        # If still too long, do hard split
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Hard split by character limit
                for i in range(0, len(chunk), self.chunk_size):
                    final_chunks.append(chunk[i:i + self.chunk_size])
        
        return final_chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """Get the last few sentences that fit within overlap_size"""
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
        """Create a chunk dictionary with text and metadata"""
        chunk = {
            "text": text,
            "chunk_index": index,
            "char_count": len(text),
        }
        
        if source_metadata:
            chunk.update(source_metadata)
        
        return chunk
