from typing import List, Optional
import uuid
from rag_workbench.core.interfaces import ChunkingStrategy, Document

class FixedSizeChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            text = doc.content
            if not text:
                continue
            
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                
                # Create new document for chunk
                new_id = str(uuid.uuid4())
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = len(chunked_docs)
                metadata["parent_id"] = doc.id
                
                chunked_docs.append(Document(content=chunk_text, metadata=metadata, id=new_id))
                
                start += (self.chunk_size - self.chunk_overlap)
                
        return chunked_docs

class RecursiveCharacterChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, documents: List[Document]) -> List[Document]:
        # Simple implementation of recursive splitting
        # For a production-grade version, we might want to wrap LangChain's splitter
        # But here is a lightweight version to keep it dependency-free for now
        chunked_docs = []
        for doc in documents:
            chunks = self._split_text(doc.content, self.separators)
            for i, chunk_text in enumerate(chunks):
                 new_id = str(uuid.uuid4())
                 metadata = doc.metadata.copy()
                 metadata["chunk_index"] = i
                 metadata["parent_id"] = doc.id
                 chunked_docs.append(Document(content=chunk_text, metadata=metadata, id=new_id))
        return chunked_docs

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break
        
        # Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text) # Split by character if no separator found (shouldn't happen with "")

        # Merge
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = split
                # If the single split is too big, we need to recurse on it
                if len(current_chunk) > self.chunk_size and new_separators:
                    # This is a simplification; a full implementation would be more complex
                    # For now, let's just accept it might be slightly over size or implement full recursion
                    pass 
        
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks
