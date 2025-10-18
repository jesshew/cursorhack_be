from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    document_summary: str
    section_summary: str
    
class EnrichedChunk(BaseModel):
    original_chunk: str
    metadata: ChunkMetadata
    enriched_chunk_to_be_embedded: str
