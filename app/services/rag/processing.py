"""
RAG Processing Service for Text File Chunking and Embedding

This service adapts the consolidated RAG sample code to work with the existing
database schema, processing text files by file ID and storing chunks in the
database with file_id references instead of document names.
"""

import hashlib
import asyncio
import time
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from concurrent.futures import ThreadPoolExecutor

from app.schemas.db.chunk import ChunkDB
from app.schemas.db.file import FileDB
from app.schemas.rag import EnrichedChunk

logger = logging.getLogger(__name__)

from app.services.rag.utils import (
    split_text_markdown,
    enrich_chunks_with_metadata_and_file,
    generate_embedding,
    EMBEDDING_BATCH_SIZE,
)
from app.services.openai_service import openai_session

# Configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 5
BATCH_SIZE = 30


class TextFileRAGProcessor:
    """
    Process text files for RAG pipeline using existing database schema.
    Adapts the consolidated sample to work with file_id instead of document_name.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_chunk_hash(self, text: str, file_name: str, chunk_index: int = 0) -> str:
        """Generate a consistent hash for a text chunk for duplicate detection."""
        unique_string = f"file_{file_name}_{chunk_index}_{text}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
    
    async def generate_embeddings_async(self, texts: List[str], max_workers: int = 8) -> List[List[float]]:
        """
        Generate embeddings asynchronously for a list of texts with parallel processing.
        
        Args:
            texts: List of texts to embed
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []

        start_time = time.time()

        # Split texts into batches for sequential processing
        batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
        all_embeddings = []

        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} for embeddings")
            try:
                with openai_session() as client:
                    response = client.embeddings.create(
                        input=batch,
                        model=DEFAULT_EMBEDDING_MODEL
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating batch embeddings for batch {i+1}: {str(e)}")
                # Fallback to individual generation for this batch
                batch_embeddings = []
                for text in batch:
                    try:
                        embedding = generate_embedding(text)
                        batch_embeddings.append(embedding)
                    except Exception as inner_e:
                        logger.error(f"Error generating individual embedding: {str(inner_e)}")
                        batch_embeddings.append([0.0] * 1536)  # Fallback embedding
                all_embeddings.extend(batch_embeddings)

            # Wait for 5 seconds between batches, but not after the last one
            if i < len(batches) - 1:
                logger.info("Waiting 5 seconds before next batch...")
                await asyncio.sleep(5)

        end_time = time.time()
        logger.info(f"Generated {len(all_embeddings)} embeddings in {end_time - start_time:.2f} seconds")

        return all_embeddings
    
    async def store_chunks_to_db(
        self,
        file_name: str,
        enriched_chunks: List[EnrichedChunk],
        chunk_embeddings: List[List[float]]
    ) -> None:
        """Store chunks to the database using our schema."""
        try:
            logger.info(f"Storing {len(enriched_chunks)} chunks for file_name: {file_name}")

            # Prepare chunk records for insertion
            chunk_records = []
            for i, (enriched_chunk, embedding) in enumerate(zip(enriched_chunks, chunk_embeddings)):
                chunk_text = enriched_chunk.original_chunk
                chunk_hash = self.generate_chunk_hash(chunk_text, file_name, i)

                chunk_record = ChunkDB(
                    file_name=file_name,
                    chunk_index=i,
                    chunk_content=chunk_text,
                    chunk_metadata=enriched_chunk.metadata.model_dump(),
                    chunk_embedding_vector=embedding,
                    chunk_num_of_char=len(chunk_text),
                    chunk_hash=chunk_hash
                )
                chunk_records.append(chunk_record)

            # Insert chunks in batches
            for i in range(0, len(chunk_records), BATCH_SIZE):
                batch = chunk_records[i:i + BATCH_SIZE]
                
                # Check for existing chunks with same hash to avoid duplicates
                existing_hashes = [record.chunk_hash for record in batch]
                existing_chunks = self.db.query(ChunkDB).filter(
                    ChunkDB.chunk_hash.in_(existing_hashes)
                ).all()
                existing_hash_set = {chunk.chunk_hash for chunk in existing_chunks}
                
                # Only insert new chunks
                new_chunks = [record for record in batch if record.chunk_hash not in existing_hash_set]
                
                if new_chunks:
                    self.db.add_all(new_chunks)
                    logger.info(f"Added batch {i//BATCH_SIZE + 1} with {len(new_chunks)} new chunks")
                else:
                    logger.info(f"Batch {i//BATCH_SIZE + 1}: All chunks already exist, skipping")

            self.db.commit()
            logger.info(f"Successfully stored all chunks for file_name {file_name}")

        except Exception as e:
            logger.error(f"Error storing chunks for file_name {file_name}: {str(e)}")
            self.db.rollback()
            raise
    
    async def process_text_content(self, file_name: str, text_content: str) -> Dict[str, Any]:
        """
        Process raw text content through the RAG pipeline.
        
        Args:
            file_name: The name of the file associated with the text
            text_content: The raw text content to process
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing text for file: {file_name}")

        try:
            if not text_content.strip():
                logger.warning(f"Text content for {file_name} is empty")
                return {
                    'file_name': file_name,
                    'status': 'skipped',
                    'reason': 'Empty content'
                }

            # Split text using markdown splitter
            logger.info(f"Processing {file_name} with markdown splitter")
            markdown_chunks = split_text_markdown(text_content)
            chunk_texts = [chunk.text for chunk in markdown_chunks]

            if not chunk_texts:
                logger.warning(f"No chunks generated for {file_name}")
                return {
                    'file_name': file_name,
                    'status': 'completed',
                    'chunks_processed': 0
                }
            
            # Create an in-memory FileDB object to pass to the enrichment function
            file_obj = FileDB(
                file_original_filename=file_name,
                file_source="text"
            )

            # Enrich chunks with metadata
            enriched_chunks = await enrich_chunks_with_metadata_and_file(
                chunks=chunk_texts,
                file=file_obj,
                document_text=text_content
            )

            if not enriched_chunks:
                logger.warning(f"No chunks were enriched for {file_name}")
                return {
                    'file_name': file_name,
                    'status': 'completed',
                    'chunks_processed': 0,
                    'enriched_chunks': 0
                }
            
            # Generate embeddings for the enriched content
            texts_to_embed = [chunk.enriched_chunk_to_be_embedded for chunk in enriched_chunks]
            embeddings = await self.generate_embeddings_async(texts_to_embed)

            # Store in database
            await self.store_chunks_to_db(
                file_name=file_name,
                enriched_chunks=enriched_chunks,
                chunk_embeddings=embeddings
            )

            logger.info(f"Successfully processed {file_name}: {len(enriched_chunks)} chunks")
            return {
                'file_name': file_name,
                'status': 'completed',
                'chunks_processed': len(enriched_chunks)
            }

        except Exception as e:
            logger.error(f"Error processing text for {file_name}: {str(e)}")
            return {
                'file_name': file_name,
                'status': 'error',
                'error': str(e)
            }
    
