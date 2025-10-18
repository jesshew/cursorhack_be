"""
RAG Processing API endpoints for text file chunking and processing.

This API provides endpoints to process text files and store them as chunks
in the database for RAG (Retrieval-Augmented Generation) workflows.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
import logging
import asyncio
import json
from datetime import datetime

from app.services.rag.processing import TextFileRAGProcessor
from app.services.supabase_service import get_db, get_supabase_client
from app.services.rag.utils import generate_embedding
from app.services.rag.chunk_crud import delete_chunks_by_file_name


logger = logging.getLogger(__name__)
router = APIRouter()


# Keep tab, newline, carriage return which are valid in JSON strings
_ALLOWED_CONTROL_CHARS = {9, 10, 13}
_ALL_CONTROL_CHARS = {i for i in range(32)}
_INVALID_CONTROL_CHARS_CODES = _ALL_CONTROL_CHARS - _ALLOWED_CONTROL_CHARS
_TRANSLATION_TABLE = {i: None for i in _INVALID_CONTROL_CHARS_CODES}

def _clean_json_string(s: str) -> str:
    """Removes invalid JSON control characters from a string."""
    return s.translate(_TRANSLATION_TABLE)


class ProcessTextRequest(BaseModel):
    """Request model for processing raw text."""
    file_name: str
    text_content: str


class SearchChunksRequest(BaseModel):
    """Request model for searching chunks."""
    query: str
    match_count: int = 10
    filter_metadata: Dict[str, Any] = None


@router.delete("/chunks/file/{file_name}")
async def clear_file_chunks(
    file_name: str,
    db: Session = Depends(get_db)
):
    """
    Delete all chunks associated with a specific file.

    Args:
        file_name: The name of the file whose chunks should be deleted.
        db: Database session dependency.

    Returns:
        A confirmation message.
    """
    try:
        logger.info(f"Attempting to delete all chunks for file_name: {file_name}")
        
        deleted_count = delete_chunks_by_file_name(db, file_name)
        
        if deleted_count > 0:
            message = f"Successfully deleted {deleted_count} chunks for file_name {file_name}."
            logger.info(message)
            return {"message": message, "file_name": file_name, "deleted_chunks": deleted_count}
        else:
            message = f"No chunks found for file_name {file_name}. Nothing to delete."
            logger.info(message)
            raise HTTPException(status_code=404, detail=message)

    except Exception as e:
        error_message = f"An error occurred while deleting chunks for file_name {file_name}: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/process-text-from-file", response_model=Dict[str, Any])
async def process_text_from_file(db: Session = Depends(get_db)):
    """
    Process text from a local file for RAG pipeline.
    
    This endpoint:
    1. Reads content from a hardcoded local file 'test_chunk_file.txt'
    2. Splits text into chunks using markdown splitter
    3. Enriches chunks with metadata using AI
    4. Generates embeddings for semantic search
    5. Stores chunks in the database with the file_name reference
    
    Args:
        db: Database session dependency
        
    Returns:
        A summary of the processing results.
    """
    try:
        file_name = "recipe.txt"
        
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                text_content = f.read()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {file_name}")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail=f"Text content in {file_name} cannot be empty.")

        logger.info(f"Processing text from file: {file_name}")
        
        # Clear existing chunks for the file before processing
        # logger.info(f"Clearing existing chunks for file_name: {file_name}")
        # delete_chunks_by_file_name(db, file_name)
        
        processor = TextFileRAGProcessor(db)
        result = await processor.process_text_content(file_name, text_content)
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process text for file {file_name}: {result.get('error')}"
            )
        
        return {
            "message": f"Successfully processed text from file {file_name}",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing text from file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/process-text", response_model=Dict[str, Any])
async def process_text(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Process raw text for RAG pipeline.
    
    This endpoint:
    1. Takes raw text from the request body
    2. Generates a unique file_name based on the current timestamp
    3. Splits text into chunks using markdown splitter
    4. Enriches chunks with metadata using AI
    5. Generates embeddings for semantic search
    6. Stores chunks in the database with the generated file_name reference
    
    Args:
        request: The raw FastAPI request object containing the text content.
        db: Database session dependency
        
    Returns:
        A summary of the processing results.
    """
    try:
        text_content = (await request.body()).decode('utf-8')
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty.")

        # Generate a unique file name based on the current timestamp
        file_name = f"text-input-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.txt"

        logger.info(f"Processing text with generated file name: {file_name}")
        
        processor = TextFileRAGProcessor(db)
        result = await processor.process_text_content(file_name, text_content)
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process text for file {file_name}: {result.get('error')}"
            )
        
        return {
            "message": f"Successfully processed text for generated file {file_name}",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing raw text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/chunks/{file_name}")
async def get_file_chunks(
    file_name: str,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Retrieve chunks for a specific file.
    
    Args:
        file_name: Name of the file
        offset: Number of chunks to skip
        db: Database session dependency
        
    Returns:
        A list of chunks for the specified file.
    """
    try:
        from app.schemas.db.chunk import ChunkDB
        
        chunks_query = db.query(ChunkDB).filter(ChunkDB.file_name == file_name)
        total_count = chunks_query.count()
        
        chunks = chunks_query.order_by(ChunkDB.chunk_index).offset(offset).all()
        
        return {
            "file_name": file_name,
            "total_chunks": total_count,
            "returned_chunks": len(chunks),
            "offset": offset,
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for file {file_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search_chunks")
async def search_chunks(
    request: SearchChunksRequest
):
    """
    Search chunks using similarity search with embeddings.
    
    This endpoint:
    1. Generates an embedding for the search query
    2. Uses the match_chunks RPC function to find similar chunks
    3. Returns ranked results based on similarity
    
    Args:
        request: SearchChunksRequest containing query and optional parameters
        
    Returns:
        A list of matching chunks.
    """
    try:
        logger.info(f"Searching chunks for query: {request.query}")
        
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: generate_embedding(request.query)
        )
        
        filter_json = request.filter_metadata if request.filter_metadata else {}
        
        supabase = get_supabase_client()
        result = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": request.match_count,
                "filter_metadata": filter_json
            }
        ).execute()
        
        return {
            "query": request.query,
            "match_count": request.match_count,
            "chunks": result.data
        }
        
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
