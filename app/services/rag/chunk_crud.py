"""
CRUD operations for Chunk entities.
"""

from sqlalchemy.orm import Session
from app.schemas.db.chunk import ChunkDB
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_chunk(db: Session, chunk_data: ChunkDB) -> ChunkDB:
    """
    Create a new chunk in the database.
    
    Args:
        db: Database session
        chunk_data: ChunkDB object to create
        
    Returns:
        Created ChunkDB object
    """
    try:
        db.add(chunk_data)
        db.commit()
        db.refresh(chunk_data)
        return chunk_data
    except Exception as e:
        logger.error(f"Error creating chunk: {str(e)}")
        db.rollback()
        raise


def get_chunk_by_id(db: Session, chunk_id: int) -> Optional[ChunkDB]:
    """
    Retrieve a chunk by its ID.
    
    Args:
        db: Database session
        chunk_id: ID of the chunk to retrieve
        
    Returns:
        ChunkDB object if found, None otherwise
    """
    return db.query(ChunkDB).filter(ChunkDB.chunk_id == chunk_id).first()

def get_chunk_count_by_file_name(db: Session, file_name: str) -> int:
    """
    Get the total number of chunks for a file.
    
    Args:
        db: Database session
        file_name: Name of the file
        
    Returns:
        Number of chunks
    """
    return db.query(ChunkDB).filter(ChunkDB.file_name == file_name).count()


def delete_chunks_by_file_name(db: Session, file_name: str) -> int:
    """
    Delete all chunks for a specific file.
    
    Args:
        db: Database session
        file_name: Name of the file
        
    Returns:
        Number of chunks deleted
    """
    try:
        deleted_count = db.query(ChunkDB).filter(ChunkDB.file_name == file_name).delete()
        db.commit()
        logger.info(f"Deleted {deleted_count} chunks for file_name {file_name}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error deleting chunks for file_name {file_name}: {str(e)}")
        db.rollback()
        raise

