from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, TEXT, func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class ChunkDB(Base):
    __tablename__ = 'chunk'

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(TEXT)
    chunk_index = Column(Integer)
    chunk_content = Column(TEXT)
    chunk_metadata = Column(JSON)
    chunk_embedding_vector = Column(Vector(1536))
    created_at = Column(TIMESTAMP, server_default=func.now())
    chunk_num_of_char = Column(Integer)
    chunk_hash = Column(TEXT)
