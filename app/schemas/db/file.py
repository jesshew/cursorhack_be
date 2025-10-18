from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from sqlalchemy import Column, Integer, Text, Boolean, Numeric, Float
from sqlalchemy.dialects.postgresql import TIMESTAMP
from .common import Base


# ---------- Pydantic Model ----------
class File(BaseModel):
    file_id: int
    file_full_path: Optional[str] = None
    file_original_filename: Optional[str] = None
    file_stored_object_name: Optional[str] = None
    file_upload_status: Optional[bool] = None
    file_size_mb: Optional[float] = None
    file_uploaded_date: Optional[datetime] = None
    file_source: Optional[str] = None  # e.g. 'pdf' or 'txt'
    file_bucket: Optional[str] = None
    file_extraction_time_seconds: Optional[float] = None

    class Config:
        orm_mode = True


# ---------- SQLAlchemy Model ----------
class FileDB(Base):
    __tablename__ = "file"

    file_id = Column(Integer, primary_key=True, index=True)
    file_full_path = Column(Text, nullable=True)
    file_original_filename = Column(Text, nullable=True)
    file_stored_object_name = Column(Text, nullable=True)
    file_upload_status = Column(Boolean, nullable=True)
    file_size_mb = Column(Numeric, nullable=True)
    file_uploaded_date = Column(TIMESTAMP(timezone=True), nullable=True)
    file_source = Column(Text, nullable=True)  # e.g. 'pdf' or 'txt'
    file_bucket = Column(Text, nullable=True)
    file_extraction_time_seconds = Column(Float, nullable=True)
