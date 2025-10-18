import os
from supabase import create_client, Client
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy.pool import NullPool  


def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("Supabase URL and Key must be set in environment variables.")
        return None
        
    try:
        client: Client = create_client(url, key)
        return client
    except Exception as e:
        logger.error(f"Could not create Supabase client: {e}")
        return None

# SQLAlchemy setup
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL must be set in environment variables for SQLAlchemy connection.")
    engine = None
else:
    # Ensure the URL is in the correct format for SQLAlchemy
    db_url = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://")
    engine = create_engine(db_url,poolclass=NullPool)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def insert_file_metadata(metadata: dict):
    supabase = get_supabase_client()
    if not supabase:
        raise Exception("Supabase client could not be initialized for metadata insertion.")

    try:
        data, count = supabase.table('file').insert(metadata).execute()
        logger.info(f"Inserted metadata for {metadata.get('file_original_filename')}")
        return data
    except Exception as e:
        logger.error(f"Error inserting metadata: {e}")
        raise
