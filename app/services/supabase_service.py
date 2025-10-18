import os
from supabase import create_client, Client
from loguru import logger

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
