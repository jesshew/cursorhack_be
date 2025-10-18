from fastapi import APIRouter, HTTPException, status
from app.services.supabase_service import get_supabase_client

router = APIRouter()

@router.get("/")
def get_health():
    return {"status": "ok"}

@router.get("/db")
def get_db_health():
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client could not be initialized. Check environment variables."
        )
    try:
        # A lightweight query to confirm the connection and table access.
        # This will select the first user from the 'user' table.
        # If your table is named differently, you'll need to update the table name here.
        response = supabase.table('HealthCheck').select('*').limit(1).execute()
        print(response)
        return {"db_status": "ok"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Supabase connection failed: {e}"
        )

