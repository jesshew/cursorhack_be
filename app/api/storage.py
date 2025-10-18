from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.supabase_service import get_supabase_client
import datetime

router = APIRouter()

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client could not be initialized."
        )

    bucket_name = "file"
    timestamp = get_current_timestamp()
    file_path = f"{timestamp}_{file.filename}"
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Upload the file to Supabase storage
        response = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        return {"message": "File uploaded successfully", "path": file_path}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {e}"
        )
