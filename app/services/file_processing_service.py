import datetime
import time
from loguru import logger
from app.services.pdf_conversion_service import convert_pdf_to_markdown
from app.services.supabase_service import get_supabase_client, insert_file_metadata
from pathlib import Path

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

async def process_and_upload_pdf(file, file_content: bytes, filename: str):
    supabase = get_supabase_client()
    if not supabase:
        raise Exception("Supabase client could not be initialized.")

    # 1. Convert PDF to Markdown
    logger.info("Starting PDF to markdown conversion...")
    start_time = time.time()
    markdown_content = convert_pdf_to_markdown(file_content)
    end_time = time.time()
    extraction_time = end_time - start_time
    logger.info(f"PDF to markdown conversion took {extraction_time:.2f} seconds.")

    if not markdown_content:
        raise Exception("Failed to extract content from the PDF.")

    # 2. Upload original PDF and extracted Markdown to Supabase Storage
    timestamp = get_current_timestamp()
    pdf_object_name = f"{timestamp}_{filename}"
    md_filename = f"{Path(filename).stem}.md"
    md_object_name = f"{timestamp}_{md_filename}"
    bucket_name = "file"

    try:
        # Upload PDF
        logger.info(f"Uploading original PDF to Supabase: {pdf_object_name}")
        supabase.storage.from_(bucket_name).upload(
            path=pdf_object_name,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        logger.info("Original PDF uploaded successfully.")

        # Upload Markdown
        logger.info(f"Uploading extracted markdown to Supabase: {md_object_name}")
        supabase.storage.from_(bucket_name).upload(
            path=md_object_name,
            file=markdown_content.encode('utf-8'),
            file_options={"content-type": "text/markdown"}
        )
        logger.info("Extracted markdown uploaded successfully.")

    except Exception as e:
        logger.error(f"Error uploading files to Supabase: {e}")
        raise

    # 3. Populate the file table for both files
    file_size_mb = len(file_content) / (1024 * 1024)
    
    pdf_metadata = {
        "file_full_path": f"{bucket_name}/{pdf_object_name}",
        "file_original_filename": filename,
        "file_stored_object_name": pdf_object_name,
        "file_upload_status": True,
        "file_size_mb": file_size_mb,
        "file_uploaded_date": datetime.datetime.now().isoformat(),
        "file_source": "pdf",
        "file_bucket": bucket_name,
        "file_extraction_time_seconds": None,
    }

    md_metadata = {
        "file_full_path": f"{bucket_name}/{md_object_name}",
        "file_original_filename": md_filename,
        "file_stored_object_name": md_object_name,
        "file_upload_status": True,
        "file_size_mb": len(markdown_content.encode('utf-8')) / (1024 * 1024),
        "file_uploaded_date": datetime.datetime.now().isoformat(),
        "file_source": "txt",
        "file_bucket": bucket_name,
        "file_extraction_time_seconds": extraction_time,
    }

    try:
        logger.info("Inserting file metadata into database...")
        insert_file_metadata(pdf_metadata)
        insert_file_metadata(md_metadata)
        logger.info("File metadata inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting file metadata into database: {e}")
        # Here you might want to handle rollback of the storage uploads
        raise

    return {
        "pdf_path": f"{bucket_name}/{pdf_object_name}",
        "markdown_path": f"{bucket_name}/{md_object_name}",
        "extraction_time_seconds": extraction_time
    }

