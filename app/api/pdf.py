from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger

from app.services.pdf_conversion_service import convert_pdf_to_markdown

router = APIRouter()

@router.post("/pdf/convert-to-markdown", response_class=PlainTextResponse)
async def convert_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file and returns its content as markdown.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")
    
    try:
        pdf_content = await file.read()
        logger.info(f"Received PDF file: {file.filename}, size: {len(pdf_content)} bytes")
        
        markdown_content = convert_pdf_to_markdown(pdf_content)
        
        if not markdown_content:
            raise HTTPException(status_code=500, detail="Failed to extract content from the PDF.")
        
        return PlainTextResponse(content=markdown_content)
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
