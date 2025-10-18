import re
from loguru import logger
from docling.document_converter import DocumentConverter
import tempfile
import os

def fix_heading_hierarchy(markdown_content: str) -> str:
    """Fix heading levels - convert ## headers to proper hierarchy levels."""
    lines = markdown_content.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        if line.startswith('## '):
            heading_text = line[3:].strip()
            
            # Determine proper heading level
            heading_level = 2  # default
            
            # Main titles and key sections → H1
            if (i < 5 and len(heading_text.split()) <= 6) or \
               any(keyword in heading_text.lower() for keyword in 
                   ['abstract', 'introduction', 'conclusion', 'summary', 'references', 'bibliography']):
                heading_level = 1
            
            # Numbered sections → H1
            elif re.match(r'^[IVX]+\.?\s', heading_text) or \
                 re.match(r'^[A-Z]\.?\s', heading_text) or \
                 re.match(r'^\d+\.?\s', heading_text):
                heading_level = 1
            
            # Sub-numbered sections → H2
            elif re.match(r'^\d+\.\d+', heading_text) or \
                 re.match(r'^[A-Z]\.\d+', heading_text):
                heading_level = 2
            
            # Sub-sub sections → H3
            elif re.match(r'^\d+\.\d+\.\d+', heading_text):
                heading_level = 3
            
            # ALL CAPS short text → H1
            elif heading_text.isupper() and len(heading_text.split()) <= 5:
                heading_level = 1
            
            # Very short headings → H3
            elif len(heading_text.split()) <= 2 and not heading_text.isupper():
                heading_level = 3
            
            # Common subsection keywords → H2
            elif any(keyword in heading_text.lower() for keyword in 
                    ['method', 'result', 'discussion', 'analysis', 'approach', 'technique']):
                heading_level = 2
            
            # Apply the heading level
            heading_prefix = '#' * min(heading_level, 6)
            processed_lines.append(f"{heading_prefix} {heading_text}")
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def convert_pdf_to_markdown(pdf_content: bytes) -> str:
    try:
        logger.info("Converting PDF to markdown")
        
        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_content)
            temp_pdf_path = temp_pdf.name
        
        # Initialize Docling converter
        converter = DocumentConverter()
        
        # Convert PDF
        result = converter.convert(temp_pdf_path)
        
        # Get markdown content
        markdown_content = result.document.export_to_markdown()
        
        # FIX HEADING HIERARCHY HERE
        markdown_content = fix_heading_hierarchy(markdown_content)
        
        if not markdown_content.strip():
            logger.warning("No content extracted from PDF")
            return ""
            
        logger.info("Successfully converted PDF to markdown")
        
        return markdown_content
        
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise
    finally:
        # Clean up the temporary file
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
