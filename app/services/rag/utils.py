#!/usr/bin/env python3
"""
ßThis module contains all the core functionality for a Supabase-based RAG pipeline
including text splitting, embedding generation, metadata enrichment, and vector search.

All dependencies have been consolidated into this single file to minimize complexity
while maintaining full functionality.
"""


import asyncio
import time
import re

from typing import List, Any, Optional
from contextlib import contextmanager, asynccontextmanager

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from logging import getLogger

from app.services.rag.markdown_splitter import process_markdown_text
from app.schemas.db.file import FileDB
from app.schemas.rag import EnrichedChunk, ChunkMetadata
from app.services.groq_service import call_groq_api, async_call_groq_api


# Load environment variables
load_dotenv()

# Set up logger
logger = getLogger(__name__)

#==============================================================
# Pydantic Models for RAG
#==============================================================
class ChunkMetadata(BaseModel):
    document_summary: str
    section_summary: str
    
class EnrichedChunk(BaseModel):
    original_chunk: str
    metadata: ChunkMetadata
    enriched_chunk_to_be_embedded: str

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Embedding configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 2
EMBEDDING_BATCH_SIZE = 20

# Processing configuration
MAX_CONCURRENT_ENRICHMENT_TASKS = 15
BATCH_SIZE = 100

# ============================================================================
# PROMPTS
# ============================================================================

CHUNK_METADATA_EXTRACTION_PROMPT_DEFAULT = """Extract Metadata from Text Chunk

You are an assistant preparing text for a Retrieval-Augmented Generation (RAG) system.

Given the following text chunk, extract a minimal, clean JSON object with the following fields:

{
  "section_summary": "1–2 sentence summary of what is described in the chunk.",
  "themes": ["theme1", "theme2", ...],
  "keywords": ["keyword1", "keyword2", ...]
}

Keep the JSON concise — omit keys if the content doesn’t support them. Never fabricate details.
"""

CHUNK_METADATA_EXTRACTION_PROMPT = """Extract Narrative Metadata from Memoir Chunk

You are an assistant preparing text for a storytelling RAG system powered by Lee Kuan Yew’s wisdom.

Given the following memoir or reflective passage, extract a minimal, clean JSON object with the following fields:

{
  "scene_summary": "1–2 sentence summary of what is described or reflected upon.",
  "time_period": "Approximate time or era if identifiable (e.g. '1940s Cambridge', 'post-independence Singapore').",
  "themes": ["leadership", "education", "nation building", "family", ...],
  "values_or_principles": ["discipline", "pragmatism", "resilience", ...],
  "emotions": ["nostalgia", "pride", "gratitude", ...],
  "tone": "Narrative tone (e.g. reflective, persuasive, proud, contemplative)",
  "possible_user_requests": [
    "1–3 concise natural questions this chunk could answer (e.g. 'Tell me about LKY’s Cambridge years', 'What did LKY learn from his parents?')"
  ]
}

Keep the JSON concise — omit keys if the content doesn’t support them. Never fabricate details.
"""


RECIPE_METADATA_PROMPT = """Extract Culinary Metadata from Recipe Chunk

You are an assistant preparing text for a culinary storytelling RAG system that generates recipes and narratives in Lee Kuan Yew’s voice.

Given the following recipe-like passage, extract a clean JSON object with these fields:

{
  "dish_name": "Exact name of the dish (e.g. 'sotong soup')",
  "ingredients": ["explicitly mentioned ingredients"],
  "cooking_method": "boil, fry, stew, steam, etc. if mentioned",
  "cultural_context": "Any cultural or familial note if identifiable (e.g. 'Teochew home cooking', 'mother’s kitchen').",
  "culinary_values": ["patience", "simplicity", "discipline", "frugality", ...],
  "voice_prompt": "A single sentence describing how LKY might narrate this recipe (e.g. 'My mother taught me that simplicity is the soul of good cooking.')",
  "possible_user_requests": [
    "1–3 short natural queries (e.g. 'How to make sotong soup?', 'What can I cook with sotong?', 'Tell me about LKY’s family recipes.')"
  ]
}

If the passage lacks certain details, omit the missing keys. Keep the JSON short and factual.
"""



DOCUMENT_SUMMARIZATION_PROMPT = """You are a narrative and cultural analysis assistant for a storytelling Retrieval-Augmented Generation (RAG) system inspired by Lee Kuan Yew’s wisdom.

You will be given a document that may be a personal memoir, historical reflection, or a recipe (from sources such as “Lee Kuan Yew’s Singapore” or “Her Mother’s Cookbook”). 
Your goal is to produce one information-dense sentence that provides a **narrative-aware summary** useful for semantic search and storytelling.

Your summary must:

1. Identify the **content type** — memoir, recipe, or other — based on the text.
2. Summarize what the document is fundamentally about (the core story, event, or dish), grounded strictly in what is actually written.
3. Express the **underlying principles, wisdom, or values** that appear (e.g., resilience, pragmatism, family, simplicity, discipline).
4. Describe the **tone or emotion** of the text (e.g., reflective, determined, nostalgic, instructional).
5. Mention several **explicitly present topics, names, or ingredients** to improve retrieval (e.g., Cambridge, independence, sotong, ginger, Teochew cooking).

Your output must follow this exact format:

"This document is a [content_type] that tells a story about [central narrative or subject], emphasizing [key values or lessons], expressed in a [tone/emotion] tone, and includes topics or keywords such as [explicitly mentioned phrases, people, dishes, or ingredients]."

— Keep it to **one sentence only**, but make it richly descriptive.
— Base everything strictly on the provided text. Never hallucinate or infer unstated details.

DOCUMENT INFO
{truncation_message}
Document Source Info: {document_info}

DOCUMENT CONTENT
{document_text}
"""


TRUNCATION_MESSAGE = """Also note that the document text provided below is just the first ~{num_characters} characters of the document. That should be plenty for this task. Your response should still pertain to the entire document, not just the text provided below."""

# ============================================================================
# DATA MODELS
# ============================================================================

class ChunkData(BaseModel):
    """Data model for text chunks."""
    text: str
    length: int

# ============================================================================
# OPENAI CLIENT MANAGEMENT
# ============================================================================

@contextmanager
def openai_session():
    """Context manager to properly handle OpenAI API sessions."""
    client = None
    try:
        client = openai.OpenAI()
        yield client
    finally:
        if client is not None and hasattr(client, 'close'):
            client.close()

@asynccontextmanager
async def async_openai_session():
    """Context manager to properly handle asynchronous OpenAI API sessions."""
    client = None
    try:
        client = openai.OpenAI()
        yield client
    finally:
        if client is not None and hasattr(client, 'close'):
            await client.close()

def call_groq_api_with_prompt(prompt: str) -> str:
    """Call Groq API with the given prompt."""
    return call_groq_api(prompt)

async def async_call_groq_api_with_prompt(prompt: str) -> str:
    """Asynchronously call Groq API with the given prompt."""
    return await async_call_groq_api(prompt)

# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def generate_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """
    Generate a single embedding for a text string using OpenAI API.
    
    Args:
        text: Text to generate embedding for
        model: Name of the OpenAI embedding model to use
        
    Returns:
        List of embedding values
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            with openai_session() as client:
                response = client.embeddings.create(
                    input=text,
                    model=model
                )
                return response.data[0].embedding
        except Exception as e:
            retries += 1
            logger.warning(f"Embedding generation failed (attempt {retries}/{MAX_RETRIES}): {str(e)}")
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to generate embedding after {MAX_RETRIES} attempts")
                raise

# ============================================================================
# TEXT SPLITTING FUNCTIONS
# ============================================================================

def _is_content_meaningful(content: str) -> bool:
    """Check if content has meaningful text (not just images/empty)."""
    if not content or len(content.strip()) < 10:
        return False
    
    # Remove common markdown image patterns
    content_without_images = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    content_without_images = re.sub(r'<img[^>]*>', '', content_without_images)
    
    return bool(content_without_images.strip())

def split_text_markdown(
    markdown_text: str,
) -> List[ChunkData]:
    """
    Split markdown text with intelligent chunking and hierarchical context preservation.
    
    Args:
        markdown_text: The markdown content to split
        
    Returns:
        List of ChunkData objects with intelligent chunking applied
    """
    logger.debug(f"Intelligent markdown splitting with hierarchy preservation (length: {len(markdown_text)})")

    # Parse structure and extract chunks with context
    processed_chunks = process_markdown_text(markdown_text)
    
    final_chunks = [ChunkData(text=chunk.page_content, length=len(chunk.page_content)) for chunk in processed_chunks]

    logger.info(f"Generated {len(final_chunks)} chunked markdown sections")

    return final_chunks

# ============================================================================
# METADATA ENRICHMENT FUNCTIONS
# ============================================================================

def create_extraction_prompt(chunk: str, file_info: str, document_summary: str) -> str:
    """Create a prompt for metadata extraction."""

    final_prompt = f"""{CHUNK_METADATA_EXTRACTION_PROMPT}<file_info>
        {file_info}
        </file_info>

        <document_summary>
        {document_summary}
        </document_summary>

        <chunk>
        {chunk}
        </chunk>"""
    
    return final_prompt


def get_document_summary(document_text: str, file_info: str):
    """Get a summary of the entire document."""
    # max_content_tokens = 8000
    # document_text, num_tokens = truncate_content(document_text, max_content_tokens)

    max_num_characters = 30000
    if len(document_text) > max_num_characters:
        document_text = document_text[:max_num_characters]

    if len(document_text) < max_num_characters:
        truncation_message = ""
    else:
        truncation_message = TRUNCATION_MESSAGE.format(num_characters=max_num_characters)
    
    prompt = DOCUMENT_SUMMARIZATION_PROMPT.format(
        document_text=document_text,
        document_info=file_info,
        truncation_message=truncation_message
    )

    # print(f"Prompt: {prompt}")

    # replace with agent call, check the enum to use
    document_summary = call_groq_api_with_prompt(prompt)
    # document_summary = "This is a placeholder document summary for now"
    print(f"Document summary: {document_summary}")
    time.sleep(3)

    return document_summary

async def _enrich_single_chunk_file_async(
    chunk: str, 
    document_summary: str, 
    file_info: str
) -> Optional[EnrichedChunk]:
    """Asynchronously enrich a single chunk with metadata."""
    try:
        prompt = create_extraction_prompt(chunk, file_info, document_summary)
        raw_llm_response = await async_call_groq_api_with_prompt(prompt)
        
        section_summary_str = raw_llm_response.strip().removeprefix("```json").removesuffix("```").strip()

        if not section_summary_str:
            logger.warning(f"LLM response for chunk is empty.")
            return None

        metadata = ChunkMetadata(
            document_summary=document_summary,
            section_summary=section_summary_str
        )

        enriched_chunk_to_be_embedded = (
            "<metadata>\n"
            f"document_summary: {metadata.document_summary}\n"
            f"section_summary: {metadata.section_summary}\n"
            "</metadata>\n\n"
            "<chunk>\n"
            f"{chunk}\n"
            "</chunk>"
        )

        return EnrichedChunk(
            original_chunk=chunk,
            metadata=metadata,
            enriched_chunk_to_be_embedded=enriched_chunk_to_be_embedded
        )

    except Exception as e:
        logger.error(f"Error extracting metadata for chunk: {str(e)}")
        return None

async def _process_tasks_concurrently(tasks: List[Any]) -> List[EnrichedChunk]:
    """
    Run a list of async tasks with controlled concurrency.
    
    Args:
        tasks: List of awaitable tasks to run.
        document_title: The title of the document for logging purposes.
        
    Returns:
        A list of results from the completed tasks.
    """
    if not tasks:
        return []
    
    total_tasks = len(tasks)
    completed_tasks = 0
    
    # Control concurrency
    effective_concurrency = min(MAX_CONCURRENT_ENRICHMENT_TASKS, len(tasks))
    semaphore = asyncio.Semaphore(effective_concurrency)
    
    async def run_with_semaphore(task_to_run):
        nonlocal completed_tasks
        async with semaphore:
            result = await task_to_run
            completed_tasks += 1
            if completed_tasks % MAX_CONCURRENT_ENRICHMENT_TASKS == 0 or completed_tasks == total_tasks:
                logger.info(f"Enriched {completed_tasks}/{total_tasks} chunks...")
            return result
    
    tasks_with_semaphore_control = [run_with_semaphore(task) for task in tasks]
    
    logger.info(f"Processing {len(tasks)} chunks for with concurrency: {effective_concurrency}")
    
    # Gather results
    all_results = await asyncio.gather(*tasks_with_semaphore_control, return_exceptions=True)
    
    enriched_chunks_results = []
    for result in all_results:
        if isinstance(result, Exception):
            logger.error(f"A concurrent enrichment task failed: {result}")
        elif result:
            enriched_chunks_results.append(result)
    
    return enriched_chunks_results

async def enrich_chunks_with_metadata_and_file(
    chunks: List[str],
    file: FileDB,
    document_text: str
) -> List[EnrichedChunk]:
    """
    Process a list of chunks and extract metadata for each.
    
    Args:
        chunks: List of text chunks
        file: The FileDB object associated with the document
        document_text: The full document text
        
    Returns:
        List of dictionaries containing original text, metadata, and enriched text
    """

    # Build file info string, handling None values gracefully
    file_info = (
        f"File Source: {file.file_source or ''}\n"
        f"File Original Filename: {file.file_original_filename or ''}\n"
    )

    # Get document-level summary
    document_summary = get_document_summary(
        document_text=document_text,
        file_info=file_info
    )
    
    # Create tasks for concurrent processing
    tasks = [
        _enrich_single_chunk_file_async(
            chunk, 
            document_summary,
            file_info
        ) 
        for chunk in chunks
    ]
    
    if not tasks:
        return []
        
    return await _process_tasks_concurrently(tasks)
