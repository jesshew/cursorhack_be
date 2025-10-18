from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# Simple configuration - easy to modify
MIN_TOKENS = 128
MAX_TOKENS = 256
CHARS_PER_TOKEN = 4
OVERLAP_PERCENT = 0.15
OVERFLOW_MARGIN = 1.1
ABSOLUTE_MIN_CHUNK_SIZE = 300

# Calculated sizes
MIN_CHUNK_SIZE = MIN_TOKENS * CHARS_PER_TOKEN
MAX_CHUNK_SIZE = MAX_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP = int(MIN_CHUNK_SIZE * OVERLAP_PERCENT)

def merge_small_chunks(docs: List[Document]) -> List[Document]:
    """
    Merges small chunks in a single pass, ensuring no chunk is too small unless unavoidable.
    """
    if not docs:
        return []

    # Helper to create a merged doc from a list of docs
    def create_merged_doc(chunk_list: List[Document]) -> Document:
        content = "\n\n---\n\n".join(doc.page_content for doc in chunk_list)
        metadata = chunk_list[-1].metadata if chunk_list else {}
        return Document(page_content=content, metadata=metadata)

    # Helper to calculate size of a list of docs
    def get_content_size(chunk_list: List[Document]) -> int:
        if not chunk_list: return 0
        content_size = sum(len(doc.page_content) for doc in chunk_list)
        separator_size = (len(chunk_list) - 1) * 7
        return content_size + separator_size

    final_chunks: List[Document] = []
    buffer: List[Document] = []

    def flush_buffer():
        """A smart flush that merges small buffers with the previous chunk if possible."""
        nonlocal buffer, final_chunks
        if not buffer:
            return

        # If the buffer is the very first thing to be flushed and it's too small,
        # just leave it. It will be handled by the next flush.
        if not final_chunks and get_content_size(buffer) < ABSOLUTE_MIN_CHUNK_SIZE:
            return

        # If buffer is too small, try merging it with the previous final chunk
        if get_content_size(buffer) < ABSOLUTE_MIN_CHUNK_SIZE and final_chunks:
            last_chunk = final_chunks.pop()
            combined_buffer = [last_chunk] + buffer

            # Prioritize merging small chunks, even if it creates a slightly oversized chunk.
            # Allow a 10% overflow margin.
            if get_content_size(combined_buffer) <= MAX_CHUNK_SIZE * OVERFLOW_MARGIN:
                final_chunks.append(create_merged_doc(combined_buffer))
            else:
                # Merging would be too big, so add them back separately
                final_chunks.append(last_chunk)
                final_chunks.append(create_merged_doc(buffer))
        else:
            # Buffer is big enough or it's the first chunk, so just add it
            final_chunks.append(create_merged_doc(buffer))

        buffer = [] # Reset the buffer

    for doc in docs:
        if not doc.page_content.strip():
            continue

        # Oversized chunks are flushed and added separately
        if len(doc.page_content) > MAX_CHUNK_SIZE:
            flush_buffer() # Flush any existing buffer before adding the large doc
            final_chunks.append(doc)
            continue

        # If adding the doc would overflow the buffer, flush the buffer first
        if buffer and get_content_size(buffer + [doc]) > MAX_CHUNK_SIZE:
            flush_buffer()

        buffer.append(doc)

    # Flush any remaining buffer at the end
    flush_buffer()

    return final_chunks


def process_markdown_text(markdown_text: str) -> List[Document]:
    """
    Process a markdown file into chunks.

    Steps:
    1. Load the file
    2. Split by headers (# and ##)
    3. Split large sections into smaller pieces
    4. Merge tiny pieces together
    """

    # Step 2: Split by headers to keep related content together
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
        strip_headers=False
    )
    header_chunks = header_splitter.split_text(markdown_text)

    # Step 3: Split chunks that are too large
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    size_chunks = size_splitter.split_documents(header_chunks)

    # Step 4: Merge chunks that are too small
    final_chunks = merge_small_chunks(size_chunks)

    return final_chunks
