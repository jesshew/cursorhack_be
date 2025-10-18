from contextlib import contextmanager
import openai

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
