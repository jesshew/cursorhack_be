import os
import time
import asyncio
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Rate limit settings
REQUESTS_PER_MINUTE = 1000
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

def call_groq_api(prompt: str) -> str:
    """Call the Groq API with the given prompt."""
    time.sleep(DELAY_BETWEEN_REQUESTS)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating output: {str(e)}"

async def async_call_groq_api(prompt: str) -> str:
    """Asynchronously call the Groq API with the given prompt."""
    await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    try:
        response = await async_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating output: {str(e)}"
