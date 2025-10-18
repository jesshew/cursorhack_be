from fastapi import FastAPI
from dotenv import load_dotenv
from app.api import health, storage, pdf, rag

load_dotenv()

app = FastAPI()

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(storage.router, prefix="/storage", tags=["storage"])
app.include_router(pdf.router, prefix="/pdf", tags=["pdf"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])

