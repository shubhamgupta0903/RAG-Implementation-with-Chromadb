from fastapi import FastAPI
import sys
import os
sys.path.append(os.path.dirname(__file__))
from api.endpoints import router as api_router
from services.document_processor import init_db
import os


app = FastAPI(
    title="PanScience RAG API",
    description="An API for document ingestion and question-answering using RAG.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the PanScience RAG API Internship task. Visit /docs for API documentation."}