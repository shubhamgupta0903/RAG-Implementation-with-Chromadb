import os
import shutil
import sqlite3
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from .models import QueryRequest, QueryResponse, DocumentMetadata
from services.document_processor import process_document, DB_PATH, vector_store
from core.rag_pipeline import get_rag_chain

router = APIRouter()
UPLOAD_DIR = "temp_uploads"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add processing to a background task
        background_tasks.add_task(process_document, file_path, file.filename)

    return {"message": f"{len(files)} files received and scheduled for processing."}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        rag_chain, retriever = get_rag_chain(return_retriever=True)
        answer = rag_chain.invoke(request.query)

       

        return QueryResponse(
            answer=answer
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata", response_model=List[DocumentMetadata])
async def get_metadata():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, upload_date, processing_status 
            FROM documents
        """)
        rows = cursor.fetchall()
        conn.close()

        # Convert each sqlite3.Row to a dict
        metadata_list = [dict(row) for row in rows]
        return metadata_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/delete/{filename}")
async def delete_document(filename: str):
    try:
        # Remove from Chroma
        vector_store.delete(where={"source": filename})
        
        # Remove from SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE filename=?", (filename,))
        conn.commit()
        conn.close()

        return {"message": f"{filename} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
