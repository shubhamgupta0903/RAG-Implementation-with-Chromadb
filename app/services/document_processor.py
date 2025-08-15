import os
import pypdf
import sqlite3
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
import chromadb


DB_PATH = "app/db/metadata.db"  # Updated path for Docker

def init_db():
    # Delete the old database file if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Deleted old database: {DB_PATH}")

    # Create a fresh database and table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            processing_status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"Initialized fresh database: {DB_PATH}")


# Minimal Embeddings Wrapper 
class MinimalHFEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# CHROMA & EMBEDDING SETUP 
COLLECTION_NAME = "rag_collection"

# In-process Chroma client
chroma_client = chromadb.Client()


existing_collections = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME in existing_collections:
    chroma_client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted old Chroma collection: {COLLECTION_NAME}")

# Create new collection
chroma_client.create_collection(name=COLLECTION_NAME)
embedding_function = MinimalHFEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# DOCUMENT PROCESSING 
def process_document(file_path: str, filename: str):
    """Processes a single PDF document, stores embeddings in ChromaDB, metadata in SQLite, and deletes the PDF after processing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    #  Add initial metadata record
    upload_date = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO documents (filename, upload_date, processing_status) VALUES (?,?,?)",
        (filename, upload_date, "processing")
    )
    conn.commit()
    doc_id = cursor.lastrowid

    try:
        #  Extract text from PDF
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        #  Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        #  Create metadata for each chunk
        metadatas = [{"source": filename, "page": i} for i in range(len(chunks))]

        vector_store.add_texts(texts=chunks, metadatas=metadatas)

        cursor.execute(
            "UPDATE documents SET processing_status =? WHERE id =?",
            ("completed", doc_id)
        )
        conn.commit()

    except Exception as e:
        # If processing fails, update status to 'failed'
        cursor.execute(
            "UPDATE documents SET processing_status =? WHERE id =?",
            ("failed", doc_id)
        )
        conn.commit()
        print(f"Failed to process {filename}: {e}")

    finally:
        conn.close()
        #  Delete the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted uploaded file: {file_path}")


init_db()
