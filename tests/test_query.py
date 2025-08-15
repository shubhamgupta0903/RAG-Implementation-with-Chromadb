import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_documents():
    response = client.post("/api/query", json={"query": "What is the document about?"})
    assert response.status_code == 200
    assert "answer" in response.json()
