import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_document():
    file_content = b"Test PDF Content"
    response = client.post(
        "/api/upload",
        files={"files": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
    )
    assert response.status_code == 200
    assert "files received" in response.json()["message"]
