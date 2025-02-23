from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_process_text_request():
    response = client.post(
        "/process",
        json={
            "modality": "text",
            "content": "What is machine learning?",
            "confidence_threshold": 0.5
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_process_image_request():
    with open("tests/fixtures/test_image.jpg", "rb") as f:
        image_bytes = f.read()
    
    response = client.post(
        "/process",
        json={
            "modality": "image",
            "content": image_bytes,
            "confidence_threshold": 0.5
        }
    )
    assert response.status_code == 200
    assert "response" in response.json() 