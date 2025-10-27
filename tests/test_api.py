import pytest
import requests
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_recommend_endpoint_valid_user():
    """Test the recommendation endpoint with a valid user ID"""
    response = client.get("/api/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    # Check that recommendations have the expected structure
    if data["recommendations"]:
        rec = data["recommendations"][0]
        assert "title" in rec
        assert "genres" in rec
        assert "predicted_rating" in rec

def test_recommend_endpoint_invalid_user():
    """Test the recommendation endpoint with an invalid user ID"""
    response = client.get("/api/recommend/9999")
    # This might return empty recommendations or an error depending on implementation
    assert response.status_code in [200, 404]

def test_recommend_endpoint_non_numeric_user():
    """Test the recommendation endpoint with a non-numeric user ID"""
    response = client.get("/api/recommend/abc")
    assert response.status_code == 422  # FastAPI validation error

def test_cors_headers():
    """Test that CORS headers are properly set"""
    response = client.options("/api/health")
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

# Integration tests (require the actual model to be loaded)
@pytest.mark.integration
def test_recommendations_have_expected_structure():
    """Test that recommendations have all required fields"""
    response = client.get("/api/recommend/1")
    assert response.status_code == 200
    data = response.json()

    for rec in data["recommendations"][:5]:  # Test first 5 recommendations
        assert isinstance(rec["title"], str)
        assert len(rec["title"]) > 0
        assert isinstance(rec["genres"], list)
        assert len(rec["genres"]) > 0
        assert isinstance(rec["predicted_rating"], (int, float))
        assert 1.0 <= rec["predicted_rating"] <= 5.0

@pytest.mark.integration
def test_recommendations_are_diverse():
    """Test that recommendations include different genres"""
    response = client.get("/api/recommend/1")
    assert response.status_code == 200
    data = response.json()

    genres = set()
    for rec in data["recommendations"]:
        genres.update(rec["genres"])

    # Should have at least a few different genres
    assert len(genres) >= 3