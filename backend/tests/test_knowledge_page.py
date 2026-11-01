from fastapi.testclient import TestClient

from app.main import create_app


def test_knowledge_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/knowledge")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "knowledge.js" in response.text


def test_knowledge_page_assets_are_mounted() -> None:
    client = TestClient(create_app())

    for path in ("/static/js/knowledge.js", "/static/css/knowledge.css"):
        response = client.get(path)
        assert response.status_code == 200, path
