import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.server import app

MOCK_RESPONSE = {
    "overall_summary": "Test API Summary",
    "final_sentiment": "Bullish",
    "recommendation": "BUY",
    "major_risks": ["Risk A", "Risk B"]
}


@patch('src.api.server.setup_logging')
@patch('src.api.server.CachedNewsProvider')
@patch('src.api.server.StatsCollector')
@patch('src.api.server.GeminiAnalyzer')
@patch('src.api.server.NewsAPIClient')
@patch('src.api.server.StockAnalysisPipeline')
def test_analyze_endpoint(
        mock_pipeline_class,
        mock_news_class,
        mock_gemini_class,
        mock_stats_class,
        mock_cached_provider_class,
        mock_setup_logging
):
    """
    Tests the /analyze endpoint with normal flow.
    """
    mock_instance = mock_pipeline_class.return_value
    mock_instance.run.return_value = MOCK_RESPONSE

    with TestClient(app) as client:
        response = client.post("/analyze", json={"ticker": "GOOGL"})

        assert response.status_code == 200
        assert response.json() == MOCK_RESPONSE

        mock_instance.run.assert_called_once()
        args, _ = mock_instance.run.call_args
        assert args[0] == "GOOGL"

        mock_setup_logging.assert_called_once()


@patch('src.api.server.setup_logging')
@patch('src.api.server.CachedNewsProvider')
@patch('src.api.server.StatsCollector')
@patch('src.api.server.GeminiAnalyzer')
@patch('src.api.server.NewsAPIClient')
@patch('src.api.server.StockAnalysisPipeline')
def test_analyze_endpoint_error_handling(
        mock_pipeline_class,
        mock_news,
        mock_gemini,
        mock_stats,
        mock_cache,
        mock_logging
):
    """
    Tests that server returns 500 error with details when pipeline fails.
    """
    mock_instance = mock_pipeline_class.return_value
    mock_instance.run.side_effect = Exception("Database Connection Failed")

    with TestClient(app) as client:
        response = client.post("/analyze", json={"ticker": "FAIL"})

        assert response.status_code == 500
        assert "Database Connection Failed" in response.json()["detail"]


def test_health_check():
    """Simple health check test"""

    with patch('src.api.server.setup_logging'), \
            patch('src.api.server.CachedNewsProvider'), \
            patch('src.api.server.StatsCollector'), \
            patch('src.api.server.GeminiAnalyzer'):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
