import pytest
from unittest.mock import MagicMock
from src.core.pipeline import StockAnalysisPipeline
from src.core.interfaces import INewsProvider, IStockAnalyzer
from src.utils.stats_collector import StatsCollector


@pytest.fixture
def mock_news_provider():
    """Creates a fake news provider"""
    provider = MagicMock(spec=INewsProvider)
    # Configure it to return a list with one article when asked for articles
    provider.fetch_articles.return_value = [
        {"title": "Fake News", "content": "Something happened"}
    ]
    return provider


@pytest.fixture
def mock_analyzer():
    """Creates a fake analyzer (Gemini)"""
    analyzer = MagicMock(spec=IStockAnalyzer)

    # Step 1: Filtering (RAG)
    analyzer.filter_relevant.return_value = ["Fake News Content"]

    # Step 2: Analysis of each article
    analyzer.analyze.return_value = {
        "news_items": [
            {"sentiment_score": 9, "sentiment_category": "POSITIVE", "headline": "Fake", "impact_reason": "Good"}
        ]
    }

    # Step 3: Synthesis (final report)
    analyzer.synthesize.return_value = {
        "overall_summary": "Looks good",
        "final_sentiment": "Bullish",
        "recommendation": "BUY",
        "major_risks": []
    }
    return analyzer


def test_pipeline_flow(mock_news_provider, mock_analyzer, tmp_path):
    """
    Tests that the pipeline runs all steps in the correct order
    without actually calling external APIs.
    """
    temp_csv = tmp_path / "test_pipeline.csv"
    stats = StatsCollector(filename=str(temp_csv))
    stats.set_initial_context("AAPL", 10, 2)

    # Create a pipeline with mock objects
    pipeline = StockAnalysisPipeline(
        news_provider=mock_news_provider,
        analyzer=mock_analyzer,
        stats=stats
    )

    # Run the pipeline
    result = pipeline.run("AAPL", fetch_count=10, inference_count=2)

    # --- Assertions ---

    # 1. Verify the final result came from mock Analyzer
    assert result['recommendation'] == "BUY"
    assert result['final_sentiment'] == "Bullish"

    # 2. Verify the pipeline called fetch_articles exactly once
    mock_news_provider.fetch_articles.assert_called_once_with("AAPL", 10)

    # 3. Verify pipeline performed filtering (RAG)
    mock_analyzer.filter_relevant.assert_called_once()

    # 4. Verify pipeline sent for analysis
    mock_analyzer.analyze.assert_called_once()

    # 5. Check that statistics were updated [cite: 94]
    assert stats.stats['articles_returned'] == 1
    assert stats.stats['run_status'] == "IN_PROGRESS"  # Changes to OK only in main.py
