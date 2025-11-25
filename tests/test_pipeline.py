import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.pipeline import StockAnalysisPipeline
from src.core.interfaces import INewsProvider, IStockAnalyzer
from src.utils.stats_collector import StatsCollector


@pytest.fixture
def mock_news_provider():
    """Creates a fake news provider"""
    provider = MagicMock(spec=INewsProvider)

    provider.fetch_articles.return_value = [
        {"title": "Fake News", "content": "Something happened"}
    ]
    return provider


@pytest.fixture
def mock_analyzer():
    """Creates a fake analyzer (Gemini)"""
    analyzer = MagicMock(spec=IStockAnalyzer)

    analyzer.filter_relevant = AsyncMock(return_value=["Fake News Content"])

    analyzer.analyze.return_value = {
        "news_items": [
            {"sentiment_score": 9, "sentiment_category": "POSITIVE", "headline": "Fake", "impact_reason": "Good"}
        ]
    }

    analyzer.synthesize.return_value = {
        "overall_summary": "Looks good",
        "final_sentiment": "Bullish",
        "recommendation": "BUY",
        "major_risks": []
    }
    return analyzer

@pytest.mark.asyncio
async def test_pipeline_happy_flow(mock_news_provider, mock_analyzer, tmp_path):
    """
    Tests that the pipeline runs all steps in the correct order
    without actually calling external APIs.
    """
    temp_csv = tmp_path / "test_pipeline.csv"

    stats = StatsCollector(filename=str(temp_csv))

    stats.set_initial_context("AAPL", 10, 2)

    pipeline = StockAnalysisPipeline(
        news_provider=mock_news_provider,
        analyzer=mock_analyzer,
        stats=stats
    )

    result = await pipeline.run("AAPL", fetch_count=10, inference_count=2)

    assert result['recommendation'] == "BUY"
    assert result['final_sentiment'] == "Bullish"

    mock_news_provider.fetch_articles.assert_called_once_with("AAPL", 10)
    mock_analyzer.filter_relevant.assert_called_once()
    mock_analyzer.analyze.assert_called_once()

    assert stats.stats['articles_returned'] == 1
    assert stats.stats['run_status'] == "IN_PROGRESS"

@pytest.mark.asyncio
async def test_pipeline_fails_on_no_rag_results(mock_news_provider, mock_analyzer, tmp_path):
    """
    Tests the scenario where the RAG filtering component fails
    to return any relevant articles, verifying the pipeline's error handling.
    """
    mock_analyzer.filter_relevant = AsyncMock(return_value=[])

    temp_csv = tmp_path / "test_rag_fail.csv"
    stats = StatsCollector(filename=str(temp_csv))

    pipeline = StockAnalysisPipeline(
        news_provider=mock_news_provider,
        analyzer=mock_analyzer,
        stats=stats
    )

    with pytest.raises(Exception, match="RAG returned no relevant articles."):
        await pipeline.run("AAPL", fetch_count=10, inference_count=2)