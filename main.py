import logging
import os
import sys
import asyncio
sys.path.append(os.getcwd())
from src.core.config import settings
from src.providers.news_client import NewsAPIClient, CachedNewsProvider, AutoRetryProvider
from src.providers.analysis_client import GeminiAnalyzer
from src.core.pipeline import StockAnalysisPipeline
from src.core.logger import setup_logging
from src.utils.stats_collector import StatsCollector
from typing import Dict, Any

logger = logging.getLogger("MAIN")

def print_final_recommendation(recommendation_data: Dict[str, Any], ticker: str):
    print("\n" + "💰" * 15 + " FINAL INVESTMENT REPORT " + "💰" * 15)
    print(f"📈 Ticker: {ticker}")
    print(f"🎯 Sentiment: {recommendation_data.get('final_sentiment', 'N/A')}")
    print(f"⭐ Recommendation: {recommendation_data.get('recommendation', 'N/A')}")
    print("-" * 65)
    print(f"📝 Summary: {recommendation_data.get('overall_summary', 'N/A')}")
    print("\n⚠️ Risks:")
    for risk in recommendation_data.get('major_risks', []):
        print(f"  - {risk}")
    print("\n" + "💰" * 53 + "\n")


async def main():
    setup_logging()

    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    logger.info(f"🚀 Starting analysis for ticker: {ticker}")

    news_key = settings.NEWS_API_KEY.get_secret_value()
    gemini_key = settings.GEMINI_API_KEY.get_secret_value()

    fetch_count = settings.ARTICLES_TO_FETCH
    inference_count = settings.ARTICLES_TO_INFERENCE
    ttl_seconds = settings.CACHE_TTL_SECONDS

    stats_collector = StatsCollector()
    stats_collector.set_initial_context(
        ticker=ticker,
        articles_to_fetch=fetch_count,
        articles_to_inference=inference_count
    )

    base_provider = NewsAPIClient(api_key=news_key)
    retry_provider = AutoRetryProvider(inner_provider=base_provider, stats=stats_collector, max_retries=3)
    final_news_provider = CachedNewsProvider(retry_provider, cache_dir="data/cache", ttl_seconds=ttl_seconds)
    my_gemini_client = GeminiAnalyzer(api_key=gemini_key)

    pipeline = StockAnalysisPipeline(
        news_provider=final_news_provider,
        analyzer=my_gemini_client,
        stats=stats_collector
    )

    try:
        final_report = await pipeline.run(ticker, fetch_count, inference_count)
        print_final_recommendation(final_report, ticker)
        logger.info("✅ Analysis completed successfully.")
        stats_collector.finalize(status='OK')
    except Exception as e:
        print(f"🛑 Error: {e}")
        logger.critical(f"🛑 Critical failure: {e}", exc_info=True)
        stats_collector.update_error(stage="PIPELINE_ERROR", message=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())