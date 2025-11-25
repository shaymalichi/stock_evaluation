import sys
from typing import Dict, Any
from src.core import config
from src.providers.news_client import NewsAPIClient, CachedNewsProvider, AutoRetryProvider
from src.providers.analysis_client import GeminiAnalyzer
from src.core.pipeline import StockAnalysisPipeline
from src.utils.stats_collector import StatsCollector


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    news_key, gemini_key = config.load_and_validate_keys()

    fetch_count = int(config.ARTICLES_TO_FETCH)
    inference_count = int(config.ARTICLES_TO_INFERENCE)

    base_provider = NewsAPIClient(api_key=news_key)
    retry_provider = AutoRetryProvider(inner_provider=base_provider, max_retries=3)
    final_news_provider = CachedNewsProvider(retry_provider, cache_dir=config.CACHE_DIR, ttl_seconds=config.CACHE_TTL_SECONDS)
    my_gemini_client = GeminiAnalyzer(api_key=gemini_key)
    stats_collector = StatsCollector()

    stats_collector.set_initial_context(
        ticker=ticker,
        articles_to_fetch=fetch_count,
        articles_to_inference=inference_count
    )

    pipeline = StockAnalysisPipeline(
        news_provider=final_news_provider,
        analyzer=my_gemini_client,
        stats=stats_collector
    )

    try:
        final_report = pipeline.run(ticker, fetch_count, inference_count)
        print_final_recommendation(final_report, ticker)
        stats_collector.finalize(status='OK')
    except Exception as e:
        print(f"🛑 Error: {e}")
        stats_collector.update_error(stage="PIPELINE_ERROR", message=str(e))
        sys.exit(1)