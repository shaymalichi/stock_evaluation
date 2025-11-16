#!/usr/bin/env python3

import sys
from typing import Dict, Any
import time
import config
from news_client import fetch_articles
from analysis_client import search_relevant_articles, embed_articles, synthesize_report, analyze_articles_concurrently
from stats_collector import StatsCollector


def parse_ticker_from_args() -> str:
    """Parses the ticker symbol from command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1].upper()


def print_final_recommendation(recommendation_data: Dict[str, Any], ticker: str):
    """
    Prints the final synthesized recommendation and summary from the AI.
    """
    print("\n" + "💰" * 15 + " FINAL INVESTMENT REPORT " + "💰" * 15)
    print(f"📈 Ticker: {ticker}")
    print(f"🎯 **Consolidated Sentiment:** {recommendation_data.get('final_sentiment', 'N/A')}")
    print(f"⭐ **Actionable Recommendation:** {recommendation_data.get('recommendation', 'N/A')}")
    print("-" * 65)

    print("\n📝 **Overall Summary:**")
    print(f"  {recommendation_data.get('overall_summary', 'No summary provided.')}")

    print("\n⚠️ **Major Risks/Opportunities:**")
    risks = recommendation_data.get('major_risks', [])
    if risks:
        for i, risk in enumerate(risks, 1):
            print(f"  {i}. {risk}")
    else:
        print("  N/A")

    print("\n" + "💰" * 53 + "\n")


def main():
    """Main function to run the stock news finder and sentiment analyzer."""
    # 1. Initialize Ticker and API Keys
    ticker_symbol = parse_ticker_from_args()
    news_api_key, gemini_api_key = config.load_and_validate_keys()

    # 2. Initialize Stats Collector
    stats_collector = StatsCollector()

    articles_to_fetch = int(config.ARTICLES_TO_FETCH)
    articles_to_inference = int(config.ARTICLES_TO_INFERENCE)

    stats_collector.set_initial_context(
        ticker=ticker_symbol,
        articles_to_fetch=articles_to_fetch,
        articles_to_inference=articles_to_inference
    )

    try:
        # --- Step 1: Gather Data (Fetch & Filter) ---
        start_time_fetch = time.time()
        articles = fetch_articles(ticker_symbol, news_api_key, articles_to_fetch)
        end_time_fetch = time.time()
        fetch_duration = end_time_fetch - start_time_fetch

        stats_collector.update('articles_requested', articles_to_fetch)
        stats_collector.update('articles_returned', len(articles))
        stats_collector.update('articles_after_filter', len(articles))
        stats_collector.update('fetch_duration_sec', round(fetch_duration, 3))
        stats_collector.update('news_fetch_status', 'OK' if articles else 'NO_ARTICLES')

        print(f"✅ [STEP 1] Completed in {fetch_duration:.2f} seconds.")

        if not articles:
            raise Exception("No articles returned from the news API.")

        # --- Step 1.5: Process Data (Embedding & Retrieval) ---
        print("🔍 [STEP 1.5] Embedding articles and searching for relevant context (RAG)...")
        start_time_rag = time.time()
        article_texts, index = embed_articles(articles)
        relevant_articles_text = search_relevant_articles(
            ticker_symbol,
            article_texts,
            index,
            articles_to_inference
        )
        end_time_rag = time.time()

        stats_collector.update('relevant_articles_found', len(relevant_articles_text))

        if not relevant_articles_text:
            raise Exception("RAG process failed to find relevant articles for analysis.")

        # --- Step 2: Analysis (Concurrent Threading) ---
        print(f"🧠 [STEP 2] Analyzing {len(relevant_articles_text)} relevant articles concurrently...")
        start_time_analysis = time.time()
        analysis_data = analyze_articles_concurrently(
            ticker_symbol,
            relevant_articles_text,
            gemini_api_key,
        )
        end_time_analysis = time.time()
        analysis_duration = end_time_analysis - start_time_analysis

        stats_collector.calculate_analysis_metrics(analysis_data, analysis_duration)

        if not analysis_data.get('news_items'):
            raise Exception("Concurrent analysis returned no successfully analyzed articles.")

        # --- Step 3: Synthesis / Inference ---
        print(f"✨ [STEP 3] Synthesizing final report and recommendation...")
        start_time_synthesis = time.time()
        final_recommendation_data = synthesize_report(
            ticker_symbol,
            analysis_data.get('news_items', []),
            gemini_api_key,
        )
        end_time_synthesis = time.time()
        synthesis_duration = end_time_synthesis - start_time_synthesis
        stats_collector.calculate_synthesis_metrics(final_recommendation_data, synthesis_duration)

        # --- Step 4: Print Report ---
        print_final_recommendation(final_recommendation_data, ticker_symbol)

        # --- Step 5: Finalize and Save Stats ---
        stats_collector.finalize(status='OK')

    except Exception as e:
        error_stage = "UNKNOWN"
        if 'start_time_fetch' not in locals():
            error_stage = "STARTUP"
        elif 'articles' not in locals():
            error_stage = "FETCH"
        elif 'relevant_articles_text' not in locals():
            error_stage = "RAG"
        elif 'analysis_data' not in locals() or 'news_items' not in analysis_data:
            error_stage = "ANALYSIS"
        else:
            error_stage = "SYNTHESIS"

        print(f"🛑 FATAL ERROR in run at stage {error_stage}: {e}", file=sys.stderr)
        stats_collector.update_error(stage=error_stage, message=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()