#!/usr/bin/env python3
import concurrent
import sys
from typing import Dict, Any, List
import time
import config
from news_client import fetch_articles, NUM_OF_ARTICLES
from analysis_client import search_relevant_articles, embed_articles, analyze_single_article

def parse_ticker_from_args() -> str:
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1].upper()

def analyze_articles_concurrently(
    ticker_symbol: str,
    relevant_articles_text: List[str],
    gemini_api_key: str,
) -> Dict[str, Any]:
    """
    Run sentiment analysis for multiple articles concurrently and return
    the aggregated analysis data structure expected by print_analysis_report.
    """
    articles_for_threading = [{"content": text} for text in relevant_articles_text]
    results = []

    max_workers = len(articles_for_threading) or 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(
                analyze_single_article,
                ticker_symbol,
                article,
                gemini_api_key,
            ): article
            for article in articles_for_threading
        }

        for future in concurrent.futures.as_completed(future_to_article):
            try:
                result = future.result()
                if "news_items" in result:
                    results.extend(result["news_items"])
                elif "error" in result:
                    print(f"⚠️ Thread error: {result['error']}", file=sys.stderr)
            except Exception as exc:
                print(f"A thread generated an exception: {exc}", file=sys.stderr)

    return {
        "ticker": ticker_symbol,
        "analysis_date": time.strftime("%Y-%m-%d"),
        "news_items": results,
    }


def print_analysis_report(analysis_data: Dict[str, Any], ticker: str, num_articles: int):
    """
    Calculates statistics from the analysis data and prints a formatted report.
    """
    news_items = analysis_data.get('news_items', [])
    if not news_items:
        print("🛑 The analysis did not return any news items for processing.", file=sys.stderr)
        sys.exit(1)

    # Calculate the average score (The Fear/Greed Index)
    total_score = sum(item.get('sentiment_score', 0) for item in news_items)
    average_score = total_score / len(news_items)

    # Find the most positive and most negative news
    best_news = max(news_items, key=lambda x: x.get('sentiment_score', 0))
    worst_news = min(news_items, key=lambda x: x.get('sentiment_score', 0))

    # Determine overall sentiment category
    if average_score >= 7.0:
        overall_sentiment = "Strong Bullish (High Greed)"
    elif 6.0 <= average_score < 7.0:
        overall_sentiment = "Neutral/Mixed (Uncertainty)"
    else:
        overall_sentiment = "Bearish (High Fear)"

    print("\n" + "=" * 90)
    print(f"📊 FINAL SENTIMENT INDEX FOR {ticker} (Based on {num_articles} Articles)")
    print("=" * 90)

    print(f"Overall Average Score: {average_score:.2f} / 10.00")
    print(f"Overall Sentiment:     {overall_sentiment}")
    print("\n")

    # Highlight Key Findings
    print("⭐ Most Positive News:")
    print(f"  Score: {best_news['sentiment_score']}/10. | Category: {best_news['sentiment_category']}")
    print(f"  Headline: {best_news['headline']}")
    print(f"  Reason: {best_news['impact_reason']}")

    print("\n🔻 Most Negative News:")
    print(f"  Score: {worst_news['sentiment_score']}/10. | Category: {worst_news['sentiment_category']}")
    print(f"  Headline: {worst_news['headline']}")
    print(f"  Reason: {worst_news['impact_reason']}")

    print("\n" + "=" * 90)


def main():
    """Main function to run the stock news finder and sentiment analyzer."""
    ticker_symbol = parse_ticker_from_args()

    news_api_key, gemini_api_key = config.load_and_validate_keys()

    # --- Step 1: Gather Data ---
    start_time_fetch = time.time()
    articles = fetch_articles(ticker_symbol, news_api_key, NUM_OF_ARTICLES)
    end_time_fetch = time.time()
    print(f"fetch_articles: {end_time_fetch - start_time_fetch:.2f} seconds")

    # --- Step 1.5: Process Data ---
    article_texts, index = embed_articles(articles)
    relevant_articles_text = search_relevant_articles(ticker_symbol, article_texts, index)

    # --- Step 2: Analyze Sentiment (concurrently) ---
    analysis_data = analyze_articles_concurrently(
        ticker_symbol,
        relevant_articles_text,
        gemini_api_key,
    )

    # --- Step 3: Print Report ---
    print(f"\n Generating report...")
    print_analysis_report(analysis_data, ticker_symbol, len(articles))

if __name__ == "__main__":
    main()