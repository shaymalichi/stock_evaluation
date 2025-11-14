#!/usr/bin/env python3

import sys
from typing import Dict, Any

import config
from news_client import fetch_articles
from analysis_client import analyze_sentiment


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
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()

    # --- Step 0: Load Configuration ---
    news_api_key, gemini_api_key = config.load_and_validate_keys()

    # --- Step 1: Gather Data ---
    print(f"🚀 [STEP 1] Searching for recent news articles for {ticker}...")
    articles = fetch_articles(ticker, news_api_key, num_results=15)
    if not articles:
        print(f'no articles found for {ticker}')
        sys.exit(1)

    # --- Step 2: Analyze Sentiment ---
    print(f"🧠 [STEP 2] Found {len(articles)} articles. Sending for structured sentiment analysis...")
    analysis_data = analyze_sentiment(ticker, articles, gemini_api_key)

    if "error" in analysis_data:
        print(f"🛑 Analysis Error: {analysis_data['error']}", file=sys.stderr)
        print(f"Raw response: {analysis_data.get('raw_response', 'N/A')}", file=sys.stderr)
        sys.exit(1)

    print_analysis_report(analysis_data, ticker, len(articles))


if __name__ == "__main__":
    main()