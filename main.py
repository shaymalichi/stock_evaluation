#!/usr/bin/env python3
import concurrent
import sys
from typing import Dict, Any, List
import time
import config
from news_client import fetch_articles, NUM_OF_ARTICLES
from analysis_client import search_relevant_articles, embed_articles, analyze_single_article, synthesize_report

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

    analysis_data = analyze_articles_concurrently(
        ticker_symbol,
        relevant_articles_text,
        gemini_api_key,
    )

    # --- Step 2: Inference ---
    final_recommendation_data = synthesize_report(
        ticker_symbol,
        analysis_data['news_items'],
        gemini_api_key
    )

    # --- Step 3: Print Report ---
    print(f"\n Generating report...")
    print_final_recommendation(final_recommendation_data, ticker_symbol)

if __name__ == "__main__":
    main()