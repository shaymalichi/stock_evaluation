#!/usr/bin/env python3

import os
import sys
import json
from typing import List, Dict, Any
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def fetch_articles(ticker: str, news_api_key: str, num_results: int = 15) -> List[Dict[str, str]]:
    """
    Search for news articles about a stock ticker using NewsAPI and return filtered article data.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        news_api_key: NewsAPI key
        num_results: Number of articles to retrieve (default: 15)

    Returns:
        List of dictionaries containing filtered article data (author, content, title).
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{ticker} stock",
        'apiKey': news_api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': num_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] != 'ok':
            print(f"Error: {data.get('message', 'Unknown error')}")
            return []

        articles = data.get('articles', [])
        filtered_articles = []
        for article in articles:
            filtered_article = {
                'author': article.get('author', ''),
                'title': article.get('title', ''),
                'content': article.get('content', '')
            }
            filtered_articles.append(filtered_article)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return []

    return filtered_articles[:num_results]


def analyze_sentiment(ticker: str, articles: List[Dict[str, str]], api_key: str) -> Dict[str, Any]:
    """
    Analyzes news articles for sentiment using Gemini API and returns a structured JSON.

    Args:
        ticker: Stock ticker symbol
        articles: List of article dictionaries
        api_key: Google Gemini API key

    Returns:
        Dictionary containing the full structured analysis or an error message.
    """
    if not articles:
        return {"error": "No articles found for analysis."}

    # Prepare the content for analysis
    articles_text = "\n\n---\n\n".join([
        f"AUTHOR: {article['author']}\n"
        f"TITLE: {article['title']}\n"
        f"CONTENT: {article['content']}"
        for article in articles
    ])

    # Configure Gemini API
    genai.configure(api_key=api_key)
    # Using Gemini-2.5-Flash for faster, cost-effective structured analysis
    model = genai.GenerativeModel('gemini-2.5-flash')

    # The sophisticated prompt forcing a structured output:
    prompt = f"""
    You are a professional Senior Capital Market Analyst. Your mission is to analyze the following news snippets concerning the company {ticker} and provide a detailed, structured sentiment analysis. You must analyze each article individually.

    **STRICT INSTRUCTIONS:**
    1. **ONLY RETURN VALID JSON.** Do not include any other text, greetings, or explanations outside the JSON structure.
    2. **Sentiment Score:** A number from 1 (Extremely Negative, High Risk) to 10 (Extremely Positive, High Opportunity).
    3. **Impact Reason:** A short explanation (max 15 words) on the potential impact on the stock price.

    **DATA FOR ANALYSIS:**
    {articles_text}

    **REQUIRED JSON OUTPUT FORMAT:**
    ```json
    {{
      "ticker": "{ticker}",
      "analysis_date": "{os.popen('date +%Y-%m-%d').read().strip()}",
      "news_items": [
        {{
          "headline": "...",
          "sentiment_score": 0,
          "sentiment_category": "POSITIVE/NEGATIVE/NEUTRAL",
          "impact_reason": "..."
        }}
        // ... include an object for every article provided above
      ]
    }}
    ```
    """

    try:
        generation_config = {
            "max_output_tokens": 4096,
            "temperature": 0.0,  # Low temperature forces deterministic, structured output
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Clean the output to ensure valid JSON (remove markdown ticks)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)

    except Exception as e:
        return {"error": f"Error during sentiment analysis: {str(e)}",
                "raw_response": response.text if 'response' in locals() else 'N/A'}


def main():
    """Main function to run the stock news finder and sentiment analyzer."""
    if len(sys.argv) < 2:
        sys.exit(1)
    ticker = sys.argv[1].upper()

    news_api_key = os.getenv('NEWS_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not all([news_api_key, gemini_api_key]):
        print("🛑 Error: One or more API keys/IDs are missing from the .env file.", file=sys.stderr)
        sys.exit(1)

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

    # --- Step 3: Calculation & Presentation (Code + AI Summary) ---
    news_items = analysis_data.get('news_items', [])
    if not news_items:
        print("🛑 The analysis did not return any news items for processing.", file=sys.stderr)
        sys.exit(1)

    # Calculate the average score (The Fear/Greed Index)
    total_score = sum(item.get('sentiment_score', 0) for item in news_items)
    average_score = total_score / len(news_items)

    # Find the most positive and most negative news items
    best_news = max(news_items, key=lambda x: x.get('sentiment_score', 0))
    worst_news = min(news_items, key=lambda x: x.get('sentiment_score', 0))

    # Determine overall sentiment category
    if average_score >= 7.0:
        overall_sentiment = "Strong Bullish (High Greed)"
    elif 6.0 < average_score < 7.0:
        overall_sentiment = "Neutral/Mixed (Uncertainty)"
    else:
        overall_sentiment = "Bearish (High Fear)"

    print("\n" + "=" * 90)
    print(f"📊 FINAL SENTIMENT INDEX FOR {ticker} (Based on {len(news_items)} Articles)")
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


if __name__ == "__main__":
    main()