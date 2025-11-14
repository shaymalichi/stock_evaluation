#!/usr/bin/env python3

import requests
from typing import List, Dict

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