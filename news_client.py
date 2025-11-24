#!/usr/bin/env python3

import requests
from typing import List, Dict

from interfaces import INewsProvider


class NewsAPIClient(INewsProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        return fetch_articles(ticker, self.api_key, count)

def fetch_articles(ticker_symbol: str, news_api_key: str, num_results) -> List[Dict[str, str]]:
    """
    Search for news articles about a stock ticker using NewsAPI and return filtered article data.

    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        news_api_key: NewsAPI key
        num_results: Number of articles to retrieve (default: 15)

    Returns:
        List of dictionaries containing filtered article data (author, content, title).
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{ticker_symbol} stock",
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
            # We filter only articles that have non-empty content for embedding
            if article.get('content') and article.get('title'):
                filtered_article = {
                    'author': article.get('author', ''),
                    'title': article.get('title', ''),
                    'content': article.get('content', '')
                }
                filtered_articles.append(filtered_article)

        return filtered_articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return []
