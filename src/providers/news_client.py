#!/usr/bin/env python3

import json
import os
import time

import requests
from typing import List, Dict

from tenacity import stop_after_attempt, retry, wait_fixed

from src.core.interfaces import INewsProvider


class NewsAPIClient(INewsProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        return fetch_articles(ticker, self.api_key, count)


class CachedNewsProvider(INewsProvider):
    """
    A caching wrapper for INewsProvider that persists results to local JSON files.
    """

    def __init__(self, inner_provider: INewsProvider, cache_dir: str = "data/cache", ttl_seconds: int = 3600):
        self.inner_provider = inner_provider
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        cache_file = os.path.join(self.cache_dir, f"{ticker}_news.json")

        if self._is_cache_valid(cache_file):
            print(f"📦 Loading {ticker} news from local cache...")
            return self._load_from_cache(cache_file)

        print(f"🌐 Missed cache. Asking inner provider for {ticker}...")
        articles = self.inner_provider.fetch_articles(ticker, count)

        if articles:
            self._save_to_cache(cache_file, articles)

        return articles

    def _is_cache_valid(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        file_age = time.time() - os.path.getmtime(filepath)
        return file_age < self.ttl_seconds

    def _load_from_cache(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _save_to_cache(self, filepath: str, data: List[Dict]):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")


class AutoRetryProvider(INewsProvider):
    """
    Retry decorator for INewsProvider.
    """

    def __init__(self, inner_provider: INewsProvider, max_retries: int = 3, wait_seconds: int = 2):
        self.inner_provider = inner_provider
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        print(f"🛡️ entering retry protection zone for {ticker}...")

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(self.wait_seconds))
        def _safe_fetch():
            return self.inner_provider.fetch_articles(ticker, count)

        try:
            return _safe_fetch()
        except Exception as e:
            print(f"💀 All retry attempts failed: {e}")
            return []


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
