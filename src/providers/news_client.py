#!/usr/bin/env python3

import json
import os
import time
import requests
import logging
from typing import List, Dict, Any

from tenacity import stop_after_attempt, retry, wait_fixed

from src.core.interfaces import INewsProvider

logger = logging.getLogger(__name__)

class NewsAPIClient(INewsProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        return fetch_articles_raw(ticker, self.api_key, count)


class CachedNewsProvider(INewsProvider):
    def __init__(self, inner_provider: INewsProvider, cache_dir: str = "data/cache", ttl_seconds: int = 3600):
        self.inner_provider = inner_provider
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        cache_file = os.path.join(self.cache_dir, f"{ticker}_news.json")

        if self._is_cache_valid(cache_file):
            logger.info(f"📦 Loading {ticker} news from local cache...")
            return self._load_from_cache(cache_file)

        logger.info(f"🌐 Missed cache. Asking inner provider for {ticker}...")
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
            logger.warning(f"⚠️ Cache save failed: {e}")


class AutoRetryProvider(INewsProvider):
    """
    Retry decorator for INewsProvider.
    """

    def __init__(self, inner_provider: INewsProvider, stats: Any, max_retries: int = 3, wait_seconds: int = 2):
        self.inner_provider = inner_provider
        self.stats = stats
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds

    def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:

        def update_retry_stats(retry_state):
            retries = retry_state.attempt_number - 1
            if retries > 0:
                logger.warning(f"⚠️ Retry #{retries} triggered for {ticker}...")
                self.stats.update('retry_count', retries)

        logger.info(f"🛡️ Entering retry protection zone for {ticker}...")

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_fixed(self.wait_seconds),
            after=update_retry_stats
        )
        def _safe_fetch():
            return self.inner_provider.fetch_articles(ticker, count)

        try:
            return _safe_fetch()
        except Exception as e:
            logger.error(f"💀 All retry attempts failed: {e}")
            return []


def fetch_articles_raw(ticker_symbol: str, news_api_key: str, num_results) -> List[Dict[str, str]]:
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
            logger.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
            return []

        articles = data.get('articles', [])
        filtered_articles = []
        for article in articles:
            if article.get('content') and article.get('title'):
                filtered_article = {
                    'author': article.get('author', ''),
                    'title': article.get('title', ''),
                    'content': article.get('content', '')
                }
                filtered_articles.append(filtered_article)

        return filtered_articles

    except requests.exceptions.RequestException as e:
        logger.error(f"Network Error fetching search results: {e}")
        return []
