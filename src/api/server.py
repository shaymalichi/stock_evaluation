#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.core.config import settings
from src.core.pipeline import StockAnalysisPipeline
from src.providers.news_client import NewsAPIClient, CachedNewsProvider, AutoRetryProvider
from src.providers.analysis_client import GeminiAnalyzer
from src.utils.stats_collector import StatsCollector
from src.core.logger import setup_logging
import logging

pipeline: StockAnalysisPipeline = None
logger = logging.getLogger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    setup_logging()
    logger.info("🚀 Server starting up...")

    news_key = settings.NEWS_API_KEY.get_secret_value()
    gemini_key = settings.GEMINI_API_KEY.get_secret_value()

    stats = StatsCollector()

    base_provider = NewsAPIClient(api_key=news_key)
    retry_provider = AutoRetryProvider(base_provider, stats=stats)
    cached_provider = CachedNewsProvider(retry_provider, ttl_seconds=settings.CACHE_TTL_SECONDS)
    analyzer = GeminiAnalyzer(api_key=gemini_key)

    pipeline = StockAnalysisPipeline(cached_provider, analyzer, stats)

    yield

    logger.info("🛑 Server shutting down...")


app = FastAPI(title="Stock AI Analyst", lifespan=lifespan)


class AnalysisRequest(BaseModel):
    ticker: str


@app.post("/analyze")
def analyze_stock(request: AnalysisRequest):
    """
    Receives a ticker, runs the Pipeline and returns the complete report.
    """
    logger.info(f"📨 Received analysis request for {request.ticker}")

    try:
        fetch_count = settings.ARTICLES_TO_FETCH
        inference_count = settings.ARTICLES_TO_INFERENCE

        report = pipeline.run(request.ticker, fetch_count, inference_count)
        return report

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
