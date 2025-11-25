#!/usr/bin/env python3

import concurrent.futures
import json
import sys
import time
from typing import List, Dict, Any
import faiss
import google.generativeai as genai
import numpy as np

from src.core.config import settings
from src.core.interfaces import IStockAnalyzer


class GeminiAnalyzer(IStockAnalyzer):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def filter_relevant(self, ticker: str, articles: List[Dict], count: int) -> List[str]:
        article_texts, index = embed_articles(articles)
        relevant_texts = search_relevant_articles(ticker, article_texts, index, count)
        return relevant_texts

    def analyze(self, ticker: str, articles: List[str]) -> Dict[str, Any]:
        return analyze_articles_concurrently(ticker, articles, self.api_key)

    def synthesize(self, ticker: str, analysis_results: List[Dict]) -> Dict[str, Any]:
        return synthesize_report(ticker, analysis_results, self.api_key)


# For articles summaries
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "The original headline of the article."},
        "sentiment_score": {"type": "integer",
                            "description": "A score from 1 (Extremely Negative) to 10 (Extremely Positive)."},
        "sentiment_category": {"type": "string", "description": "One of: POSITIVE, NEGATIVE, NEUTRAL."},
        "impact_reason": {"type": "string",
                          "description": "A short summary (max 15 words) explaining the article and its impact."}
    },
    "required": ["headline", "sentiment_score", "sentiment_category", "impact_reason"]
}

# For inference
REPORT_SCHEMA = {
        "type": "object",
        "properties": {
            "overall_summary": {"type": "string"},
            "final_sentiment": {"type": "string", "enum": ["Bullish", "Neutral", "Bearish"]},
            "recommendation": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
            "major_risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overall_summary", "final_sentiment", "recommendation"]
    }


def embed_articles(articles: List[Dict[str, str]]):
    genai.configure(api_key=settings.GEMINI_API_KEY.get_secret_value())
    genai.GenerativeModel('models/text-embedding-004')
    article_texts = [article['content'] for article in articles]

    response = genai.embed_content(
        model='models/text-embedding-004',
        content=article_texts,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings_list = response['embedding']
    d = len(embeddings_list[0])
    embeddings_np = np.array(embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)  # type: ignore[arg-type]
    return article_texts, index


def search_relevant_articles(ticker_symbol: str, article_texts: list, index: faiss.Index, articles_for_inference) -> list:
    """
    Search for relevant articles based on a query about stock price and sentiment impact.

    Args:
        ticker_symbol: Stock ticker symbol
        article_texts: List of article content texts
        index: FAISS index containing article embeddings
        articles_for_inference: Number of articles to filter

    Returns:
        List of relevant article texts
    """
    search_query = f"Significant positive or negative news impacting {ticker_symbol} stock price and sentiment."
    query_embedding_list = genai.embed_content(
        model='models/text-embedding-004',
        content=search_query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    query_embedding_np = np.array([query_embedding_list]).astype('float32')
    D, I = index.search(query_embedding_np, articles_for_inference)
    relevant_indices = I[0]
    return [article_texts[i] for i in relevant_indices]


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
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

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
                    errors.append(result["error"])
                    print(f"⚠️ Thread error: {result['error']}", file=sys.stderr)
            except Exception as exc:
                errors.append(str(exc))
                print(f"A thread generated an exception: {exc}", file=sys.stderr)

    return {
        "ticker": ticker_symbol,
        "analysis_date": time.strftime("%Y-%m-%d"),
        "news_items": results,
        "errors": errors,
        "errors_count": len(errors),
    }


def analyze_single_article(ticker: str, article: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """
    Analyzes a single news article for sentiment using Gemini API and returns a structured JSON
    ready for merging.
    """
    article_raw_text = (
        f"TITLE: {article.get('title', 'N/A')}\n"
        f"CONTENT: {article.get('content', 'N/A')}"
    )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are a professional Senior Capital Market Analyst. Analyze the following news snippet concerning the company {ticker} and provide a structured sentiment analysis.

    **STRICT INSTRUCTIONS:**
    1. **ONLY RETURN VALID JSON.** Do not include any other text, greetings, or explanations.
    2. **Sentiment Score:** A number from 1 (Extremely Negative) to 10 (Extremely Positive).
    3. **Impact Reason (Summary):** A short summary (max 20 words) explaining the article and why it received that score.

    **DATA FOR ANALYSIS:**
    {article_raw_text}
    """

    try:
        generation_config = {
            "max_output_tokens": 1024,
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": ANALYSIS_SCHEMA
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        json_text = response.text.strip().replace('```json', '').replace('```', '')
        result_dict = json.loads(json_text)
        return {"news_items": [result_dict]}

    except Exception as e:
        return {"error": f"Error during sentiment analysis for article: {article.get('title', 'N/A')}: {str(e)}",
                "raw_response": response.text if 'response' in locals() else 'N/A'}


def synthesize_report(ticker: str, analyzed_news_items: List[Dict[str, Any]], api_key: str) -> Dict[str, str]:
    context_text = "Analysis Results from Individual Articles:\n"
    for item in analyzed_news_items:
        context_text += f"- Score {item['sentiment_score']}/10 ({item['sentiment_category']}): {item['impact_reason']} (Source: {item['headline']})\n"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are the Chief Investment Strategist. Your task is to synthesize the following discrete sentiment analysis results for the stock {ticker} and provide a final, actionable recommendation.

    **DATA SYNTHESIS:**
    {context_text}

    **FINAL OUTPUT MUST BE A JSON object with the following structure:**
    1. **overall_summary**: A 2-3 sentence summary of the key findings.
    2. **final_sentiment**: The consolidated sentiment (Bullish, Neutral, Bearish).
    3. **recommendation**: The final action (BUY, HOLD, SELL).
    4. **major_risks**: A list of 2 key risks mentioned in the analysis.

    """

    generation_config = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": REPORT_SCHEMA
    }

    response = model.generate_content(prompt, generation_config=generation_config)
    return json.loads(response.text)
