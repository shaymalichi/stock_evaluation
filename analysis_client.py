#!/usr/bin/env python3

import json
from typing import List, Dict, Any
import faiss
import google.generativeai as genai
import numpy as np
import config
from news_client import NUM_OF_ARTICLES_FOR_ANALYSIS

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
    genai.configure(api_key=config.GEMINI_API_KEY)
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
    index.add(embeddings_np) # type: ignore[arg-type]
    return article_texts, index


def search_relevant_articles(ticker_symbol: str, article_texts: list, index: faiss.Index) -> list:
    """
    Search for relevant articles based on a query about stock price and sentiment impact.

    Args:
        ticker_symbol: Stock ticker symbol
        article_texts: List of article content texts
        index: FAISS index containing article embeddings

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
    k = NUM_OF_ARTICLES_FOR_ANALYSIS
    D, I = index.search(query_embedding_np, k)
    relevant_indices = I[0]
    return [article_texts[i] for i in relevant_indices]

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