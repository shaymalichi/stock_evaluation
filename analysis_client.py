#!/usr/bin/env python3

import os
import json
from typing import List, Dict, Any
import google.generativeai as genai


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
    4. **Headline:** Use the original headline.

    **DATA FOR ANALYSIS:**
    {article_raw_text}

    **REQUIRED JSON OUTPUT FORMAT:**
    ```json
    {{
      "headline": "{article.get('title', 'N/A')}",
      "sentiment_score": 0,
      "sentiment_category": "POSITIVE/NEGATIVE/NEUTRAL",
      "impact_reason": "..."
    }}
    ```
    """

    try:
        generation_config = {
            "max_output_tokens": 1024,
            "temperature": 0.0,
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

    articles_raw_text = "\n\n---\n\n".join([
        f"AUTHOR: {article['author']}\n"
        f"TITLE: {article['title']}\n"
        f"CONTENT: {article['content']}"
        for article in articles
    ])

    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are a professional Senior Capital Market Analyst. Your mission is to analyze the following news snippets concerning the company {ticker} and provide a detailed, structured sentiment analysis. You must analyze each article individually.

    **STRICT INSTRUCTIONS:**
    1. **ONLY RETURN VALID JSON.** Do not include any other text, greetings, or explanations outside the JSON structure.
    2. **Sentiment Score:** A number from 1 (Extremely Negative, High Risk) to 10 (Extremely Positive, High Opportunity).
    3. **Impact Reason:** A short explanation (max 15 words) on the potential impact on the stock price.

    **DATA FOR ANALYSIS:**
    {articles_raw_text}

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

        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)

    except Exception as e:
        return {"error": f"Error during sentiment analysis: {str(e)}",
                "raw_response": response.text if 'response' in locals() else 'N/A'}