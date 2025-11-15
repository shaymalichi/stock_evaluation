#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ARTICLES_TO_FETCH = os.getenv('ARTICLES_TO_FETCH')
ARTICLES_TO_INFERENCE = os.getenv('ARTICLES_TO_INFERENCE')

def load_and_validate_keys():
    """
    Checks if API keys are loaded and exits if they are missing.
    Returns:
        (str, str): A tuple containing (news_api_key, gemini_api_key)
    """
    if not all([NEWS_API_KEY, GEMINI_API_KEY]):
        print("🛑 Error: One or more API keys/IDs are missing from the .env file.", file=sys.stderr)
        sys.exit(1)

    return NEWS_API_KEY, GEMINI_API_KEY