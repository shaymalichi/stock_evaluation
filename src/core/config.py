#!/usr/bin/env python3

import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """
    Application configuration using Pydantic.
    This class automatically loads environment variables from the .env file,
    validates their types, and ensures required keys are present.
    """

    # --- REQUIRED FIELDS ---
    NEWS_API_KEY: SecretStr
    GEMINI_API_KEY: SecretStr

    # --- OPTIONAL FIELDS (With Defaults) ---
    ARTICLES_TO_FETCH: int = Field(default=50, gt=0, description="Number of articles to fetch from NewsAPI")
    ARTICLES_TO_INFERENCE: int = Field(default=5, gt=0, description="Number of articles to pass to the LLM")
    CACHE_TTL_SECONDS: int = Field(default=3600, gt=0, description="Time-to-live for cache in seconds")

    # --- CONFIGURATION ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

try:
    settings = Settings()

except Exception as e:
    print(f"🛑 Configuration Error: {e}", file=sys.stderr)
    sys.exit(1)