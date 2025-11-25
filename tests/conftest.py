import sys
import os
from unittest.mock import patch

import pytest

os.environ["NEWS_API_KEY"] = "test_news_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["ARTICLES_TO_FETCH"] = "10"
os.environ["ARTICLES_TO_INFERENCE"] = "2"

# Add the root directory to PATH so tests can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_logger_file_handler():
    """
    מונע יצירת קבצים וגם דואג שה-Mock יתנהג כמו Handler תקין
    כדי לא לשבור את הלוגים הפנימיים של asyncio.
    """
    with patch("logging.FileHandler") as MockHandler:
        # We define a 'level' field for the mocked Instance as a number (0)
        # This prevents the error in comparison (int >= MagicMock)
        MockHandler.return_value.level = 0
        yield