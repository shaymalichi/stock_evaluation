# Stock Evaluation – News Sentiment Analyzer

This tool fetches recent news articles for a given stock ticker and uses Google Gemini to perform a structured sentiment analysis on them. It then computes an overall “fear/greed” style sentiment index for the stock and highlights the most positive and most negative news items.
 
---

## Features

- Fetches recent news articles related to a stock (via NewsAPI).
- Sends article snippets to the Gemini API for structured sentiment analysis.
- Computes an average sentiment score across articles.
- Classifies overall sentiment (e.g., Bullish, Neutral, Bearish).
- Highlights:
  - Most positive article
  - Most negative article

---

## Requirements

- Python 3.14 (or compatible 3.x version)
- `virtualenv` for managing the virtual environment
- NewsAPI key
- Google Gemini API key

---

## Setup

1. **Clone the project** (or place the script in a directory of your choice).

2. **Create and activate a virtual environment** (recommended):

   ```bash
   virtualenv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the project root and add:

   ```bash
   NEWS_API_KEY=your_newsapi_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## Usage

Run the script from the project root, passing a stock ticker symbol as an argument:
```bash
  python main.py TSLA
```

The script will:

1. Fetch recent news for the ticker.
2. Send the articles to Gemini for sentiment analysis.
3. Print:
   - The overall sentiment index and category.
   - The most positive and most negative headlines, with brief explanations.

Run the server to expose the api
```bash
   uvicorn src.api.server:app --reload
```
there is a ui interface for the api at `http://localhost:8000/docs`

---

## Example Output (Conceptual)
```
text
🚀 [STEP 1] Searching for recent news articles for TSLA...
🧠 [STEP 2] Found 15 articles. Sending for structured sentiment analysis...

==========================================================================================
📊 FINAL SENTIMENT INDEX FOR TSLA (Based on 15 Articles)
==========================================================================================
Overall Average Score: 6.75 / 10.00
Overall Sentiment:     Neutral/Mixed (Uncertainty)


⭐ Most Positive News:
  Score: 9/10. | Category: POSITIVE
  Headline: Tesla shares surge after strong earnings beat
  Reason: Strong earnings drive bullish expectations for future performance.

🔻 Most Negative News:
  Score: 3/10. | Category: NEGATIVE
  Headline: Regulatory probe raises concerns over Tesla’s safety practices
  Reason: Potential fines and reputational damage may weigh on the stock.

==========================================================================================
```
---

## Notes

- The quality of sentiment analysis depends on:
  - The relevance and freshness of the retrieved news.
  - The Gemini model’s interpretation of the snippets.
- If no articles are found or the analysis fails, the script will print an error and exit with a non-zero status code.

