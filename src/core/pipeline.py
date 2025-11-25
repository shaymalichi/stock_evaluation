import logging
import time
from src.utils.stats_collector import StatsCollector
from src.core.interfaces import INewsProvider, IStockAnalyzer

logger = logging.getLogger(__name__)

class StockAnalysisPipeline:
    def __init__(self, news_provider: INewsProvider, analyzer: IStockAnalyzer, stats: StatsCollector):
        self.news_provider = news_provider
        self.analyzer = analyzer
        self.stats = stats

    def run(self, ticker: str, fetch_count: int, inference_count: int):
        logger.info(f"🔄 Pipeline started for {ticker}")  # שימוש במקום print

        logger.debug(f"Step 1: Fetching {fetch_count} articles...")
        start_time_fetch = time.time()

        articles = self.news_provider.fetch_articles(ticker, fetch_count)

        end_time_fetch = time.time()
        fetch_duration = end_time_fetch - start_time_fetch

        self.stats.update('articles_requested', fetch_count)
        self.stats.update('articles_returned', len(articles))
        self.stats.update('articles_after_filter', len(articles))
        self.stats.update('fetch_duration_sec', round(fetch_duration, 3))
        self.stats.update('news_fetch_status', 'OK' if articles else 'NO_ARTICLES')

        if not articles:
            logger.warning(f"⚠️ No articles found for {ticker}")
            raise Exception("No articles found")

        logger.info(f"✅ Fetched {len(articles)} articles.")

        print(f"✅ Fetched {len(articles)} articles in {fetch_duration:.2f}s.")

        print(" 2. Filtering relevant news (RAG)...")
        relevant_texts = self.analyzer.filter_relevant(ticker, articles, count=inference_count)

        self.stats.update('relevant_articles_found', len(relevant_texts))

        if not relevant_texts:
            raise Exception("RAG returned no relevant articles.")

        print(f" 3. Analyzing {len(relevant_texts)} articles concurrently...")
        start_time_analysis = time.time()

        analysis_data = self.analyzer.analyze(ticker, relevant_texts)

        end_time_analysis = time.time()
        analysis_duration = end_time_analysis - start_time_analysis

        self.stats.calculate_analysis_metrics(analysis_data, analysis_duration)

        if not analysis_data.get('news_items'):
            raise Exception("Analysis failed to produce items.")

        print("✨ Synthesizing final report...")
        start_time_synthesis = time.time()

        final_report = self.analyzer.synthesize(ticker, analysis_data['news_items'])

        end_time_synthesis = time.time()
        synthesis_duration = end_time_synthesis - start_time_synthesis

        self.stats.calculate_synthesis_metrics(final_report, synthesis_duration)

        return final_report
