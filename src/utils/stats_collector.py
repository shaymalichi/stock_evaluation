import csv
import os
import sys
import time
import json
from typing import Dict, Any, List

CSV_HEADERS = [
    'run_id', 'run_timestamp', 'analysis_date', 'ticker',
    'articles_to_fetch', 'articles_to_inference', 'news_fetch_status', 'articles_requested',
    'articles_returned', 'articles_after_filter', 'fetch_duration_sec',
    'relevant_articles_found', 'analysis_success_count', 'analysis_error_count',
    'analysis_duration_sec', 'sentiment_score_avg', 'sentiment_score_min', 'sentiment_score_max',
    'news_items_positive_count', 'news_items_negative_count', 'news_items_neutral_count',
    'final_sentiment', 'final_recommendation', 'major_risks_json',
    'total_runtime_sec', 'run_status', 'error_stage', 'error_message'
]


class StatsCollector:
    """
    Class for collecting and managing run statistics, saving them to a CSV file.
    """

    def __init__(self, filename: str = os.path.join('data', 'reports', 'run_stats.csv')):
        self.filename = filename
        self.stats: Dict[str, Any] = {}
        self.start_time = time.time()
        # Ensure the reports directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self._initialize_csv()

    def _initialize_csv(self):
        """Initializes the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(CSV_HEADERS)

    def set_initial_context(self, ticker: str, articles_to_fetch: int, articles_to_inference: int):
        """Sets up the initial, context-independent data for the run."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        run_id = f"{int(self.start_time)}_{ticker}"

        self.stats.update({
            'run_id': run_id,
            'run_timestamp': timestamp,
            'analysis_date': time.strftime('%Y-%m-%d'),
            'ticker': ticker,
            'articles_to_fetch': articles_to_fetch,
            'articles_to_inference': articles_to_inference,
            'run_status': 'IN_PROGRESS',
            'error_stage': 'N/A',
            'error_message': 'N/A',
            # Initialize numeric fields to 0 or N/A to ensure CSV consistency
            'articles_requested': 0, 'articles_returned': 0, 'articles_after_filter': 0, 'fetch_duration_sec': 0.0,
            'relevant_articles_found': 0, 'analysis_success_count': 0, 'analysis_error_count': 0,
            'analysis_duration_sec': 0.0, 'sentiment_score_avg': 'N/A', 'sentiment_score_min': 'N/A',
            'sentiment_score_max': 'N/A', 'news_items_positive_count': 0, 'news_items_negative_count': 0,
            'news_items_neutral_count': 0, 'final_sentiment': 'N/A', 'final_recommendation': 'N/A',
            'major_risks_json': 'N/A', 'total_runtime_sec': 0.0, 'news_fetch_status': 'N/A',
        })

    def update(self, key: str, value: Any):
        """Updates a specific statistic key."""
        self.stats[key] = value

    def update_error(self, stage: str, message: str):
        """Updates the run status when an error occurs."""
        self.stats['run_status'] = 'FAILED'
        self.stats['error_stage'] = stage
        self.stats['error_message'] = message.replace('\n', ' | ')  # Remove newlines for single-line CSV cell
        self.finalize()  # Immediately write on error

    def finalize(self, status: str = 'OK'):
        """Calculates final metrics and writes the complete row to CSV."""
        self.stats['total_runtime_sec'] = round(time.time() - self.start_time, 3)
        if self.stats['run_status'] == 'IN_PROGRESS':
            self.stats['run_status'] = status

        try:
            with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
                # Ensure all required headers are present, filling missing with 'N/A'
                row = {header: self.stats.get(header, 'N/A') for header in CSV_HEADERS}
                writer.writerow(row)
            print(f"📊 [STATS] Run results saved to {self.filename} ({self.stats['run_id']}).")
        except Exception as e:
            print(f"🛑 CRITICAL ERROR: Could not write statistics to CSV: {e}", file=sys.stderr)

    def calculate_analysis_metrics(self, analysis_data: Dict[str, Any], analysis_duration_sec: float):
        """Calculates and updates metrics based on the analysis results."""
        news_items: List[Dict[str, Any]] = analysis_data.get('news_items', [])

        self.update('analysis_duration_sec', round(analysis_duration_sec, 3))
        self.update('relevant_articles_found', len(news_items))
        self.update('analysis_success_count', len(news_items))  # Assuming any item in news_items is a success

        if not news_items:
            return

        total_score = sum(item.get('sentiment_score', 0) for item in news_items)
        self.update('sentiment_score_avg', round(total_score / len(news_items), 2))
        self.update('sentiment_score_min', min(item.get('sentiment_score', 0) for item in news_items))
        self.update('sentiment_score_max', max(item.get('sentiment_score', 0) for item in news_items))

        # Count categories
        positive_count = sum(1 for item in news_items if item.get('sentiment_category') == 'POSITIVE')
        negative_count = sum(1 for item in news_items if item.get('sentiment_category') == 'NEGATIVE')
        neutral_count = sum(1 for item in news_items if item.get('sentiment_category') == 'NEUTRAL')

        self.update('news_items_positive_count', positive_count)
        self.update('news_items_negative_count', negative_count)
        self.update('news_items_neutral_count', neutral_count)

    def calculate_synthesis_metrics(self, recommendation_data: Dict[str, Any], synthesis_duration_sec: float):
        """Updates metrics based on the final synthesis results."""
        self.update('synthesis_duration_sec', round(synthesis_duration_sec, 3))
        self.update('final_sentiment', recommendation_data.get('final_sentiment', 'N/A'))
        self.update('final_recommendation', recommendation_data.get('recommendation', 'N/A'))

        # Store major risks as a JSON string
        major_risks = recommendation_data.get('major_risks', [])
        self.update('major_risks_json', json.dumps(major_risks))
