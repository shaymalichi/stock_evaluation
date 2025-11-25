from src.utils.stats_collector import StatsCollector


def test_initialization(tmp_path):
    """Tests that the class is initialized with correct values in a temporary path"""
    temp_file = tmp_path / "test_stats.csv"

    stats = StatsCollector(filename=str(temp_file))
    stats.set_initial_context(ticker="TSLA", articles_to_fetch=50, articles_to_inference=5)

    assert stats.stats['ticker'] == "TSLA"
    assert stats.stats['run_status'] == "IN_PROGRESS"


def test_calculate_analysis_metrics(tmp_path):
    """Tests the calculation of sentiment averages and counts"""
    temp_file = tmp_path / "test_stats.csv"
    stats = StatsCollector(filename=str(temp_file))

    mock_news_items = [
        {'sentiment_score': 8, 'sentiment_category': 'POSITIVE'},
        {'sentiment_score': 2, 'sentiment_category': 'NEGATIVE'},
        {'sentiment_score': 5, 'sentiment_category': 'NEUTRAL'}
    ]

    analysis_data = {'news_items': mock_news_items}

    stats.calculate_analysis_metrics(analysis_data, analysis_duration_sec=1.5)

    assert stats.stats['relevant_articles_found'] == 3
    assert stats.stats['sentiment_score_avg'] == 5.0
