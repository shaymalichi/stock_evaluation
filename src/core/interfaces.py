from abc import ABC, abstractmethod
from typing import List, Dict, Any


class INewsProvider(ABC):
    @abstractmethod
    async def fetch_articles(self, ticker: str, count: int) -> List[Dict[str, str]]:
        pass


class IStockAnalyzer(ABC):
    @abstractmethod
    async def filter_relevant(self, ticker: str, articles: List[Dict], count: int) -> List[str]:
        pass

    @abstractmethod
    async def analyze(self, ticker: str, articles: List[str]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def synthesize(self, ticker: str, analysis_results: List[Dict]) -> Dict[str, Any]:
        pass
