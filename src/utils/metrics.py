"""
Métricas e análises auxiliares.
"""

from typing import List, Dict, Any
from collections import Counter


def emotion_distribution(results: List[Dict[str, Any]], key: str = "dominant_emotion") -> Dict[str, float]:
    """Calcula distribuição percentual de emoções."""
    emotions = [r.get(key, "unknown") for r in results]
    total = len(emotions)
    if total == 0:
        return {}
    counts = Counter(emotions)
    return {emo: count / total for emo, count in counts.items()}
