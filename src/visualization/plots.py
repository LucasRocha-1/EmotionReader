"""
Visualizações básicas de resultados de emoções.
"""

from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt


def plot_emotion_distribution(
    distribution: Dict[str, float],
    title: str = "Distribuição de Emoções",
    save_path: Optional[str] = None
):
    """Gráfico de barras da distribuição de emoções."""
    if not distribution:
        print("Nenhuma distribuição para plotar.")
        return

    emotions = list(distribution.keys())
    values = list(distribution.values())

    colors = {
        "anger": "#E74C3C", "disgust": "#8E44AD", "fear": "#3498DB",
        "happiness": "#F1C40F", "sadness": "#2980B9", "surprise": "#E67E22",
        "neutral": "#95A5A6"
    }
    bar_colors = [colors.get(e, "#7F8C8D") for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(emotions, values, color=bar_colors)
    ax.set_ylabel("Proporção")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confidence_bars(
    scores: Dict[str, float],
    title: str = "Confiança por Emoção",
    save_path: Optional[str] = None
):
    """Barras horizontais de confiança para um único exemplo."""
    if not scores:
        return

    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    emotions, values = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(emotions, values, color="#3498DB")
    ax.set_xlabel("Confiança")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
