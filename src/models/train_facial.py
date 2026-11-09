"""
Esqueleto para treinamento de classificador de emoções faciais.
Ideal para datasets como FER2013, AffectNet ou RAF-DB.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def train_facial_emotion_model(
    data_dir: str,
    output_dir: str = "models/facial_emotion",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4
):
    """
    Placeholder de treinamento.
    Fluxo sugerido:
    1. Carregar dataset (ImageFolder ou custom Dataset)
    2. Data augmentation (flip, rotação, brightness)
    3. Modelo base (ResNet18 / EfficientNet / MobileNet)
    4. Fine-tuning + early stopping
    5. Salvar best weights + métricas
    """
    logger.info("Treinamento de modelo facial ainda não implementado.")
    logger.info(f"Parâmetros recebidos: data_dir={data_dir}, epochs={epochs}, batch_size={batch_size}")
    # TODO: implementar com PyTorch ou TensorFlow/Keras
    raise NotImplementedError("Implementar treinamento com dataset de emoções faciais.")


if __name__ == "__main__":
    print("Use: from src.models.train_facial import train_facial_emotion_model")
