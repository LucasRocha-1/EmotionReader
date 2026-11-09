"""
Pré-processamento de imagens para detecção de emoções faciais.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class ImageProcessor:
    """Utilitários básicos de pré-processamento de imagem."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def load_image(self, path: str) -> Optional[np.ndarray]:
        if cv2 is None:
            raise ImportError("opencv-python não está instalado.")
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize(self, image: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("opencv-python não está instalado.")
        return cv2.resize(image, self.target_size)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normaliza para [0, 1]."""
        return image.astype(np.float32) / 255.0

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Pipeline completo: resize + normalize."""
        resized = self.resize(image)
        return self.normalize(resized)
