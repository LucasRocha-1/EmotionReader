"""
Classificador de emoções faciais (esqueleto).
Pode ser evoluído com modelos treinados em FER2013, AffectNet, etc.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FacialEmotionClassifier:
    """
    Esqueleto de classificador de emoções a partir de faces.
    Atualmente usa heurística simples baseada em landmarks (placeholder).
    Em produção, substituir por modelo CNN/Transformer fine-tuned.
    """

    EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    def __init__(self, model_type: str = "simple"):
        self.model_type = model_type
        self.model = None
        # Aqui no futuro: carregar pesos de um modelo treinado
        logger.info(f"FacialEmotionClassifier inicializado (tipo={model_type}) – modo esqueleto")

    def predict(self, face_image: np.ndarray, landmarks: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prediz emoção a partir de uma imagem de face (já recortada).
        """
        # Placeholder: retorna distribuição quase uniforme com leve viés para neutral
        # Substituir por inferência real do modelo
        scores = {emo: 1.0 / len(self.EMOTIONS) for emo in self.EMOTIONS}
        scores["neutral"] = 0.35
        # Normaliza
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}

        dominant = max(scores, key=scores.get)

        return {
            "dominant_emotion": dominant,
            "confidence": scores[dominant],
            "all_scores": scores,
            "note": "esqueleto – substituir por modelo treinado"
        }

    def predict_from_bbox(self, image: np.ndarray, bbox: tuple, landmarks: Optional[Dict] = None) -> Dict[str, Any]:
        """Atalho: recorta a face e prediz."""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        if face.size == 0:
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.0,
                "all_scores": {},
                "error": "face crop vazio"
            }
        return self.predict(face, landmarks)
