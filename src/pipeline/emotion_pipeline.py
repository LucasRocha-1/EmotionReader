"""
Pipeline principal de leitura de emoções.

Nesta versão o foco é texto. A modalidade facial permanece
como extensão pronta para ser ativada/ consumida da mesma forma.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Sequence
import logging
from pathlib import Path
import yaml

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.image_processor import ImageProcessor
from src.detection.text_emotion import TextEmotionClassifier
from src.detection.face_detector import FaceDetector
from src.detection.facial_emotion import FacialEmotionClassifier
from src.corrections import CorrectionMemory
from src.schema.emotion_result import EmotionResult, BatchEmotionResult

logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Pipeline unificado.

    Uso principal (texto):
        pipeline = EmotionPipeline()
        result = pipeline.analyze_text("Estou muito feliz!")
        batch  = pipeline.analyze_text_batch(["msg1", "msg2"], source="chat")

    O retorno é sempre EmotionResult / BatchEmotionResult
    (contrato estável para integração com outros sistemas).
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        corrections_path: Optional[str] = None,
    ):
        self.config = self._load_config(config_path)
        self.text_cleaner = TextCleaner()
        self.image_processor = ImageProcessor()
        self.correction_memory = CorrectionMemory(corrections_path)

        self._text_classifier: Optional[TextEmotionClassifier] = None
        self._face_detector: Optional[FaceDetector] = None
        self._facial_classifier: Optional[FacialEmotionClassifier] = None

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config não encontrada em %s. Usando defaults.", path)
            return {}

    # ------------------------------------------------------------------
    # Componentes (lazy)
    # ------------------------------------------------------------------

    @property
    def text_classifier(self) -> TextEmotionClassifier:
        if self._text_classifier is None:
            cfg = self.config.get("text_model", {})
            self._text_classifier = TextEmotionClassifier(
                model_name=cfg.get(
                    "name",
                    "j-hartmann/emotion-english-distilroberta-base",
                ),
                device=cfg.get("device", "cpu"),
                max_length=cfg.get("max_length", 128),
                language_hint=cfg.get("language_hint", "pt"),
                min_confidence_threshold=cfg.get("min_confidence_threshold", 0.0),
            )
        return self._text_classifier

    @property
    def face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            conf = self.config.get("face", {}).get("min_detection_confidence", 0.5)
            self._face_detector = FaceDetector(min_detection_confidence=conf)
        return self._face_detector

    @property
    def facial_classifier(self) -> FacialEmotionClassifier:
        if self._facial_classifier is None:
            self._facial_classifier = FacialEmotionClassifier()
        return self._facial_classifier

    # ------------------------------------------------------------------
    # Texto (foco principal)
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> EmotionResult:
        """Analisa um único texto e retorna EmotionResult."""
        cleaned = self.text_cleaner.clean(text)
        result = self.text_classifier.predict(cleaned, original_text=text)
        return self.correction_memory.apply(result)

    def save_correction(self, text: str, emotion: str) -> None:
        """Persist a manually reviewed label for future analyses."""
        self.correction_memory.save(text, emotion)

    def analyze_text_batch(
        self,
        texts: Sequence[str],
        source: Optional[str] = None,
    ) -> BatchEmotionResult:
        """
        Analisa um lote de textos.
        `source` é metadado livre (ex: "support_tickets", "game_dialog", "twitter").
        """
        cleaned = self.text_cleaner.clean_batch(list(texts))
        batch = self.text_classifier.predict_batch(
            cleaned,
            original_texts=list(texts),
            source=source,
        )
        batch.results = self.correction_memory.apply_many(batch.results)
        return batch

    # ------------------------------------------------------------------
    # Facial (extensão – mantida para evolução futura)
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Detecta faces e classifica emoções em uma imagem (esqueleto)."""
        image = self.image_processor.load_image(image_path)
        if image is None:
            return {
                "error": f"Não foi possível carregar a imagem: {image_path}",
                "modality": "facial",
            }

        faces = self.face_detector.detect(image)
        face_results = []

        for i, face in enumerate(faces):
            emotion = self.facial_classifier.predict_from_bbox(
                image, face["bbox"], face.get("keypoints")
            )
            face_results.append(
                {
                    "face_id": i,
                    "bbox": face["bbox"],
                    "detection_confidence": face["confidence"],
                    **emotion,
                }
            )

        return {
            "image_path": image_path,
            "num_faces": len(faces),
            "faces": face_results,
            "modality": "facial",
        }

    # ------------------------------------------------------------------
    # Interface genérica
    # ------------------------------------------------------------------

    def analyze(
        self,
        input_data: Union[str, List[str]],
        mode: str = "auto",
        source: Optional[str] = None,
    ) -> Union[EmotionResult, BatchEmotionResult, Dict[str, Any]]:
        """
        Interface unificada.
        - mode="text" (default) → EmotionResult / BatchEmotionResult
        - mode="facial" → dict (esqueleto)
        - mode="auto" → decide pela extensão do arquivo ou assume texto
        """
        if mode == "auto":
            if isinstance(input_data, str) and Path(input_data).suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp",
            }:
                mode = "facial"
            else:
                mode = "text"

        if mode == "text":
            if isinstance(input_data, (list, tuple)):
                return self.analyze_text_batch(input_data, source=source)
            return self.analyze_text(input_data)

        if mode == "facial":
            return self.analyze_image(input_data)

        raise ValueError(f"Modo não suportado: {mode}")
