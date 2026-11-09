"""
Detecção de faces usando MediaPipe (leve e eficiente).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    import cv2
except ImportError:
    mp = None
    cv2 = None


class FaceDetector:
    """Detector de faces baseado em MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe e opencv-python são necessários para FaceDetector.")

        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = curto alcance, 1 = longo alcance
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta faces em uma imagem RGB.
        Retorna lista de dicionários com bounding box e confiança.
        """
        results = self.detector.process(image)
        faces = []

        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                faces.append({
                    "bbox": (x, y, width, height),
                    "confidence": detection.score[0] if detection.score else 0.0,
                    "keypoints": self._extract_keypoints(detection, w, h)
                })

        return faces

    def _extract_keypoints(self, detection, w: int, h: int) -> Dict[str, Tuple[int, int]]:
        """Extrai pontos-chave básicos (olho esquerdo, direito, nariz, etc.)."""
        keypoints = {}
        if detection.location_data.relative_keypoints:
            names = ["right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear", "left_ear"]
            for i, kp in enumerate(detection.location_data.relative_keypoints):
                if i < len(names):
                    keypoints[names[i]] = (int(kp.x * w), int(kp.y * h))
        return keypoints

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
        """Recorta a face com padding opcional."""
        x, y, w, h = bbox
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)

        return image[y1:y2, x1:x2]

    def close(self):
        self.detector.close()
