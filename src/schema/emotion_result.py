"""
Contrato de dados de emoção.

Este módulo define a estrutura oficial de saída do EmotionReader.
Outros programas devem consumir este formato (ou o dict equivalente)
para garantir compatibilidade em expansões futuras.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid


# Versão do contrato – incrementa quando a estrutura pública muda
SCHEMA_VERSION = "1.0.0"

# Conjunto canônico de emoções (Ekman + neutral)
CANONICAL_EMOTIONS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)

EMOTION_LABELS_PT = {
    "anger": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happiness": "Alegria",
    "sadness": "Tristeza",
    "surprise": "Surpresa",
    "neutral": "Neutro",
}


@dataclass
class EmotionScore:
    """Score de uma única emoção."""
    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 6)}


@dataclass
class EmotionResult:
    """
    Resultado padronizado de análise de emoção (uma unidade de texto).

    Campos pensados para:
    - Consumo por outros programas (API, eventos, bancos, jogos, etc.)
    - Evolução futura (multimodal, tracking temporal, confiança calibrada)
    """
    # Identificação
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Conteúdo
    original_text: str = ""
    cleaned_text: str = ""

    # Resultado principal
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)  # label -> score (0-1)

    # Metadados do modelo / pipeline
    modality: str = "text"          # text | facial | multimodal
    model_name: Optional[str] = None
    model_backend: str = "unknown"  # transformers | fallback | ensemble
    language_hint: Optional[str] = None

    # Extensibilidade
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialização estável para JSON / APIs / mensageria."""
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "dominant_emotion": self.dominant_emotion,
            "confidence": round(self.confidence, 6),
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "modality": self.modality,
            "model_name": self.model_name,
            "model_backend": self.model_backend,
            "language_hint": self.language_hint,
            "metadata": self.metadata,
        }

    def to_flat(self) -> Dict[str, Any]:
        """
        Versão achatada – útil para CSV, DataFrames, logs de eventos.
        scores_* como colunas separadas.
        """
        flat = {
            "id": self.id,
            "timestamp": self.timestamp,
            "original_text": self.original_text,
            "dominant_emotion": self.dominant_emotion,
            "confidence": round(self.confidence, 6),
            "modality": self.modality,
            "model_backend": self.model_backend,
        }
        for emo in CANONICAL_EMOTIONS:
            flat[f"score_{emo}"] = round(self.scores.get(emo, 0.0), 6)
        return flat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionResult":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            original_text=data.get("original_text", ""),
            cleaned_text=data.get("cleaned_text", ""),
            dominant_emotion=data.get("dominant_emotion", "neutral"),
            confidence=float(data.get("confidence", 0.0)),
            scores=data.get("scores", {}),
            modality=data.get("modality", "text"),
            model_name=data.get("model_name"),
            model_backend=data.get("model_backend", "unknown"),
            language_hint=data.get("language_hint"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchEmotionResult:
    """Resultado de um lote de textos – útil para pipelines e exportação em massa."""
    results: List[EmotionResult] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Optional[str] = None          # ex: "chat_log", "support_tickets", "game_dialog"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": self.source,
            "count": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def to_flat_list(self) -> List[Dict[str, Any]]:
        return [r.to_flat() for r in self.results]

    def dominant_distribution(self) -> Dict[str, float]:
        if not self.results:
            return {}
        from collections import Counter
        counts = Counter(r.dominant_emotion for r in self.results)
        total = len(self.results)
        return {k: v / total for k, v in counts.items()}
