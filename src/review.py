"""Helpers for reviewing and correcting emotion predictions."""

from __future__ import annotations

from dataclasses import replace

from src.schema.emotion_result import CANONICAL_EMOTIONS, EmotionResult


def review_emotion_result(result: EmotionResult, emotion: str) -> EmotionResult:
    """Return a reviewed copy, preserving the original model prediction."""
    if emotion not in CANONICAL_EMOTIONS:
        raise ValueError(f"Emoção inválida: {emotion}")

    original_emotion = result.metadata.get(
        "predicted_emotion",
        result.dominant_emotion,
    )
    metadata = {
        **result.metadata,
        "predicted_emotion": original_emotion,
        "reviewed_by_user": emotion != original_emotion,
    }

    return replace(
        result,
        dominant_emotion=emotion,
        metadata=metadata,
    )
