"""Persistent, local memory for manually corrected emotion labels."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
import sqlite3
import unicodedata
from pathlib import Path

from src.schema.emotion_result import CANONICAL_EMOTIONS, EmotionResult


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize case, accents, punctuation, and whitespace for matching."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(_TOKEN_PATTERN.findall(without_accents))


def text_similarity(first: str, second: str) -> float:
    """Compare normalized token overlap and phrase order."""
    first_normalized = normalize_text(first)
    second_normalized = normalize_text(second)
    if not first_normalized or not second_normalized:
        return 0.0
    if first_normalized == second_normalized:
        return 1.0

    first_tokens = set(first_normalized.split())
    second_tokens = set(second_normalized.split())
    overlap = len(first_tokens & second_tokens) / len(first_tokens | second_tokens)
    sequence = SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
        autojunk=False,
    ).ratio()
    return 0.55 * overlap + 0.45 * sequence


class CorrectionMemory:
    """Store reviewed text/emotion pairs in a local SQLite database."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        similarity_threshold: float = 0.82,
        ambiguity_margin: float = 0.08,
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.database_path = Path(
            database_path
            or project_root / "data" / "processed" / "emotion_corrections.sqlite3"
        )
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin

    def save(self, text: str, emotion: str) -> None:
        if emotion not in CANONICAL_EMOTIONS:
            raise ValueError(f"Emoção inválida: {emotion}")
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("Não é possível salvar a correção de um texto vazio.")

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path, timeout=10) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_corrections (
                    normalized_text TEXT PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO emotion_corrections (
                    normalized_text, original_text, emotion, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(normalized_text) DO UPDATE SET
                    original_text = excluded.original_text,
                    emotion = excluded.emotion,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized,
                    text.strip(),
                    emotion,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def apply(self, result: EmotionResult) -> EmotionResult:
        return self.apply_many([result])[0]

    def apply_many(self, results: list[EmotionResult]) -> list[EmotionResult]:
        if not results:
            return []

        if not self.database_path.exists():
            return results

        with sqlite3.connect(self.database_path, timeout=10) as connection:
            rows = connection.execute(
                """
                SELECT normalized_text, original_text, emotion
                FROM emotion_corrections
                """
            ).fetchall()

        corrections = [
            (normalized, original, emotion)
            for normalized, original, emotion in rows
            if emotion in CANONICAL_EMOTIONS
        ]
        return [self._apply_one(result, corrections) for result in results]

    def _apply_one(
        self,
        result: EmotionResult,
        corrections: list[tuple[str, str, str]],
    ) -> EmotionResult:
        normalized = normalize_text(result.original_text)
        if not normalized:
            return result

        exact = next(
            (
                correction
                for correction in corrections
                if correction[0] == normalized
            ),
            None,
        )
        if exact is not None:
            match, similarity = exact, 1.0
        else:
            candidates = sorted(
                (
                    (
                        correction,
                        text_similarity(result.original_text, correction[1]),
                    )
                    for correction in corrections
                ),
                key=lambda candidate: candidate[1],
                reverse=True,
            )
            if not candidates or candidates[0][1] < self.similarity_threshold:
                return result
            match, similarity = candidates[0]
            competing_labels = {
                correction[2]
                for correction, score in candidates[1:]
                if score >= self.similarity_threshold
                and score >= similarity - self.ambiguity_margin
            }
            if any(label != match[2] for label in competing_labels):
                return result

        emotion = match[2]
        metadata = {
            **result.metadata,
            "predicted_emotion": result.metadata.get(
                "predicted_emotion",
                result.dominant_emotion,
            ),
            "correction_source": "memory",
            "correction_similarity": round(similarity, 4),
            "correction_example": match[1],
        }
        return replace(result, dominant_emotion=emotion, metadata=metadata)
