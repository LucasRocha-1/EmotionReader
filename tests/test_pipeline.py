"""
Testes do EmotionReader – foco em texto e contrato de dados.
"""

from __future__ import annotations

import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.detection.text_emotion import TextEmotionClassifier
from src.schema.emotion_result import (
    EmotionResult,
    BatchEmotionResult,
    CANONICAL_EMOTIONS,
    SCHEMA_VERSION,
)
from src.export.exporters import export_pdf, result_to_event
from src.corrections import CorrectionMemory
from src.pipeline.emotion_pipeline import EmotionPipeline
from src.review import review_emotion_result


def test_text_cleaner_basic():
    cleaner = TextCleaner()
    dirty = "  Olá!!! Visite https://example.com @user   "
    cleaned = cleaner.clean(dirty)
    assert "https" not in cleaned
    assert "@user" not in cleaned
    assert cleaned == cleaned.lower()


def test_text_cleaner_preserves_accents():
    cleaner = TextCleaner(lowercase=True)
    text = "Estou muito feliz e ansioso com a situação!"
    cleaned = cleaner.clean(text)
    assert "feliz" in cleaned
    assert "ansioso" in cleaned


def test_emotion_result_contract():
    r = EmotionResult(
        original_text="teste",
        cleaned_text="teste",
        dominant_emotion="happiness",
        confidence=0.87,
        scores={"happiness": 0.87, "neutral": 0.13},
        model_backend="fallback",
    )
    d = r.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["dominant_emotion"] == "happiness"
    assert "scores" in d
    assert r.to_flat()["score_happiness"] == 0.87

    r2 = EmotionResult.from_dict(d)
    assert r2.dominant_emotion == r.dominant_emotion


def test_text_emotion_fallback():
    clf = TextEmotionClassifier()
    result = clf._predict_fallback(
        "Estou muito feliz e alegre hoje!",
        original_text="Estou muito feliz e alegre hoje!",
    )
    assert isinstance(result, EmotionResult)
    assert result.dominant_emotion in CANONICAL_EMOTIONS
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.scores.keys()) == set(CANONICAL_EMOTIONS)


def test_batch_and_distribution():
    clf = TextEmotionClassifier()
    batch = clf.predict_batch(
        ["Estou feliz", "Estou triste", "Tudo normal"],
        source="test",
    )
    assert isinstance(batch, BatchEmotionResult)
    assert len(batch.results) == 3
    dist = batch.dominant_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_result_to_event():
    r = EmotionResult(
        original_text="Que raiva!",
        dominant_emotion="anger",
        confidence=0.91,
        scores={"anger": 0.91},
    )
    event = result_to_event(r, extra={"user_id": "u1"})
    assert event["event_type"] == "emotion.detected"
    assert event["payload"]["dominant_emotion"] == "anger"
    assert event["payload"]["user_id"] == "u1"


def test_pipeline_text_returns_emotion_result():
    pipeline = EmotionPipeline()
    result = pipeline.analyze_text("Estou muito feliz!")
    assert isinstance(result, EmotionResult)
    assert result.modality == "text"
    assert result.dominant_emotion in CANONICAL_EMOTIONS


def test_review_emotion_result_preserves_prediction():
    result = EmotionResult(
        original_text="Que ótimo!",
        dominant_emotion="happiness",
        confidence=0.8,
        scores={"happiness": 0.8, "sadness": 0.2},
        metadata={"source": "test"},
    )

    reviewed = review_emotion_result(result, "sadness")

    assert result.dominant_emotion == "happiness"
    assert reviewed.dominant_emotion == "sadness"
    assert reviewed.confidence == 0.8
    assert reviewed.metadata["predicted_emotion"] == "happiness"
    assert reviewed.metadata["reviewed_by_user"] is True
    assert reviewed.metadata["source"] == "test"


def test_review_emotion_result_rejects_unknown_emotion():
    result = EmotionResult(dominant_emotion="neutral")

    with pytest.raises(ValueError, match="Emoção inválida"):
        review_emotion_result(result, "unknown")


def test_correction_memory_persists_and_matches_normalized_text(tmp_path):
    memory = CorrectionMemory(tmp_path / "corrections.sqlite3")
    memory.save("Estou MUITO feliz!", "sadness")

    result = EmotionResult(
        original_text="estou muito feliz",
        dominant_emotion="happiness",
        confidence=0.91,
        scores={"happiness": 0.91, "sadness": 0.09},
    )
    corrected = CorrectionMemory(tmp_path / "corrections.sqlite3").apply(result)

    assert corrected.dominant_emotion == "sadness"
    assert corrected.confidence == 0.91
    assert corrected.metadata["predicted_emotion"] == "happiness"
    assert corrected.metadata["correction_source"] == "memory"
    assert corrected.metadata["correction_similarity"] == 1.0


def test_correction_memory_matches_similar_but_not_unrelated_text(tmp_path):
    memory = CorrectionMemory(tmp_path / "corrections.sqlite3")
    memory.save("Estou muito feliz com o resultado", "happiness")

    similar = memory.apply(
        EmotionResult(
            original_text="Estou feliz com o resultado!",
            dominant_emotion="neutral",
        )
    )
    unrelated = memory.apply(
        EmotionResult(
            original_text="Estou muito triste com o resultado",
            dominant_emotion="sadness",
        )
    )

    assert similar.dominant_emotion == "happiness"
    assert similar.metadata["correction_similarity"] >= memory.similarity_threshold
    assert unrelated.dominant_emotion == "sadness"
    assert "correction_source" not in unrelated.metadata


def test_correction_memory_avoids_conflicting_similar_labels(tmp_path):
    memory = CorrectionMemory(tmp_path / "corrections.sqlite3")
    memory.save("Estou muito feliz com o resultado", "happiness")
    memory.save("Estou muito triste com o resultado", "sadness")

    result = memory.apply(
        EmotionResult(
            original_text="Estou contente com o resultado",
            dominant_emotion="neutral",
        )
    )

    assert result.dominant_emotion == "neutral"
    assert "correction_source" not in result.metadata


def test_correction_memory_rejects_empty_text(tmp_path):
    memory = CorrectionMemory(tmp_path / "corrections.sqlite3")

    with pytest.raises(ValueError, match="texto vazio"):
        memory.save("   ", "neutral")


def test_pipeline_applies_saved_corrections_to_single_and_batch_texts(
    tmp_path,
    monkeypatch,
):
    pipeline = EmotionPipeline(corrections_path=str(tmp_path / "corrections.sqlite3"))
    pipeline.save_correction("Estou muito feliz com o resultado", "sadness")

    def use_fallback(classifier):
        classifier._backend = "fallback"

    monkeypatch.setattr(TextEmotionClassifier, "_load_model", use_fallback)

    single = pipeline.analyze_text("Estou muito feliz com o resultado")
    batch = pipeline.analyze_text_batch(["Estou feliz com o resultado!"])

    assert single.dominant_emotion == "sadness"
    assert batch.results[0].dominant_emotion == "sadness"
    assert batch.results[0].metadata["correction_source"] == "memory"


def test_fallback_accounts_for_negation_and_intensifiers():
    classifier = TextEmotionClassifier()
    classifier.pipeline = None

    negated = classifier._predict_fallback(
        "Não estou feliz",
        original_text="Não estou feliz",
    )
    intensified = classifier._predict_fallback(
        "Estou muito feliz",
        original_text="Estou muito feliz",
    )

    assert negated.dominant_emotion != "happiness"
    assert intensified.dominant_emotion == "happiness"


def test_export_pdf_creates_report(tmp_path):
    result = review_emotion_result(
        EmotionResult(
            original_text="O resultado ficou excelente.",
            dominant_emotion="sadness",
            confidence=0.7,
            scores={"happiness": 0.9, "sadness": 0.1},
        ),
        "happiness",
    )
    output_path = tmp_path / "report.pdf"

    exported_path = export_pdf([result], str(output_path))

    assert exported_path == str(output_path)
    assert output_path.read_bytes().startswith(b"%PDF-")
    assert output_path.stat().st_size > 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
