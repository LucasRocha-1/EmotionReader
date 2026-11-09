"""
Classificador de emoções a partir de texto.

Foco principal do EmotionReader nesta versão.
Projetado para:
  - Uso standalone
  - Integração com outros sistemas (via EmotionResult)
  - Evolução futura (novos modelos, ensemble, calibração)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence
import logging
import re
import unicodedata

from src.schema.emotion_result import (
    EmotionResult,
    BatchEmotionResult,
    CANONICAL_EMOTIONS,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


# Mapeamento de labels de modelos comuns → labels canônicos
LABEL_MAP = {
    # j-hartmann/emotion-english-distilroberta-base
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happiness",
    "happiness": "happiness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    # variantes comuns
    "happy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "surprised": "surprise",
    # pysentimiento / outros
    "others": "neutral",
    "joy_": "happiness",
}


class TextEmotionClassifier:
    """
    Classificador de emoções em texto.

    Características:
    - Carrega modelo Hugging Face sob demanda
    - Fallback robusto em português/inglês quando o modelo não está disponível
    - Normaliza labels para o conjunto canônico
    - Retorna EmotionResult (contrato estável para integração)
    - Batch eficiente quando o pipeline está carregado
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: str = "cpu",
        max_length: int = 128,
        language_hint: str = "pt",
        min_confidence_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.language_hint = language_hint
        self.min_confidence_threshold = min_confidence_threshold
        self.pipeline = None
        self._backend = "fallback"
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=self.max_length,
                device=0 if self.device == "cuda" else -1,
            )
            self._backend = "transformers"
            logger.info("Modelo de texto carregado: %s", self.model_name)
        except Exception as e:
            logger.warning(
                "Não foi possível carregar o modelo de texto (%s). Usando fallback. Erro: %s",
                self.model_name,
                e,
            )
            self.pipeline = None
            self._backend = "fallback"

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    def predict(self, text: str, original_text: Optional[str] = None) -> EmotionResult:
        """Prediz emoção para um único texto e retorna EmotionResult."""
        if not text or not str(text).strip():
            return self._empty_result(original_text or text or "")

        if self.pipeline is not None:
            return self._predict_transformers(text, original_text or text)
        return self._predict_fallback(text, original_text or text)

    def predict_batch(
        self,
        texts: Sequence[str],
        original_texts: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> BatchEmotionResult:
        """
        Prediz emoções para um lote.
        Quando o backend transformers está ativo, usa batch real do pipeline.
        """
        originals = list(original_texts) if original_texts is not None else list(texts)
        cleaned = [t if t is not None else "" for t in texts]

        results: List[EmotionResult] = []

        if self.pipeline is not None and cleaned:
            try:
                # Pipeline do transformers aceita lista
                raw_batch = self.pipeline(cleaned)
                for i, raw in enumerate(raw_batch):
                    results.append(
                        self._build_result_from_raw(
                            raw_scores=raw,
                            cleaned_text=cleaned[i],
                            original_text=originals[i],
                        )
                    )
            except Exception as e:
                logger.warning("Batch transformers falhou, caindo para item-a-item: %s", e)
                for i, t in enumerate(cleaned):
                    results.append(self.predict(t, original_text=originals[i]))
        else:
            for i, t in enumerate(cleaned):
                results.append(self.predict(t, original_text=originals[i]))

        return BatchEmotionResult(
            results=results,
            source=source,
            metadata={
                "model_name": self.model_name,
                "backend": self._backend,
                "language_hint": self.language_hint,
            },
        )

    # ------------------------------------------------------------------
    # Implementações internas
    # ------------------------------------------------------------------

    def _predict_transformers(self, text: str, original_text: str) -> EmotionResult:
        raw = self.pipeline(text)[0]
        return self._build_result_from_raw(raw, text, original_text)

    def _build_result_from_raw(
        self,
        raw_scores: List[Dict[str, Any]],
        cleaned_text: str,
        original_text: str,
    ) -> EmotionResult:
        scores: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}

        for item in raw_scores:
            label = str(item.get("label", "")).lower().strip()
            score = float(item.get("score", 0.0))
            canonical = LABEL_MAP.get(label, label)
            if canonical in scores:
                # se houver colisão (dois labels mapeando pro mesmo), fica o maior
                scores[canonical] = max(scores[canonical], score)
            else:
                # label desconhecido → joga em neutral com peso baixo ou ignora
                scores["neutral"] = max(scores["neutral"], score * 0.3)

        # normaliza para somar ~1 (alguns modelos já somam 1)
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}

        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]

        if confidence < self.min_confidence_threshold:
            dominant = "neutral"
            # não altera os scores, só a decisão

        return EmotionResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            dominant_emotion=dominant,
            confidence=confidence,
            scores=scores,
            modality="text",
            model_name=self.model_name,
            model_backend=self._backend,
            language_hint=self.language_hint,
        )

    def _predict_fallback(self, text: str, original_text: str) -> EmotionResult:
        """Fallback lexical reforçado para português + inglês."""
        text_lower = self._remove_accents(text.lower())
        tokens = re.findall(r"[^\W_]+", text_lower, flags=re.UNICODE)
        negations = {"nao", "nunca", "jamais", "sem", "not", "never", "no"}
        intensifiers = {
            "muito", "muita", "muitos", "muitas", "extremamente", "super",
            "demais", "very", "really", "extremely", "so",
        }
        diminishers = {
            "pouco", "pouca", "ligeiramente", "slightly", "somewhat", "little",
        }

        # Lexicons mais ricos (pt + en)
        lexicons: Dict[str, List[str]] = {
            "happiness": [
                "feliz", "felicidade", "alegre", "alegria", "ótimo", "otimo",
                "maravilhoso", "excelente", "amor", "gostei", "adoro", "animado",
                "contente", "grato", "gratidão", "sucesso", "vitória", "vitoria",
                "happy", "happiness", "joy", "joyful", "great", "awesome",
                "love", "loved", "excited", "wonderful", "amazing", "glad",
            ],
            "sadness": [
                "triste", "tristeza", "chateado", "deprimido", "melancólico",
                "melancolico", "desanimado", "desmotivado", "sólitário", "solitario",
                "mágoa", "magoa", "choro", "chorando", "perda", "saudade",
                "sad", "sadness", "unhappy", "depressed", "lonely", "cry",
                "crying", "miserable", "heartbroken", "down",
            ],
            "anger": [
                "raiva", "irritado", "furioso", "ódio", "odio", "bravo",
                "revoltado", "indignado", "putz", "merda", "droga", "ódio",
                "angry", "anger", "mad", "furious", "hate", "hated", "rage",
                "annoyed", "pissed", "outraged",
            ],
            "fear": [
                "medo", "assustado", "ansioso", "ansiedade", "pânico", "panico",
                "preocupado", "tenso", "receio", "temor", "apavorado",
                "fear", "fearful", "scared", "afraid", "anxious", "anxiety",
                "worried", "panic", "terrified", "nervous",
            ],
            "surprise": [
                "surpreso", "surpresa", "uau", "nossa", "inesperado", "chocado",
                "impressionado", "caramba", "meudeus", "não acredito",
                "surprised", "surprise", "wow", "shocked", "amazed", "unexpected",
                "unbelievable", "astonished",
            ],
            "disgust": [
                "nojo", "repugnante", "nojento", "asco", "enjoado", "horrível",
                "horrivel", "repulsa",
                "disgust", "disgusting", "gross", "nasty", "revolting", "sick",
            ],
            "neutral": [
                "ok", "normal", "regular", "comum", "neutro", "tanto faz",
                "indiferente", "sem problemas", "tudo bem",
                "okay", "fine", "alright", "neutral", "whatever",
            ],
        }

        scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for emotion, words in lexicons.items():
            for w in words:
                normalized_word = self._remove_accents(w.lower())
                pattern = re.compile(
                    rf"(?<!\w){re.escape(normalized_word)}(?!\w)",
                    flags=re.UNICODE,
                )
                for match in pattern.finditer(text_lower):
                    token_start = len(
                        re.findall(r"[^\W_]+", text_lower[:match.start()], re.UNICODE)
                    )
                    preceding_tokens = tokens[max(0, token_start - 3):token_start]
                    if any(token in negations for token in preceding_tokens):
                        scores["neutral"] += 0.5
                        continue

                    weight = 1.0
                    if any(token in intensifiers for token in preceding_tokens[-2:]):
                        weight += 0.5
                    elif any(token in diminishers for token in preceding_tokens[-2:]):
                        weight *= 0.6
                    scores[emotion] += weight

        total = sum(scores.values())
        if total == 0:
            scores["neutral"] = 1.0
            dominant = "neutral"
            confidence = 0.55
        else:
            scores = {k: v / total for k, v in scores.items()}
            dominant = max(scores, key=scores.get)
            confidence = min(0.92, 0.40 + scores[dominant] * 0.55)

        return EmotionResult(
            original_text=original_text,
            cleaned_text=text,
            dominant_emotion=dominant,
            confidence=confidence,
            scores=scores,
            modality="text",
            model_name="lexical_fallback_pt_en",
            model_backend="fallback",
            language_hint=self.language_hint,
            metadata={"note": "fallback lexical (modelo transformers não carregado)"},
        )

    @staticmethod
    def _remove_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )

    def _empty_result(self, original_text: str) -> EmotionResult:
        scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
        scores["neutral"] = 1.0
        return EmotionResult(
            original_text=original_text,
            cleaned_text="",
            dominant_emotion="neutral",
            confidence=1.0,
            scores=scores,
            modality="text",
            model_name=self.model_name,
            model_backend=self._backend,
            language_hint=self.language_hint,
            metadata={"note": "empty input"},
        )

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    def is_model_loaded(self) -> bool:
        return self.pipeline is not None
