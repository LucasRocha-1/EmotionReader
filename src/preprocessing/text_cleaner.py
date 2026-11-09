"""
Pré-processamento de texto para análise de emoções.
Foco em português brasileiro + inglês.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


class TextCleaner:
    """
    Limpeza e normalização de textos para classificação de emoções.

    Mantém a maior parte do sinal emocional (exclamações, interjeições)
    enquanto remove ruído (URLs, menções, caracteres de controle).
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        collapse_whitespace: bool = True,
        strip: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.collapse_whitespace = collapse_whitespace
        self.strip = strip

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text

        if self.normalize_unicode:
            # NFC mantém acentos corretamente (importante para PT)
            cleaned = unicodedata.normalize("NFC", cleaned)

        if self.remove_urls:
            cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

        if self.remove_emails:
            cleaned = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", cleaned)

        if self.remove_mentions:
            cleaned = re.sub(r"@\w+", " ", cleaned)

        # Remove caracteres de controle e a maioria dos símbolos,
        # mas preserva pontuação emocional comum e acentos PT
        cleaned = re.sub(
            r"[^\w\sàáâãäéêëíïóôõöúüçÀÁÂÃÄÉÊËÍÏÓÔÕÖÚÜÇ.,!?;:()\-'\"…]",
            " ",
            cleaned,
            flags=re.UNICODE,
        )

        if self.collapse_whitespace:
            cleaned = re.sub(r"\s+", " ", cleaned)

        if self.strip:
            cleaned = cleaned.strip()

        if self.lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]
