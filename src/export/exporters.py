"""
Exportadores e adaptadores para integração com outros sistemas.

Objetivo: facilitar o uso das emoções como dado por:
  - APIs / microserviços
  - Filas de mensagens / event buses
  - Bancos de dados / data lakes
  - Jogos, dashboards, sistemas de recomendação, etc.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable
from datetime import datetime
from xml.sax.saxutils import escape

from src.schema.emotion_result import (
    EMOTION_LABELS_PT,
    EmotionResult,
    BatchEmotionResult,
)


def export_json(
    data: Union[EmotionResult, BatchEmotionResult, List[EmotionResult], Dict],
    path: str,
    indent: int = 2,
) -> str:
    """Exporta para arquivo JSON. Retorna o path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, (EmotionResult, BatchEmotionResult)):
        payload = data.to_dict()
    elif isinstance(data, list):
        payload = [r.to_dict() if isinstance(r, EmotionResult) else r for r in data]
    else:
        payload = data

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

    return path


def export_jsonl(
    results: Iterable[EmotionResult],
    path: str,
) -> str:
    """
    JSON Lines – ideal para streams, logs e ingestão em data lakes.
    Uma linha = um EmotionResult.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return path


def export_csv(
    results: Iterable[EmotionResult],
    path: str,
) -> str:
    """Exporta versão flat (scores em colunas) para CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_flat() for r in results]
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return path

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_pdf(results: Iterable[EmotionResult], path: str) -> str:
    """Exporta resultados revisados para um relatório PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "A dependência do PDF não está instalada. Execute "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    result_list = list(results)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="EmotionCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EmotionCellCenter",
            parent=styles["EmotionCell"],
            alignment=TA_CENTER,
        )
    )
    story = [
        Paragraph("Relatório de análise de emoções", styles["Title"]),
        Paragraph(
            f"Gerado em {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M')}",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(f"Total de textos analisados: {len(result_list)}", styles["Heading2"]),
    ]

    if result_list:
        corrected_count = sum(
            bool(result.metadata.get("reviewed_by_user"))
            for result in result_list
        )
        story.extend(
            [
                Paragraph(
                    f"Emoções corrigidas manualmente: {corrected_count}",
                    styles["BodyText"],
                ),
                Spacer(1, 4 * mm),
            ]
        )

        distribution: Dict[str, int] = {}
        for result in result_list:
            distribution[result.dominant_emotion] = (
                distribution.get(result.dominant_emotion, 0) + 1
            )
        summary_rows = [
            [
                Paragraph("<b>Emoção final</b>", styles["EmotionCell"]),
                Paragraph("<b>Quantidade</b>", styles["EmotionCellCenter"]),
            ]
        ]
        summary_rows.extend(
            [
                Paragraph(
                    escape(EMOTION_LABELS_PT.get(emotion, emotion)),
                    styles["EmotionCell"],
                ),
                Paragraph(str(count), styles["EmotionCellCenter"]),
            ]
            for emotion, count in sorted(distribution.items())
        )
        summary = Table(summary_rows, colWidths=[95 * mm, 35 * mm], hAlign="LEFT")
        summary.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([summary, Spacer(1, 7 * mm)])

        rows = [
            [
                Paragraph("<b>Texto</b>", styles["EmotionCell"]),
                Paragraph("<b>Detectada</b>", styles["EmotionCellCenter"]),
                Paragraph("<b>Final</b>", styles["EmotionCellCenter"]),
                Paragraph("<b>Confiança modelo</b>", styles["EmotionCellCenter"]),
            ]
        ]
        for result in result_list:
            predicted = result.metadata.get(
                "predicted_emotion",
                result.dominant_emotion,
            )
            rows.append(
                [
                    Paragraph(escape(result.original_text), styles["EmotionCell"]),
                    Paragraph(
                        escape(EMOTION_LABELS_PT.get(predicted, predicted)),
                        styles["EmotionCellCenter"],
                    ),
                    Paragraph(
                        escape(
                            EMOTION_LABELS_PT.get(
                                result.dominant_emotion,
                                result.dominant_emotion,
                            )
                        ),
                        styles["EmotionCellCenter"],
                    ),
                    Paragraph(f"{result.confidence:.1%}", styles["EmotionCellCenter"]),
                ]
            )

        results_table = Table(
            rows,
            colWidths=[89 * mm, 31 * mm, 31 * mm, 27 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        results_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(results_table)
    else:
        story.append(Paragraph("Nenhum texto foi analisado.", styles["BodyText"]))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Relatório de análise de emoções",
        author="EmotionReader",
    )
    document.build(story)
    return str(output_path)


def result_to_event(
    result: EmotionResult,
    event_type: str = "emotion.detected",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Converte EmotionResult em um evento pronto para
    message brokers (Kafka, RabbitMQ, Redis Streams, etc.)
    ou webhooks.

    Exemplo de consumo:
        event = result_to_event(result, extra={"user_id": "u123", "channel": "support"})
        publish(event)
    """
    event = {
        "event_type": event_type,
        "event_id": result.id,
        "occurred_at": result.timestamp,
        "schema_version": result.schema_version,
        "payload": {
            "dominant_emotion": result.dominant_emotion,
            "confidence": result.confidence,
            "scores": result.scores,
            "text_preview": result.original_text[:200],
            "modality": result.modality,
            "model_backend": result.model_backend,
        },
    }
    if extra:
        event["payload"].update(extra)
    return event
