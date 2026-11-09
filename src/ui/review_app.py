"""Desktop interface for analyzing and reviewing text emotions."""

from __future__ import annotations

from queue import Empty, Queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import sqlite3
from typing import Optional

from src.export.exporters import export_pdf
from src.pipeline.emotion_pipeline import EmotionPipeline
from src.review import review_emotion_result
from src.schema.emotion_result import (
    CANONICAL_EMOTIONS,
    EMOTION_LABELS_PT,
    EmotionResult,
)


class EmotionReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        pipeline: Optional[EmotionPipeline] = None,
    ):
        self.root = root
        self.pipeline = pipeline or EmotionPipeline()
        self.results: list[EmotionResult] = []
        self.reviewed_emotions: list[str] = []
        self._analysis_messages: Queue[tuple[str, object]] = Queue()
        self._busy = False

        self.root.title("EmotionReader - Revisão de emoções")
        self.root.geometry("1100x760")
        self.root.minsize(850, 620)
        self._build_interface()

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        ttk.Label(
            container,
            text="EmotionReader",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            container,
            text="Analise os textos, revise as emoções e gere um relatório PDF.",
        ).grid(row=1, column=0, sticky=tk.W, pady=(2, 14))

        input_frame = ttk.LabelFrame(container, text="Textos para análise", padding=10)
        input_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 12))
        input_frame.columnconfigure(0, weight=1)
        self.input_text = tk.Text(input_frame, height=5, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky=tk.EW)
        ttk.Label(
            input_frame,
            text="Digite um texto por linha. Linhas vazias serão ignoradas.",
        ).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

        results_frame = ttk.LabelFrame(container, text="Resultados", padding=10)
        results_frame.grid(row=3, column=0, sticky=tk.NSEW, pady=(0, 12))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        columns = ("text", "detected", "memory", "reviewed", "confidence")
        self.results_table = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            height=9,
            selectmode="browse",
        )
        self.results_table.heading("text", text="Texto")
        self.results_table.heading("detected", text="Detectada")
        self.results_table.heading("memory", text="Memória")
        self.results_table.heading("reviewed", text="Em revisão")
        self.results_table.heading("confidence", text="Confiança modelo")
        self.results_table.column("text", width=540, minwidth=250)
        self.results_table.column("detected", width=110, anchor=tk.CENTER)
        self.results_table.column("memory", width=100, anchor=tk.CENTER)
        self.results_table.column("reviewed", width=110, anchor=tk.CENTER)
        self.results_table.column("confidence", width=115, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(
            results_frame,
            orient=tk.VERTICAL,
            command=self.results_table.yview,
        )
        self.results_table.configure(yscrollcommand=scrollbar.set)
        self.results_table.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.results_table.bind("<<TreeviewSelect>>", self._select_result)

        review_frame = ttk.LabelFrame(container, text="Revisar resultado selecionado", padding=10)
        review_frame.grid(row=4, column=0, sticky=tk.EW, pady=(0, 12))
        review_frame.columnconfigure(1, weight=1)
        self.selected_text = tk.Text(review_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.selected_text.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 8))
        ttk.Label(review_frame, text="Emoção correta:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8)
        )
        self.emotion_choice = tk.StringVar()
        self.emotion_box = ttk.Combobox(
            review_frame,
            textvariable=self.emotion_choice,
            values=[EMOTION_LABELS_PT[emotion] for emotion in CANONICAL_EMOTIONS],
            state="readonly",
            width=18,
        )
        self.emotion_box.grid(row=1, column=1, sticky=tk.W)
        self.apply_review_button = ttk.Button(
            review_frame,
            text="Aplicar correção",
            command=self._apply_review,
            state=tk.DISABLED,
        )
        self.apply_review_button.grid(row=1, column=2, sticky=tk.E, padx=(8, 0))

        footer = ttk.Frame(container)
        footer.grid(row=5, column=0, sticky=tk.EW)
        footer.columnconfigure(1, weight=1)
        self.analyze_button = ttk.Button(
            footer,
            text="Analisar textos",
            command=self._start_analysis,
        )
        self.analyze_button.grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(footer, text="Pronto para analisar.")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=12)
        self.pdf_button = ttk.Button(
            footer,
            text="Finalizar e gerar PDF",
            command=self._generate_pdf,
            state=tk.DISABLED,
        )
        self.pdf_button.grid(row=0, column=2, sticky=tk.E)

    def _start_analysis(self) -> None:
        texts = [
            line.strip()
            for line in self.input_text.get("1.0", tk.END).splitlines()
            if line.strip()
        ]
        if not texts:
            messagebox.showwarning("Nenhum texto", "Digite pelo menos um texto para analisar.")
            return

        self._busy = True
        self.analyze_button.configure(state=tk.DISABLED)
        self.pdf_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Analisando textos...")
        threading.Thread(
            target=self._analyze_in_background,
            args=(texts,),
            daemon=True,
        ).start()
        self.root.after(100, self._check_analysis_result)

    def _analyze_in_background(self, texts: list[str]) -> None:
        try:
            batch = self.pipeline.analyze_text_batch(texts, source="emotion_review_app")
            self._analysis_messages.put(("success", batch.results))
        except Exception as exc:
            self._analysis_messages.put(("error", str(exc)))

    def _check_analysis_result(self) -> None:
        try:
            status, payload = self._analysis_messages.get_nowait()
        except Empty:
            self.root.after(100, self._check_analysis_result)
            return

        self._busy = False
        self.analyze_button.configure(state=tk.NORMAL)
        if status == "error":
            self.pdf_button.configure(state=tk.NORMAL if self.results else tk.DISABLED)
            self.status_label.configure(text="A análise falhou.")
            messagebox.showerror("Erro na análise", str(payload))
            return

        self.results = list(payload)
        self.reviewed_emotions = [result.dominant_emotion for result in self.results]
        self._refresh_results()
        self.pdf_button.configure(state=tk.NORMAL if self.results else tk.DISABLED)
        self.status_label.configure(text=f"{len(self.results)} texto(s) analisado(s).")
        if self.results:
            first_item = self.results_table.get_children()[0]
            self.results_table.selection_set(first_item)
            self.results_table.focus(first_item)
            self._select_result()

    def _refresh_results(self) -> None:
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        for index, result in enumerate(self.results):
            preview = " ".join(result.original_text.split())
            if len(preview) > 100:
                preview = preview[:97] + "..."
            self.results_table.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    preview,
                    EMOTION_LABELS_PT.get(
                        result.metadata.get("predicted_emotion", result.dominant_emotion),
                        result.dominant_emotion,
                    ),
                    (
                        EMOTION_LABELS_PT[result.dominant_emotion]
                        if result.metadata.get("correction_source") == "memory"
                        else "—"
                    ),
                    EMOTION_LABELS_PT[self.reviewed_emotions[index]],
                    f"{result.confidence:.1%}",
                ),
            )

    def _select_result(self, _event: Optional[tk.Event] = None) -> None:
        if _event is not None and _event.widget is not self.results_table:
            return
        selection = self.results_table.selection()
        if not selection:
            self.apply_review_button.configure(state=tk.DISABLED)
            return

        index = int(selection[0])
        result = self.results[index]
        self.selected_text.configure(state=tk.NORMAL)
        self.selected_text.delete("1.0", tk.END)
        self.selected_text.insert("1.0", result.original_text)
        self.selected_text.configure(state=tk.DISABLED)
        self.emotion_choice.set(EMOTION_LABELS_PT[self.reviewed_emotions[index]])
        self.apply_review_button.configure(state=tk.NORMAL)

    def _apply_review(self) -> None:
        selection = self.results_table.selection()
        if not selection:
            messagebox.showinfo("Selecione um resultado", "Selecione um texto para revisar.")
            return

        index = int(selection[0])
        selected_label = self.emotion_choice.get()
        emotion = next(
            (
                key
                for key, label in EMOTION_LABELS_PT.items()
                if label == selected_label
            ),
            None,
        )
        if emotion not in CANONICAL_EMOTIONS:
            messagebox.showerror("Emoção inválida", "Selecione uma emoção da lista.")
            return

        try:
            self.pipeline.save_correction(self.results[index].original_text, emotion)
        except (OSError, sqlite3.Error, ValueError) as exc:
            messagebox.showerror("Erro ao salvar correção", str(exc))
            return

        self.reviewed_emotions[index] = emotion
        self._refresh_results()
        item = str(index)
        self.results_table.selection_set(item)
        self.results_table.focus(item)
        self._select_result()
        self.status_label.configure(text="Correção aplicada.")

    def _generate_pdf(self) -> None:
        if not self.results:
            messagebox.showwarning("Sem resultados", "Analise os textos antes de gerar o PDF.")
            return

        project_root = Path(__file__).resolve().parents[2]
        output_path = filedialog.asksaveasfilename(
            title="Salvar relatório de emoções",
            initialdir=project_root / "data" / "processed",
            initialfile="emotion_analysis.pdf",
            defaultextension=".pdf",
            filetypes=[("Arquivo PDF", "*.pdf")],
        )
        if not output_path:
            return

        reviewed_results = [
            review_emotion_result(result, emotion)
            for result, emotion in zip(self.results, self.reviewed_emotions)
        ]
        try:
            export_pdf(reviewed_results, output_path)
        except Exception as exc:
            messagebox.showerror("Erro ao gerar PDF", str(exc))
            return

        self.status_label.configure(text=f"Relatório salvo em {output_path}")
        messagebox.showinfo("PDF gerado", f"Relatório salvo em:\n{output_path}")


def main() -> None:
    root = tk.Tk()
    EmotionReviewApp(root)
    root.mainloop()
