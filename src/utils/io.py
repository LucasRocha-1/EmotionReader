"""
Utilitários de entrada/saída.
"""

import json
from pathlib import Path
from typing import Any, Union, List, Dict


def save_results(results: Union[Dict, List], path: str) -> None:
    """Salva resultados em JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {path}")


def load_results(path: str) -> Any:
    """Carrega resultados de um JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
