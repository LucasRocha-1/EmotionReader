# EmotionReader – Sistema de Leitura de Emoções

Sistema de **reconhecimento de emoções** com **foco em texto**, projetado para ser consumido por outros programas e evoluir no futuro (facial / multimodal).

Alinhado com as competências do currículo:
- **NLP** e classificação de texto
- **Machine Learning** (classificação, avaliação)
- **Pré-processamento**, visualização e pipelines de Ciência de Dados
- Visão Computacional básica (módulo facial como extensão)

**Versão do contrato de dados:** `1.0.0` (`SCHEMA_VERSION`)

---

## Estrutura do Projeto

```
EmotionReader/
├── config/
│   └── settings.yaml              # Configurações (modelo de texto, etc.)
├── data/
│   ├── raw/
│   ├── processed/                 # JSON / JSONL / CSV de saída
│   └── samples/
├── src/
│   ├── schema/                    # Contrato EmotionResult (integração)
│   ├── preprocessing/             # TextCleaner (PT + EN)
│   ├── detection/                 # TextEmotionClassifier (+ facial esqueleto)
│   ├── export/                    # JSON, JSONL, CSV, eventos
│   ├── models/                    # Treinamento / avaliação futuros
│   ├── pipeline/                  # EmotionPipeline
│   ├── visualization/
│   └── utils/
├── notebooks/
├── tests/
├── main.py
└── requirements.txt
```

---

## Como começar

```bash
cd EmotionReader
pip install -r requirements.txt
python main.py
python -m pytest tests/ -v
```

O `main.py` abre uma interface gráfica para:
1. Analisar textos (um por linha)
2. Revisar e corrigir a emoção detectada em cada texto
3. Gerar um relatório PDF com as emoções detectadas e revisadas

O PDF pode ser salvo no local escolhido na janela de exportação. Para gerar o
relatório, instale as dependências com `pip install -r requirements.txt`.
As correções aplicadas são salvas localmente em
`data/processed/emotion_corrections.sqlite3` e reutilizadas em análises futuras.
Correspondências semelhantes são conservadoras; frases próximas com correções
conflitantes não substituem a previsão do modelo. Esse histórico não é enviado
para serviços externos.

---

## Contrato de dados (o mais importante para integração)

Todo resultado de texto é um **`EmotionResult`** (dataclass + `to_dict()`).

```python
from src.pipeline.emotion_pipeline import EmotionPipeline

pipeline = EmotionPipeline()
result = pipeline.analyze_text("Estou muito feliz com o resultado!")

print(result.dominant_emotion)   # "happiness"
print(result.confidence)         # 0.0 – 1.0
print(result.scores)             # dict canônico com as 7 emoções
print(result.to_dict())          # payload estável para APIs / bancos
```

### Campos principais de `EmotionResult`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | str (uuid) | Identificador único do resultado |
| `schema_version` | str | Versão do contrato (`1.0.0`) |
| `timestamp` | str (ISO 8601) | Momento da análise (UTC) |
| `original_text` | str | Texto original |
| `cleaned_text` | str | Texto após pré-processamento |
| `dominant_emotion` | str | Emoção dominante (canônica) |
| `confidence` | float | Confiança da emoção dominante |
| `scores` | dict[str, float] | Score de cada emoção (soma ≈ 1) |
| `modality` | str | `"text"` (ou futuramente facial/multimodal) |
| `model_name` | str | Nome do modelo usado |
| `model_backend` | str | `transformers` \| `fallback` |
| `language_hint` | str | Ex: `"pt"` |
| `metadata` | dict | Campo livre para extensões |

### Emoções canônicas

`anger` · `disgust` · `fear` · `happiness` · `sadness` · `surprise` · `neutral`

### Lote

```python
batch = pipeline.analyze_text_batch(
    ["msg1", "msg2", "msg3"],
    source="support_tickets"   # metadado livre
)
batch.to_dict()                # JSON completo
batch.to_flat_list()           # lista de dicts achatados (bom para CSV/DataFrame)
batch.dominant_distribution()  # {"happiness": 0.4, ...}
```

---

## Exportação e integração com outros programas

```python
from src.export.exporters import export_json, export_jsonl, export_csv, result_to_event

export_json(batch, "saida.json")
export_jsonl(batch.results, "saida.jsonl")   # stream / data lake
export_csv(batch.results, "saida.csv")       # scores em colunas

# Evento para fila / webhook / game engine
event = result_to_event(
    result,
    event_type="emotion.detected",
    extra={"user_id": "u123", "channel": "chat"}
)
```

O formato de evento é pensado para ser publicado em Kafka, RabbitMQ, Redis Streams, webhooks, etc.

---

## Componentes de texto

| Módulo | Descrição | Status |
|--------|-----------|--------|
| `TextCleaner` | Limpeza PT/EN (URLs, menções, unicode NFC, pontuação emocional) | Pronto |
| `TextEmotionClassifier` | Modelo HF + fallback lexical rico em português | Pronto |
| `EmotionResult` / `BatchEmotionResult` | Contrato estável de saída | Pronto |
| Exportadores | JSON, JSONL, CSV, evento | Pronto |
| `EmotionPipeline.analyze_text*` | API principal de texto | Pronto |

### Backend

1. **Preferencial:** modelo Hugging Face configurado em `config/settings.yaml`
2. **Fallback automático:** léxico português + inglês (funciona offline e sem `torch`/transformers)

---

## Extensão futura

O desenho atual facilita:

1. **Trocar/adicionar modelos de texto**  
   Basta alterar `text_model.name` no YAML ou passar outro nome no construtor.

2. **Combinar com outro programa**  
   Consumir `EmotionResult.to_dict()` ou o evento gerado por `result_to_event`.

3. **Ativar facial / multimodal**  
   O módulo facial já existe como esqueleto e pode passar a emitir o mesmo `EmotionResult` (mudando só `modality`).

4. **Calibração, ensemble, tracking temporal**  
   O campo `metadata` e a versão de schema permitem evoluir sem quebrar consumidores.

---

Projeto preparado para expansão e integração.  
Foco atual: **texto como fonte de emoção + contrato de dados claro**.
