# 🧠 Iterative Question Generation for Fact-Checking

This repository contains the implementation of the experiments from the paper:

**"Less is More: Improved Fact-Checking on AVeriTeC with Iterative Question Generation"**

The project investigates the isolated impact of **iterative question generation** in Retrieval-Augmented Generation (RAG) pipelines for fact-checking.

---

## 📌 Overview

This system follows a standard open-domain fact-checking pipeline:

1. Receive a claim
2. Generate verification questions
3. Use questions as search queries
4. Retrieve evidence from the web
5. Rank and aggregate evidence
6. Predict a final label:
   - `SUPPORTED`
   - `REFUTED`
   - `NOT ENOUGH EVIDENCE (NEI)`
   - `CONFLICTING EVIDENCE/CHERRYPICKING`

The key contribution is a controlled comparison between:

- **Single-step question generation**
- **Iterative question generation**

---

## ⚙️ Methods

### 🔹 Single-step
- Generates up to **5 questions in a single pass**
- No access to intermediate evidence
- All queries are independent

### 🔹 Iterative
- Generates **one question at a time**
- Each question is conditioned on:
  - previously generated questions
  - previously retrieved evidence
- Stops early if enough evidence is gathered

👉 The only difference between methods is **how questions are generated**

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/lucasfrag/iterative-question-generation.git
cd iterative-question-generation/
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration (.env)

Create a `.env` file in the root directory:

```env
# =========================
# LLM CONFIGURATION
# =========================
MODEL_NAME=llama3.1:8b
TEMPERATURE=0.0
MAX_TOKENS=4096

# =========================
# RETRIEVAL CONFIGURATION
# =========================
SEARCH_ENGINE=duckduckgo
MAX_URLS=30
TOP_K_BM25=10
TOP_K_RERANK=5

# =========================
# EXPERIMENT CONFIGURATION
# =========================
MAX_QUERIES=5
DATASET_PATH=./data/averitec_dev.json

# =========================
# OPTIONAL
# =========================
CACHE_DIR=./cache
OUTPUT_DIR=./results
```

---

## ⚙️ Configuration Details

### 🔹 LLM Settings

| Variable | Description |
|--------|-------------|
| `MODEL_NAME` | Model used for generation (e.g., LLaMA, Qwen, DeepSeek) |
| `TEMPERATURE` | Set to `0.0` for deterministic outputs |
| `MAX_TOKENS` | Maximum tokens per generation |

---

### 🔹 Retrieval Settings

| Variable | Description |
|--------|-------------|
| `SEARCH_ENGINE` | Uses DuckDuckGo (free and accessible) |
| `MAX_URLS` | Number of retrieved URLs per query |
| `TOP_K_BM25` | Top documents selected via BM25 |
| `TOP_K_RERANK` | Final passages after reranking |

---

### 🔹 Experiment Settings

| Variable | Description |
|--------|-------------|
| `MAX_QUERIES` | Maximum number of questions (N = 5) |
| `DATASET_PATH` | Path to AVeriTeC dataset |

---

## ▶️ Running Experiments

### 🔹 Single-step

```bash
python experiments/run_averitec_original.py
```

---

### 🔹 Iterative

```bash
python experiments/run_averitec_iterative.py
```

---

## 📊 Outputs

Results are saved in:

```
results/
```

Includes:

- Predictions per claim
- Aggregated metrics:
  - Accuracy
  - Macro-F1
  - F1 per class
  - Number of queries
  - Token usage

---

## 📈 Key Findings

- Iterative generation:
  - Improves accuracy (+1.7 to +17.5)
  - Reduces number of queries (~50%)
  - Reduces abstention (NEI)
  - Increases token usage (7–17%)

👉 Key insight:  
**Iterative generation shifts computation from retrieval to reasoning**

---

## 🧪 Dataset

- **AVeriTeC (development set)**
- 500 real-world claims
- Open-domain (web retrieval)

Labels:

- `SUPPORTED`
- `REFUTED`
- `NOT ENOUGH EVIDENCE`
- `CONFLICTING EVIDENCE/CHERRYPICKING`

---

## ⚠️ Reproducibility Notes

Some variability is expected due to:

- Live web search (non-deterministic results)
- Changes in web content over time
- Model/tokenizer differences

💡 Recommendations:
- Use caching (`CACHE_DIR`)
- Fix model versions when possible

---

## 🧠 Important Notes

- All components are kept fixed except **question generation**
- This isolates the effect of iterative reasoning
- The stopping condition is model-driven

---

## 🛠️ Customization

You can easily:

- Change models via `.env`
- Adjust query budget
- Replace search engine
- Modify prompts in `src/prompts/`

---
<!--
## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{fraga2026iterative,
  title={Less is More: Improved Fact-Checking on AVeriTeC with Iterative Question Generation},
  author={Fraga, Lucas M. and others},
  year={2026}
}
```

---
-->