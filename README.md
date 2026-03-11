# AVeriTeC Pipeline – Modular Claim Verification Framework

This repository contains a **clean, modular implementation of a claim verification pipeline inspired by the AVeriTeC benchmark**.
The goal of this project is to provide a **research-friendly framework** that allows experiments with different retrieval, reasoning, and verification strategies for fact-checking tasks.

The system takes a **claim** as input and produces:

* evidence retrieved from the web
* answers to verification questions
* stance classification for each evidence
* a final verdict
* a justification

The implementation was designed to be **dataset-agnostic**, allowing the pipeline to be applied to **AVeriTeC or any other claim verification dataset**.

---

# Project Goals

The project was built with the following goals:

* Provide a **clean reimplementation of the AVeriTeC pipeline**
* Allow **modular experimentation**
* Enable easy replacement of components such as:

  * retrievers
  * rerankers
  * question generators
  * stance classifiers
* Support **evaluation across different datasets**

This makes the project suitable for **research and experimentation**, particularly in the context of **fact-checking and misinformation detection**.

---

# Pipeline Overview

The verification process follows this pipeline:

```
Claim
  ↓
Question Generation
  ↓
Web Search
  ↓
Document Parsing
  ↓
Passage Segmentation
  ↓
Passage Retrieval (BM25)
  ↓
Question Answering
  ↓
Stance Classification
  ↓
Verdict Aggregation
  ↓
Justification Generation
```

Each step is implemented as an **independent module**, making the pipeline easy to extend.

---

# Pipeline Steps

## 1. Claim Input

The pipeline begins with a **claim** extracted from a dataset.

Example:

```
"Hunter Biden had no experience in the energy sector before Burisma."
```

---

## 2. Question Generation

An LLM generates **verification questions** that help investigate the claim.

Example questions:

```
Did Hunter Biden have experience in the energy sector before joining Burisma?
What was Hunter Biden's professional background before Burisma?
```

These questions guide the evidence retrieval process.

---

## 3. Web Search

The system queries a web search engine (currently **Brave Search API**) using:

```
claim + generated questions
```

The results provide a list of URLs potentially containing relevant evidence.

A **search cache** is used to avoid repeated API calls.

---

## 4. Document Parsing

Web pages are downloaded and cleaned using:

```
trafilatura
```

The extracted text becomes the raw evidence corpus.

A **page cache** stores parsed pages locally to avoid repeated downloads.

---

## 5. Passage Segmentation

Documents are split into **overlapping text chunks**.

Example configuration:

```
chunk size: 100 tokens
overlap: 20 tokens
```

Chunking improves retrieval quality by preserving contextual information.

---

## 6. Passage Retrieval

Relevant passages are selected using **BM25 retrieval**.

The retrieval query combines:

```
claim + generated questions
```

This multi-query strategy improves recall.

Top passages are selected as candidate evidence.

---

## 7. Question Answering

For each generated question, the system extracts answers from the retrieved passages.

The result is a set of **Question–Answer pairs** that summarize the available evidence.

---

## 8. Stance Classification

Each retrieved passage is classified according to its stance relative to the claim:

```
SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
CONFLICTING
```

This step determines how the evidence relates to the claim.

---

## 9. Verdict Aggregation

The final claim label is derived from the set of stance predictions.

Example logic:

```
REFUTED evidence → REFUTED
SUPPORTED evidence → SUPPORTED
mixed evidence → CONFLICTING
no evidence → NOT ENOUGH EVIDENCE
```

---

## 10. Justification Generation

The system generates a textual explanation summarizing the evidence supporting the final verdict.

---

# Evaluation

The framework includes an evaluation module supporting the following metrics:

```
Accuracy
Precision
Recall
F1 Score (per class)
Macro F1
```

Macro F1 is particularly important because fact-checking datasets are typically **class-imbalanced**.

Example output:

```
Accuracy: 0.28

Per-class metrics:

supported
Precision: 0.429
Recall:    0.667
F1:        0.522

refuted
Precision: 1.000
Recall:    0.062
F1:        0.118

Macro F1: 0.311
```

---

# Project Structure

```
averitec/

datasets/
    dev.json

pipeline/
    pipeline.py
    context.py

modules/

    question_generation/
        question_generator.py

    search/
        web_search.py

    parsing/
        document_parser.py

    segmentation/
        passage_extractor.py

    retrieval/
        bm25_retriever.py

    qa/
        qa_generator.py

    stance/
        stance_classifier.py

    verdict/
        verdict_aggregator.py

    justification/
        justification_generator.py

evaluation/
    metrics.py
    evaluate_predictions.py

utils/
    cache_utils.py
    page_cache.py

cache/
    search_cache.json
    pages/

run_pipeline.py
run_dataset.py
run_evaluation.py
```

---

# Running the Pipeline

## 1. Install dependencies

Example environment:

```
pip install rank_bm25
pip install trafilatura
pip install requests
```

---

## 2. Configure the search API

Create a `.env` file:

```
BRAVE_API_KEY=your_api_key_here
```

---

## 3. Run a single claim

```
python run_pipeline.py
```

---

## 4. Run the dataset

```
python run_dataset.py
```

This generates:

```
predictions.json
```

---

## 5. Evaluate predictions

```
python run_evaluation.py
```

---

# Caching System

The pipeline includes two caching layers.

## Search Cache

Stores search results for each query.

```
cache/search_cache.json
```

Reduces API calls.

---

## Page Cache

Stores parsed web pages.

```
cache/pages/
```

Prevents repeated downloads of the same URL.

This significantly speeds up experiments.

---

# Research Extensions

The framework was designed to support experimentation with alternative components.

Examples:

### Retrieval

```
BM25
ColBERT
Hybrid Retrieval
Dense Retrieval
```

### Reranking

```
MonoT5
BGE reranker
Cross-encoder
```

### Reasoning

```
Chain-of-thought verification
Multi-hop QA
Evidence aggregation models
```

---

# Potential Future Improvements

Possible research directions include:

* Hybrid retrieval (BM25 + embeddings)
* Cross-encoder reranking
* Better stance detection models
* Evidence clustering
* Multi-hop reasoning
* Retrieval from structured knowledge bases

---

# Intended Use

This repository is intended for:

* fact-checking research
* misinformation detection experiments
* evaluation of retrieval strategies
* experimentation with LLM reasoning pipelines

---

# License

This project is intended for **research and educational purposes**.
