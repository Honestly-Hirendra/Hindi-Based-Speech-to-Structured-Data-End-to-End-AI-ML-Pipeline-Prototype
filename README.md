# Hindi Speech-to-Structured Data: End-to-End AI/ML Pipeline

An end-to-end prototype for transforming Hindi speech into structured, machine-actionable information through Automatic Speech Recognition (ASR), NLP-based question transformation, LLM-driven information extraction, deterministic semantic evidence routing, candidate-state aggregation, and classical machine learning.

## Overview

This project implements and evaluates an end-to-end pipeline for processing Hindi speech and transforming interview responses into structured candidate information and downstream matching predictions.

The overall workflow is:

```text
Hindi Speech
     │
     ▼
Hindi Speech Recognition — IndicConformer 600M
     │
     ▼
Question Transformation & Normalization
     │
     ▼
Structured Information Extraction — Gemma 3 1B IT + QLoRA
     │
     ▼
Semantic Evidence Router — Deterministic Evidence Handling
     │
     ▼
Candidate State Aggregation
     │
     ▼
Matching V6 — TF-IDF + LinearSVC
     │
     ▼
Independent Evaluation & Validation
```

## Pipeline Components

### 1. Hindi Speech Recognition

The upstream speech-recognition stage evaluates IndicConformer 600M for Hindi speech recognition.

Locked STT benchmark:

- WER: 12.98%
- CER: 9.20%
- Inference time: 0.1352 s/sample
- Real-time factor: 0.0261
- Failures: 0

A separate 553-sample Hindi test produced 18.8019% WER and 9.7966% CER. These are separate evaluations and should not be conflated with the locked benchmark.

### 2. Screening & Question Transformation

Interview questions are transformed and normalized into a downstream schema suitable for extraction.

- 4,437 interview-question rows
- 4,411 transformed successfully
- 26 retained without transformation
- No missing or extra IDs in final alignment

### 3. Structured Information Extraction

Gemma 3 1B IT was adapted using QLoRA for structured extraction from interview question-answer pairs.

Evaluation:

- 888 held-out test examples
- JSON validity: 99.662%
- Schema validity: 98.649%
- Field-presence F1: 43.181%

These structural metrics should not be interpreted as human-annotated semantic accuracy.

### 4. Semantic Evidence Router

A deterministic guardrail operating alongside the LLM extraction stage.

It provides:

- Explicit-value preservation
- Deterministic evidence handling
- Conflict handling
- Controlled candidate-state updates
- Protection against valid evidence being unintentionally erased during later inference passes

### 5. Candidate State Aggregation

Candidate State aggregates validated information across interview interactions into a consistent structured representation.

```text
InterviewRequest
      ↓
Gemma Extraction
      ↓
Semantic Evidence Router
      ↓
Candidate State
```

### 6. Matching V6

Matching V6 is the classical machine-learning component.

It combines:

- Word-level TF-IDF
- Character-level TF-IDF
- LinearSVC classification
- Feature engineering
- Model evaluation
- Error analysis

Official test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 73.2218% |
| Macro F1 | 72.6433% |
| Precision | 72.4056% |
| Recall | 72.9322% |
| POTENTIAL_FIT Recall | 72.2689% |
| False POTENTIAL_FIT | 29 |

The 73.22% accuracy is the official test-set result and must not be described as unseen-job accuracy.

Supplementary job-disjoint evaluation:

- Accuracy: 40.1846%
- Macro F1: 29.1424%
- False POTENTIAL_FIT: 588

This exposes the generalization gap under job-disjoint distribution shift.

## Evaluation & Validation

Validation results include:

- 21/21 end-to-end checks passed
- 36/36 STT-to-InterviewRequest fixture checks passed
- 239/239 matching prediction-parity checks passed
- 27/27 matching integration checks passed

These checks validate implementation consistency and integration behavior; they do not imply universal semantic correctness.

## Evaluation Philosophy

The project separates:

**Model performance**
- WER/CER for speech recognition
- Accuracy/Macro F1 for matching
- JSON/schema validity for extraction

**Integration validation**
- End-to-end checks
- Prediction-parity checks
- Interface/fixture validation

**Generalization analysis**
- Job-disjoint matching evaluation is reported separately from the official test set.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── transcription/
│   ├── screening/
│   ├── extraction/
│   ├── semantic_router/
│   └── matching/
├── evaluation/
│   ├── end_to_end/
│   ├── extraction/
│   ├── matching/
│   └── transcription/
├── results/
│   ├── confusion_matrix.png
│   ├── extraction_metrics.json
│   ├── matching_metrics.json
│   └── transcription_summary.json
└── docs/
    ├── architecture.md
    ├── evaluation.md
    └── methodology.md
```

## Technical Stack

### Programming
- Python
- SQL
- C

### Machine Learning
- scikit-learn
- LinearSVC
- TF-IDF
- DBSCAN
- Isolation Forest
- Local Outlier Factor
- XGBoost

### Deep Learning / NLP
- PyTorch
- Transformers
- Gemma 3 1B IT
- QLoRA
- LoRA
- Structured generation
- NLP / information extraction

### Speech
- IndicConformer 600M
- Automatic Speech Recognition
- WER / CER evaluation

### Data & Analysis
- Pandas
- NumPy
- Feature engineering
- EDA
- Statistical analysis
- Error analysis
- Cross-validation
- Hyperparameter tuning

### Visualization
- Matplotlib
- Seaborn
- Power BI
- Tableau
- Excel

## Data and Model Artifacts

This public repository is intended as a reproducible portfolio and technical demonstration.

Sensitive or large internal artifacts are intentionally excluded, including:

- Candidate-level interview datasets
- Resume data
- Audio recordings
- API keys and secrets
- Model checkpoints
- Large training artifacts
- Internal experiment dumps

The repository focuses on source code, methodology, evaluation logic, selected metrics, and project structure.

## Reproducibility

Each major component has a defined conceptual contract:

```text
Transcription
Input:  Hindi audio
Output: Transcript + WER/CER evaluation

Screening
Input:  Interview question
Output: Transformed/normalized question

Extraction
Input:  Question + answer
Output: Structured JSON

Semantic Router
Input:  Answer + extracted evidence
Output: Resolved evidence/state update

Matching
Input:  Candidate state + job representation
Output: Match prediction
```

Complete production path:

```text
STT
 ↓
InterviewRequest
 ↓
Screening
 ↓
Gemma Extraction
 ↓
Semantic Router
 ↓
Candidate State
 ↓
Matching V6
```

## Limitations

### Speech-to-question linkage
The STT-to-InterviewRequest fixture validates interface compatibility but does not establish real-world audio-to-question semantic linkage.

### Extraction evaluation
JSON validity and schema validity measure structural correctness, not human-annotated semantic accuracy.

### Matching generalization
The job-disjoint evaluation shows substantially lower performance than the official test set, indicating sensitivity to distribution shift and the need for broader training data.

### Human evaluation
Human-annotated semantic evaluation remains a future improvement for extraction and candidate-state correctness.

### Audio-question mapping
A robust mechanism for mapping arbitrary real-world audio segments to the correct interview questions remains future work.

### Reproducibility environment
A fully pinned, environment-independent reproduction setup remains future work.

## Future Improvements

1. Human-annotated semantic evaluation for information extraction.
2. Larger and more diverse matching datasets.
3. Better job-disjoint generalization.
4. Broader semantic-router coverage.
5. Robust audio-to-question mapping.
6. Fully reproducible environment and dependency pinning.
7. Expanded error analysis and human review.
8. Additional validation on distribution-shifted data.

## Project Summary

This project demonstrates an end-to-end AI/ML workflow spanning:

- Speech recognition
- NLP preprocessing
- Structured information extraction
- LLM fine-tuning
- Deterministic evidence routing
- Candidate-state aggregation
- Classical machine learning
- Model evaluation
- Integration testing
- Generalization analysis

The main engineering principle is separation of probabilistic model components from deterministic validation and state-management logic, making the system easier to evaluate, debug, and reason about.

## Important Metric Interpretation

- 12.98% WER / 9.20% CER refers to the locked STT benchmark.
- 18.8019% WER / 9.7966% CER refers to the separate 553-sample Hindi test.
- 73.2218% matching accuracy is the official test-set result.
- 40.1846% matching accuracy is from the supplementary job-disjoint evaluation.
- Extraction JSON/schema metrics represent structural validity, not human semantic accuracy.
- 21/21, 36/36, 239/239, and 27/27 are validation/integration checks, not model-accuracy claims.
