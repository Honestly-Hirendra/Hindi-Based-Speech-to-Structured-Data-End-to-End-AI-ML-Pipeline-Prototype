# Hindi Speech-to-Structured Data: End-to-End AI/ML Pipeline

An end-to-end prototype for transforming Hindi speech into structured, machine-actionable information through Automatic Speech Recognition (ASR), NLP-based question transformation, LLM-driven information extraction, deterministic semantic evidence routing, candidate-state aggregation, and classical machine learning.

---

## Overview

This project implements and evaluates an end-to-end pipeline for processing Hindi speech and transforming interview responses into structured candidate information and downstream matching predictions.

The system combines modern AI/ML techniques with deterministic validation and classical machine learning to address different stages of the pipeline.

The overall workflow is:

```text
Hindi Speech
     │
     ▼
┌─────────────────────────────┐
│ Hindi Speech Recognition    │
│ IndicConformer 600M         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Question Transformation     │
│ & Normalization             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Structured Information      │
│ Extraction                  │
│ Gemma 3 1B IT + QLoRA       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Semantic Evidence Router    │
│ Deterministic Evidence      │
│ Handling                    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Candidate State             │
│ Aggregation                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Matching V6                 │
│ TF-IDF + LinearSVC          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Independent Evaluation      │
│ & Validation                │
└─────────────────────────────┘
