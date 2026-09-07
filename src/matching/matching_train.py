"""
BLUEHYREAI - BLUEHYRE MATCHING V6
TF-IDF + LINEAR SVM
POTENTIAL_FIT FALSE-POSITIVE REDUCTION

V6 objective:
1. Keep the V5 TF-IDF + Linear SVM architecture.
2. Learn a decision-boundary margin on VALIDATION data.
3. Specifically reduce:
      NO_FIT -> POTENTIAL_FIT
      GOOD_FIT -> POTENTIAL_FIT
4. Preserve reasonable POTENTIAL_FIT recall and Macro F1.
5. NEVER use the test set to select the margin.
6. Final model is retrained on TRAIN + VALIDATION.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    r"F:\BlueHyre Project Final\04_Matching\evaluation"
    r"\job_resume_fit_labeled.csv"
)

OUTPUT_DIR = (
    r"F:\BlueHyre Project Final\04_Matching\evaluation"
    r"\model_output_v6"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "bluehyre_final_v6"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "bluehyre_final_v6.joblib"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "test_predictions_v6.csv"
)

SEARCH_FILE = os.path.join(
    OUTPUT_DIR,
    "v6_validation_search.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "v6_model_metrics.json"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "v6_training_report.txt"
)

CM_FILE = os.path.join(
    OUTPUT_DIR,
    "v6_confusion_matrix.png"
)


RANDOM_STATE = 42

LABELS = [
    "NO_FIT",
    "POTENTIAL_FIT",
    "GOOD_FIT",
]


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BLUEHYREAI - BLUEHYRE MATCHING V6")
print("TF-IDF + LINEAR SVM")
print("POTENTIAL_FIT FALSE-POSITIVE REDUCTION")
print("=" * 70)

print()
print("INPUT:")
print(INPUT_FILE)

print()
print("OUTPUT:")
print(OUTPUT_DIR)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

required_columns = [
    "resume_text",
    "job_text",
    "fit_label",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df["resume_text"] = (
    df["resume_text"]
    .fillna("")
    .astype(str)
)

df["job_text"] = (
    df["job_text"]
    .fillna("")
    .astype(str)
)

df["fit_label"] = (
    df["fit_label"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df = df[
    df["fit_label"].isin(LABELS)
].reset_index(drop=True)


# ============================================================
# IMPORTANT:
# EXCLUDE SOURCE MATCHING FIELDS
# ============================================================

excluded_fields = [
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score",
    "ai_matched_skills",
    "resume_skill_list",
    "job_required_skills",
]


print()
print("EXCLUDED SOURCE / MATCHING FIELDS")

for col in excluded_fields:
    if col in df.columns:
        print("EXCLUDED:", col)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(
    df["fit_label"]
    .value_counts()
    .reindex(LABELS)
)


# ============================================================
# COMBINED TEXT
# ============================================================

df["combined_text"] = (
    "RESUME "
    + df["resume_text"]
    + " JOB "
    + df["job_text"]
)


X_text = df["combined_text"]
y = df["fit_label"]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

X_train_text, X_temp_text, y_train, y_temp, idx_train, idx_temp = (
    train_test_split(
        X_text,
        y,
        df.index,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
)

X_val_text, X_test_text, y_val, y_test, idx_val, idx_test = (
    train_test_split(
        X_temp_text,
        y_temp,
        idx_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
)

print()
print("Training   :", len(X_train_text))
print("Validation :", len(X_val_text))
print("Test       :", len(X_test_text))


# ============================================================
# TF-IDF
# ============================================================

print()
print("=" * 70)
print("BUILDING TF-IDF FEATURES")
print("=" * 70)


print()
print("Building WORD TF-IDF...")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=150000,
)


X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_val_word = word_vectorizer.transform(
    X_val_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)


print(
    "Word features:",
    X_train_word.shape[1]
)


print()
print("Building CHARACTER TF-IDF...")


char_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.98,
    max_features=150000,
)


X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_val_char = char_vectorizer.transform(
    X_val_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)


print(
    "Char features:",
    X_train_char.shape[1]
)


X_train = hstack(
    [X_train_word, X_train_char]
).tocsr()

X_val = hstack(
    [X_val_word, X_val_char]
).tocsr()

X_test = hstack(
    [X_test_word, X_test_char]
).tocsr()


print()
print("Total features:", X_train.shape[1])


# ============================================================
# BASELINE SVM
# ============================================================

print()
print("=" * 70)
print("TRAINING BASELINE LINEAR SVM")
print("=" * 70)


baseline_model = LinearSVC(
    C=0.25,
    class_weight=None,
    random_state=RANDOM_STATE,
    max_iter=10000,
)


baseline_model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION DECISION SCORES
# ============================================================

val_scores = baseline_model.decision_function(
    X_val
)

classes = list(
    baseline_model.classes_
)

class_to_idx = {
    c: i
    for i, c in enumerate(classes)
}


def standard_prediction(scores):
    """
    Normal LinearSVC prediction:
    highest decision score wins.
    """
    return np.array(
        [
            classes[i]
            for i in np.argmax(scores, axis=1)
        ]
    )


def v6_prediction(
    scores,
    potential_margin,
    good_bias=0.0,
    no_bias=0.0,
):
    """
    V6 decision rule.

    POTENTIAL_FIT is accepted only when its score
    is sufficiently competitive with the strongest
    non-potential class.

    Otherwise choose NO_FIT vs GOOD_FIT directly.
    """

    no_i = class_to_idx["NO_FIT"]
    pot_i = class_to_idx["POTENTIAL_FIT"]
    good_i = class_to_idx["GOOD_FIT"]

    predictions = []

    for row in scores:

        no_score = row[no_i] + no_bias
        pot_score = row[pot_i]
        good_score = row[good_i] + good_bias

        strongest_non_potential = max(
            no_score,
            good_score
        )

        # POTENTIAL must beat the strongest alternative
        # by the selected margin.
        if pot_score >= (
            strongest_non_potential
            + potential_margin
        ):
            pred = "POTENTIAL_FIT"

        else:
            if no_score >= good_score:
                pred = "NO_FIT"
            else:
                pred = "GOOD_FIT"

        predictions.append(pred)

    return np.array(predictions)


# ============================================================
# BASELINE VALIDATION METRICS
# ============================================================

baseline_val_pred = standard_prediction(
    val_scores
)

baseline_val_accuracy = accuracy_score(
    y_val,
    baseline_val_pred
)

baseline_val_macro_f1 = f1_score(
    y_val,
    baseline_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

baseline_val_macro_precision = precision_score(
    y_val,
    baseline_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

baseline_val_macro_recall = recall_score(
    y_val,
    baseline_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)


def false_potential_metrics(y_true, y_pred):

    false_no_to_potential = int(
        np.sum(
            (np.asarray(y_true) == "NO_FIT")
            & (np.asarray(y_pred) == "POTENTIAL_FIT")
        )
    )

    false_good_to_potential = int(
        np.sum(
            (np.asarray(y_true) == "GOOD_FIT")
            & (np.asarray(y_pred) == "POTENTIAL_FIT")
        )
    )

    total_false_potential = (
        false_no_to_potential
        + false_good_to_potential
    )

    potential_actual = int(
        np.sum(
            np.asarray(y_true) == "POTENTIAL_FIT"
        )
    )

    potential_correct = int(
        np.sum(
            (np.asarray(y_true) == "POTENTIAL_FIT")
            & (np.asarray(y_pred) == "POTENTIAL_FIT")
        )
    )

    potential_recall = (
        potential_correct / potential_actual
        if potential_actual > 0
        else 0.0
    )

    return {
        "no_to_potential": false_no_to_potential,
        "good_to_potential": false_good_to_potential,
        "total_false_potential": total_false_potential,
        "potential_recall": potential_recall,
    }


baseline_fp = false_potential_metrics(
    y_val,
    baseline_val_pred
)


print()
print("BASELINE VALIDATION")
print(
    "Accuracy       :",
    f"{baseline_val_accuracy:.4f}"
)
print(
    "Macro F1       :",
    f"{baseline_val_macro_f1:.4f}"
)
print(
    "Macro Precision:",
    f"{baseline_val_macro_precision:.4f}"
)
print(
    "Macro Recall   :",
    f"{baseline_val_macro_recall:.4f}"
)

print()
print(
    "Baseline false NO -> POT:",
    baseline_fp["no_to_potential"]
)

print(
    "Baseline false GOOD -> POT:",
    baseline_fp["good_to_potential"]
)

print(
    "Baseline total false POT:",
    baseline_fp["total_false_potential"]
)

print(
    "Baseline POT recall:",
    f"{baseline_fp['potential_recall']:.4f}"
)


# ============================================================
# V6 VALIDATION SEARCH
# ============================================================

print()
print("=" * 70)
print("V6 DECISION-BOUNDARY SEARCH")
print("=" * 70)

print()
print(
    "Goal: reduce false POTENTIAL_FIT"
)

print(
    "Validation set only. Test set remains untouched."
)


# Margin values.
#
# Larger margin means POTENTIAL_FIT has to be
# more convincing before being selected.

margin_values = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


search_results = []


for margin in margin_values:

    pred = v6_prediction(
        val_scores,
        potential_margin=margin,
    )

    acc = accuracy_score(
        y_val,
        pred
    )

    macro_f1 = f1_score(
        y_val,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_val,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_val,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    fp = false_potential_metrics(
        y_val,
        pred
    )

    search_results.append(
        {
            "potential_margin": margin,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "false_no_to_potential": fp["no_to_potential"],
            "false_good_to_potential": fp["good_to_potential"],
            "false_potential_total": fp[
                "total_false_potential"
            ],
            "potential_recall": fp[
                "potential_recall"
            ],
        }
    )


search_df = pd.DataFrame(
    search_results
)


# ============================================================
# V6 SELECTION RULE
# ============================================================

# We don't want to cheat by selecting a boundary that
# simply destroys POTENTIAL_FIT recall.
#
# Minimum acceptable potential recall.
MIN_POTENTIAL_RECALL = 0.60

# Maximum acceptable Macro F1 degradation
# versus the normal SVM validation prediction.
MAX_F1_DROP = 0.05


eligible = search_df[
    (
        search_df["potential_recall"]
        >= MIN_POTENTIAL_RECALL
    )
    &
    (
        search_df["macro_f1"]
        >= (
            baseline_val_macro_f1
            - MAX_F1_DROP
        )
    )
].copy()


if len(eligible) == 0:

    print()
    print(
        "WARNING: No boundary satisfied all constraints."
    )

    print(
        "Falling back to baseline margin = 0.00"
    )

    selected_margin = 0.0

else:

    # Primary objective:
    # minimize false POTENTIAL.
    #
    # Secondary objective:
    # maximize Macro F1.
    #
    # Tertiary objective:
    # maximize accuracy.

    eligible = eligible.sort_values(
        by=[
            "false_potential_total",
            "macro_f1",
            "accuracy",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    selected_margin = float(
        eligible.iloc[0][
            "potential_margin"
        ]
    )


search_df.to_csv(
    SEARCH_FILE,
    index=False
)


# ============================================================
# SELECTED VALIDATION RESULT
# ============================================================

selected_val_pred = v6_prediction(
    val_scores,
    potential_margin=selected_margin,
)

selected_val_accuracy = accuracy_score(
    y_val,
    selected_val_pred
)

selected_val_macro_f1 = f1_score(
    y_val,
    selected_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

selected_val_macro_precision = precision_score(
    y_val,
    selected_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

selected_val_macro_recall = recall_score(
    y_val,
    selected_val_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

selected_fp = false_potential_metrics(
    y_val,
    selected_val_pred
)


print()
print("=" * 70)
print("SELECTED V6 BOUNDARY")
print("=" * 70)

print()
print(
    "Potential margin:",
    f"{selected_margin:.4f}"
)

print()
print(
    "Validation Accuracy:",
    f"{selected_val_accuracy:.4f}"
)

print(
    "Validation Macro F1:",
    f"{selected_val_macro_f1:.4f}"
)

print(
    "Validation Macro Precision:",
    f"{selected_val_macro_precision:.4f}"
)

print(
    "Validation Macro Recall:",
    f"{selected_val_macro_recall:.4f}"
)

print()
print(
    "False NO -> POT:",
    selected_fp["no_to_potential"]
)

print(
    "False GOOD -> POT:",
    selected_fp["good_to_potential"]
)

print(
    "TOTAL FALSE POTENTIAL:",
    selected_fp["total_false_potential"]
)

print(
    "Potential Recall:",
    f"{selected_fp['potential_recall']:.4f}"
)


# ============================================================
# FINAL TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("FINAL V6 TRAINING")
print("=" * 70)

print()
print(
    "Retraining on TRAIN + VALIDATION."
)

print(
    "TEST remains completely untouched."
)


X_trainval_text = pd.concat(
    [
        X_train_text,
        X_val_text,
    ]
)

y_trainval = pd.concat(
    [
        y_train,
        y_val,
    ]
)


# ============================================================
# FINAL TF-IDF
# ============================================================

print()
print("Rebuilding final TF-IDF...")


final_word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=150000,
)


final_char_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.98,
    max_features=150000,
)


X_trainval_word = (
    final_word_vectorizer
    .fit_transform(X_trainval_text)
)

X_test_word_final = (
    final_word_vectorizer
    .transform(X_test_text)
)


X_trainval_char = (
    final_char_vectorizer
    .fit_transform(X_trainval_text)
)

X_test_char_final = (
    final_char_vectorizer
    .transform(X_test_text)
)


X_trainval = hstack(
    [
        X_trainval_word,
        X_trainval_char,
    ]
).tocsr()


X_test_final = hstack(
    [
        X_test_word_final,
        X_test_char_final,
    ]
).tocsr()


print(
    "Final word features:",
    X_trainval_word.shape[1]
)

print(
    "Final char features:",
    X_trainval_char.shape[1]
)

print(
    "Final total features:",
    X_trainval.shape[1]
)


# ============================================================
# FINAL SVM
# ============================================================

print()
print("Training final Linear SVM...")


final_model = LinearSVC(
    C=0.25,
    class_weight=None,
    random_state=RANDOM_STATE,
    max_iter=10000,
)


final_model.fit(
    X_trainval,
    y_trainval
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_scores = final_model.decision_function(
    X_test_final
)


# V6 prediction using the margin learned
# exclusively from validation.
test_pred = v6_prediction(
    test_scores,
    potential_margin=selected_margin,
)


test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

test_macro_precision = precision_score(
    y_test,
    test_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

test_macro_recall = recall_score(
    y_test,
    test_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)


test_fp = false_potential_metrics(
    y_test,
    test_pred
)


print()
print(
    "Test Accuracy       :",
    f"{test_accuracy:.4f}"
)

print(
    "Test Macro F1       :",
    f"{test_macro_f1:.4f}"
)

print(
    "Test Macro Precision:",
    f"{test_macro_precision:.4f}"
)

print(
    "Test Macro Recall   :",
    f"{test_macro_recall:.4f}"
)


print()
print("FALSE POTENTIAL")

print(
    "NO_FIT -> POTENTIAL_FIT:",
    test_fp["no_to_potential"]
)

print(
    "GOOD_FIT -> POTENTIAL_FIT:",
    test_fp["good_to_potential"]
)

print(
    "TOTAL FALSE POTENTIAL:",
    test_fp["total_false_potential"]
)

print(
    "Potential Recall:",
    f"{test_fp['potential_recall']:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_pred,
    labels=LABELS,
    digits=4,
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=LABELS,
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(
    pd.DataFrame(
        cm,
        index=[
            "Actual_NO_FIT",
            "Actual_POTENTIAL_FIT",
            "Actual_GOOD_FIT",
        ],
        columns=[
            "Pred_NO_FIT",
            "Pred_POTENTIAL_FIT",
            "Pred_GOOD_FIT",
        ],
    )
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title(
    "BlueHyreAI V6 Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(LABELS)),
    LABELS,
    rotation=20,
)

plt.yticks(
    range(len(LABELS)),
    LABELS,
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)


for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )


plt.tight_layout()

plt.savefig(
    CM_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_output = df.loc[
    idx_test
].copy()


test_output[
    "predicted_fit_label"
] = test_pred


test_output[
    "v6_potential_margin"
] = selected_margin


test_output.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib


model_package = {
    "model": final_model,
    "word_vectorizer": final_word_vectorizer,
    "char_vectorizer": final_char_vectorizer,
    "potential_margin": selected_margin,
    "labels": LABELS,
    "model_type": "TF-IDF + LinearSVC",
    "version": "V6",
}


joblib.dump(
    model_package,
    MODEL_FILE
)


# ============================================================
# METRICS
# ============================================================

metrics = {
    "version": "V6",

    "model_type": "TF-IDF + LinearSVC",

    "input_file": INPUT_FILE,

    "rows": int(len(df)),

    "train_rows": int(
        X_train.shape[0]
    ),

    "validation_rows": int(
        X_val.shape[0]
    ),

    "test_rows": int(
        X_test.shape[0]
    ),

    "word_features": int(
        X_trainval_word.shape[1]
    ),

    "char_features": int(
        X_trainval_char.shape[1]
    ),

    "total_features": int(
        X_trainval.shape[1]
    ),

    "C": 0.25,

    "selected_potential_margin": (
        selected_margin
    ),

    "validation": {
        "baseline_accuracy": (
            baseline_val_accuracy
        ),
        "baseline_macro_f1": (
            baseline_val_macro_f1
        ),
        "baseline_false_potential": (
            baseline_fp[
                "total_false_potential"
            ]
        ),
        "selected_accuracy": (
            selected_val_accuracy
        ),
        "selected_macro_f1": (
            selected_val_macro_f1
        ),
        "selected_false_potential": (
            selected_fp[
                "total_false_potential"
            ]
        ),
        "selected_no_to_potential": (
            selected_fp[
                "no_to_potential"
            ]
        ),
        "selected_good_to_potential": (
            selected_fp[
                "good_to_potential"
            ]
        ),
    },

    "test": {
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "macro_precision": (
            test_macro_precision
        ),
        "macro_recall": (
            test_macro_recall
        ),
        "false_potential": (
            test_fp[
                "total_false_potential"
            ]
        ),
        "no_to_potential": (
            test_fp[
                "no_to_potential"
            ]
        ),
        "good_to_potential": (
            test_fp[
                "good_to_potential"
            ]
        ),
        "potential_recall": (
            test_fp[
                "potential_recall"
            ]
        ),
    },

    "timestamp": datetime.now().isoformat(),
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# TRAINING REPORT
# ============================================================

false_potential_change = (
    test_fp["total_false_potential"]
    - baseline_fp["total_false_potential"]
)


report_text = f"""
======================================================================
BLUEHYREAI - V6 TRAINING REPORT
======================================================================

MODEL
----------------------------------------------------------------------
TF-IDF + Linear SVM
C = 0.25

OBJECTIVE
----------------------------------------------------------------------
Reduce false POTENTIAL_FIT predictions while retaining reasonable
POTENTIAL_FIT recall and overall classification quality.

INPUT
----------------------------------------------------------------------
{INPUT_FILE}

DATASET
----------------------------------------------------------------------
Total rows: {len(df)}

Train rows: {X_train.shape[0]}
Validation rows: {X_val.shape[0]}
Test rows: {X_test.shape[0]}

FEATURES
----------------------------------------------------------------------
Word features: {X_trainval_word.shape[1]}
Character features: {X_trainval_char.shape[1]}
Total features: {X_trainval.shape[1]}

DECISION BOUNDARY
----------------------------------------------------------------------
Selected POTENTIAL_FIT margin:
{selected_margin:.6f}

The margin was selected exclusively on the validation set.

VALIDATION
----------------------------------------------------------------------
Baseline accuracy:
{baseline_val_accuracy:.4f}

Baseline Macro F1:
{baseline_val_macro_f1:.4f}

Baseline false POTENTIAL:
{baseline_fp["total_false_potential"]}

V6 validation accuracy:
{selected_val_accuracy:.4f}

V6 validation Macro F1:
{selected_val_macro_f1:.4f}

V6 validation false POTENTIAL:
{selected_fp["total_false_potential"]}

V6 validation NO -> POTENTIAL:
{selected_fp["no_to_potential"]}

V6 validation GOOD -> POTENTIAL:
{selected_fp["good_to_potential"]}

V6 validation POTENTIAL recall:
{selected_fp["potential_recall"]:.4f}

FINAL TEST
----------------------------------------------------------------------
Accuracy:
{test_accuracy:.4f}

Macro F1:
{test_macro_f1:.4f}

Macro Precision:
{test_macro_precision:.4f}

Macro Recall:
{test_macro_recall:.4f}

FALSE POTENTIAL
----------------------------------------------------------------------
NO_FIT -> POTENTIAL_FIT:
{test_fp["no_to_potential"]}

GOOD_FIT -> POTENTIAL_FIT:
{test_fp["good_to_potential"]}

TOTAL FALSE POTENTIAL:
{test_fp["total_false_potential"]}

POTENTIAL_FIT RECALL:
{test_fp["potential_recall"]:.4f}

COMPARISON
----------------------------------------------------------------------
V5 reported test accuracy:
0.7322

V5 reported test Macro F1:
0.7233

V5 reported false POTENTIAL:
30

V6 test accuracy:
{test_accuracy:.4f}

V6 test Macro F1:
{test_macro_f1:.4f}

V6 false POTENTIAL:
{test_fp["total_false_potential"]}

False POTENTIAL change relative to V5:
{test_fp["total_false_potential"] - 30}

======================================================================
IMPORTANT METHODOLOGICAL NOTE
======================================================================

V6 does NOT use the original dataset's AI matching score,
skill-string score, fuzzy score, resume skill list, or job-required
skill list as model inputs.

The fit labels are the derived labels in the labeled training
dataset.

The V6 decision margin is selected using validation data only.

The final test set is not used for model or boundary selection.

======================================================================
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(report_text)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("V6 OUTPUTS")
print("=" * 70)

print()
print("MODEL SAVED:")
print(MODEL_FILE)

print()
print("PREDICTIONS SAVED:")
print(PREDICTIONS_FILE)

print()
print("VALIDATION SEARCH SAVED:")
print(SEARCH_FILE)

print()
print("METRICS SAVED:")
print(METRICS_FILE)

print()
print("REPORT SAVED:")
print(REPORT_FILE)

print()
print("CONFUSION MATRIX SAVED:")
print(CM_FILE)

print()
print("=" * 70)
print("BLUEHYREAI V6 TRAINING COMPLETE")
print("=" * 70)

print()
print("FINAL TEST ACCURACY:")
print(f"{test_accuracy:.4f}")

print()
print("FINAL TEST MACRO F1:")
print(f"{test_macro_f1:.4f}")

print()
print("FINAL FALSE POTENTIAL:")
print(test_fp["total_false_potential"])

print()
print("V6 POTENTIAL MARGIN:")
print(f"{selected_margin:.4f}")

print()
print("DONE")
print("=" * 70)