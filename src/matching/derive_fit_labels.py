"""
BlueHyre Relative Fit-Label Derivation
======================================

Purpose
-------
Derive GOOD_FIT / POTENTIAL_FIT / NO_FIT labels relatively
from the source dataset's existing ai_match_score.

IMPORTANT
---------
1. The original job_resume_fit.csv is never modified.
2. ai_match_score is used ONLY during label derivation.
3. ai_match_score is NOT retained in the final training dataset.
4. Other external matching outputs are also removed.
5. The derived labels are NOT independent human ground truth.

Source
------
F:\\ BlueHyre Project Final\\ 04_Matching\\ evaluation \\ "job_resume_fit_clean.csv"

Output
------
F:\\ BlueHyre Project Final\\ 04_Matching\\ evaluation \\
    job_resume_fit_labeled.csv
    label_derivation_report.txt

Final model-input fields
------------------------
ID
resume_text
job_text
category
job_required_skills
resume_skill_list
fit_label
"""


from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(r"F:\BlueHyre Project Final\04_Matching")

SOURCE_FILE = BASE_DIR / "evaluation" / "job_resume_fit_clean.csv"

OUTPUT_DIR = BASE_DIR / "evaluation"

OUTPUT_FILE = OUTPUT_DIR / "job_resume_fit_labeled.csv"

REPORT_FILE = OUTPUT_DIR / "Reports" / "label_derivation_report.txt"


# ============================================================
# SOURCE FIELDS
# ============================================================

REQUIRED_COLUMNS = [
    "ID",
    "resume_text",
    "job_text",
    "category",
    "job_required_skills",
    "resume_skill_list",
    "ai_match_score",
]


# ============================================================
# EXTERNAL FIELDS THAT MUST NOT SURVIVE
# ============================================================

EXTERNAL_FIELDS = [
    "ai_match_score",
    "ai_matched_skills",
    "skill_string_match_score",
    "fuzzy_match_score",
]


# ============================================================
# RELATIVE LABEL PARAMETERS
# ============================================================

# We derive labels from the relative distribution of the
# source score rather than claiming absolute ground truth.
#
# Top 20%    -> GOOD_FIT
# Middle 50% -> POTENTIAL_FIT
# Bottom 30% -> NO_FIT
#
# These proportions are configurable.

GOOD_FIT_QUANTILE = 0.80
NO_FIT_QUANTILE = 0.30


# ============================================================
# LABEL GENERATION
# ============================================================

def derive_relative_labels(scores):
    """
    Generate relative fit labels from score distribution.

    Top 20%:
        GOOD_FIT

    Middle 50%:
        POTENTIAL_FIT

    Bottom 30%:
        NO_FIT
    """

    no_fit_cutoff = scores.quantile(NO_FIT_QUANTILE)
    good_fit_cutoff = scores.quantile(GOOD_FIT_QUANTILE)

    labels = pd.Series(
        index=scores.index,
        dtype="object"
    )

    labels.loc[
        scores <= no_fit_cutoff
    ] = "NO_FIT"

    labels.loc[
        (scores > no_fit_cutoff)
        & (scores < good_fit_cutoff)
    ] = "POTENTIAL_FIT"

    labels.loc[
        scores >= good_fit_cutoff
    ] = "GOOD_FIT"

    return (
        labels,
        no_fit_cutoff,
        good_fit_cutoff
    )


# ============================================================
# REPORT WRITER
# ============================================================

def write_report(lines):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("BLUEHYRE RELATIVE FIT-LABEL DERIVATION")
    print("=" * 72)

    print()
    print("Source:")
    print(SOURCE_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not SOURCE_FILE.exists():

        print()
        print("ERROR: Source dataset not found.")
        print(SOURCE_FILE)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("Loading source dataset...")

    df = pd.read_csv(
        SOURCE_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # VALIDATE COLUMNS
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    # --------------------------------------------------------
    # NUMERIC SCORE
    # --------------------------------------------------------

    print()
    print("Reading source score...")

    scores = pd.to_numeric(
        df["ai_match_score"],
        errors="coerce"
    )

    valid_mask = scores.notna()

    invalid_score_count = int(
        (~valid_mask).sum()
    )

    if invalid_score_count:

        print(
            f"WARNING: {invalid_score_count} "
            "invalid score rows found."
        )

    # --------------------------------------------------------
    # REMOVE ROWS WITH NO SCORE
    # --------------------------------------------------------

    working = df.loc[
        valid_mask
    ].copy()

    working["_source_score"] = scores.loc[
        valid_mask
    ]

    # --------------------------------------------------------
    # REMOVE EMPTY RESUMES
    # --------------------------------------------------------

    before_resume_filter = len(
        working
    )

    working = working[
        working["resume_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    empty_resume_removed = (
        before_resume_filter
        - len(working)
    )

    # --------------------------------------------------------
    # DERIVE RELATIVE LABELS
    # --------------------------------------------------------

    print()
    print(
        "Deriving relative fit labels..."
    )

    (
        labels,
        no_fit_cutoff,
        good_fit_cutoff
    ) = derive_relative_labels(
        working["_source_score"]
    )

    working["fit_label"] = labels

    # --------------------------------------------------------
    # FINAL MODEL INPUT COLUMNS
    # --------------------------------------------------------

    final_columns = [
        "ID",
        "resume_text",
        "job_text",
        "category",
        "job_required_skills",
        "resume_skill_list",
        "fit_label",
    ]

    final_df = working[
        final_columns
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("Validating final dataset...")

    duplicate_rows = int(
        final_df.duplicated().sum()
    )

    duplicate_ids = int(
        final_df["ID"].duplicated().sum()
    )

    missing_values = int(
        final_df.isna().sum().sum()
    )

    forbidden_remaining = [
        field
        for field in EXTERNAL_FIELDS
        if field in final_df.columns
    ]

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    label_order = [
        "NO_FIT",
        "POTENTIAL_FIT",
        "GOOD_FIT",
    ]

    label_counts = (
        final_df["fit_label"]
        .value_counts()
        .reindex(
            label_order,
            fill_value=0
        )
    )

    label_percentages = (
        label_counts
        / len(final_df)
        * 100
    )

    # --------------------------------------------------------
    # CATEGORY × LABEL
    # --------------------------------------------------------

    category_label = pd.crosstab(
        final_df["category"],
        final_df["fit_label"]
    )

    for label in label_order:

        if label not in category_label.columns:

            category_label[label] = 0

    category_label = category_label[
        label_order
    ]

    # --------------------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # CREATE REPORT
    # --------------------------------------------------------

    report = []

    report.append("=" * 72)
    report.append(
        "BLUEHYRE RELATIVE FIT-LABEL DERIVATION REPORT"
    )
    report.append("=" * 72)

    report.append("")
    report.append("SOURCE")
    report.append(
        str(SOURCE_FILE)
    )

    report.append("")
    report.append("OUTPUT")
    report.append(
        str(OUTPUT_FILE)
    )

    report.append("")
    report.append("DATASET SIZE")

    report.append(
        f"Original rows          : {len(df):,}"
    )

    report.append(
        f"Rows with valid score  : {valid_mask.sum():,}"
    )

    report.append(
        f"Invalid-score rows     : {invalid_score_count:,}"
    )

    report.append(
        f"Empty-resume rows removed: "
        f"{empty_resume_removed:,}"
    )

    report.append(
        f"Final rows             : {len(final_df):,}"
    )

    report.append("")
    report.append("FINAL MODEL INPUT FIELDS")

    for field in final_columns:

        report.append(
            f"  {field}"
        )

    report.append("")
    report.append("REMOVED EXTERNAL FIELDS")

    for field in EXTERNAL_FIELDS:

        report.append(
            f"  {field}"
        )

    report.append("")
    report.append("RELATIVE LABEL METHODOLOGY")

    report.append(
        "Labels are derived from the distribution of "
        "the source ai_match_score."
    )

    report.append(
        "Bottom 30% of valid scores -> NO_FIT"
    )

    report.append(
        "Middle 50% of valid scores -> POTENTIAL_FIT"
    )

    report.append(
        "Top 20% of valid scores -> GOOD_FIT"
    )

    report.append("")
    report.append("DERIVED SCORE CUT-OFFS")

    report.append(
        f"NO_FIT upper cutoff       : "
        f"{no_fit_cutoff:.4f}"
    )

    report.append(
        f"GOOD_FIT lower cutoff     : "
        f"{good_fit_cutoff:.4f}"
    )

    report.append("")
    report.append("FIT LABEL DISTRIBUTION")

    for label in label_order:

        count = int(
            label_counts[label]
        )

        percentage = float(
            label_percentages[label]
        )

        report.append(
            f"{label:15s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

    report.append("")
    report.append("DATA QUALITY")

    report.append(
        f"Missing values      : "
        f"{missing_values:,}"
    )

    report.append(
        f"Duplicate rows      : "
        f"{duplicate_rows:,}"
    )

    report.append(
        f"Duplicate IDs       : "
        f"{duplicate_ids:,}"
    )

    report.append("")
    report.append("CATEGORY × FIT LABEL")

    report.append(
        category_label.to_string()
    )

    report.append("")
    report.append("METHODOLOGICAL NOTE")

    report.append(
        "The source ai_match_score was used only to "
        "derive relative fit labels."
    )

    report.append(
        "The score itself is excluded from the final "
        "model-input dataset."
    )

    report.append(
        "Other source AI-generated matching fields are "
        "also excluded."
    )

    report.append(
        "The derived fit labels are weak supervision labels "
        "and are not independently human-verified ground truth."
    )

    report.append(
        "The original source dataset remains unchanged."
    )

    # --------------------------------------------------------
    # WRITE REPORT
    # --------------------------------------------------------

    write_report(
        report
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("LABEL DERIVATION COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Final rows: {len(final_df):,}"
    )

    print()
    print("RELATIVE CUT-OFFS")

    print(
        f"NO_FIT       <= {no_fit_cutoff:.4f}"
    )

    print(
        f"POTENTIAL_FIT: between "
        f"{no_fit_cutoff:.4f} and "
        f"{good_fit_cutoff:.4f}"
    )

    print(
        f"GOOD_FIT     >= {good_fit_cutoff:.4f}"
    )

    print()
    print("LABEL DISTRIBUTION")

    for label in label_order:

        print(
            f"{label:15s}: "
            f"{int(label_counts[label]):5d} "
            f"({label_percentages[label]:.2f}%)"
        )

    print()
    print("FINAL COLUMNS")

    for column in final_df.columns:

        print(
            f"  - {column}"
        )

    print()

    if forbidden_remaining:

        print(
            "ERROR: External fields remain:"
        )

        for field in forbidden_remaining:

            print(
                f"  - {field}"
            )

    else:

        print(
            "PASS: No external score/matching fields "
            "remain in the final dataset."
        )

    print()
    print("OUTPUT:")
    print(OUTPUT_FILE)

    print()
    print("REPORT:")
    print(REPORT_FILE)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()