"""
BlueHyre Candidate-Job Matching
Clean Dataset Creation

Purpose:
    Create a BlueHyre-controlled derivative of the downloaded
    job_resume_fit dataset.

Original dataset:
    F:\\ BlueHyre Project Final\\ 04_Matching\\ data\\ job_resume_fit.csv

Output:
    F:\\ BlueHyre Project Final\\ 04_Matching\\ evaluation
        job_resume_fit_clean.csv
        cleaning_report.txt

IMPORTANT:
    The original dataset is NOT modified.

Excluded external/derived fields:
    - ai_matched_skills
    - ai_match_score
    - skill_string_match_score
    - fuzzy_match_score

Retained evidence fields:
    - ID
    - resume_text
    - job_text
    - category
    - job_required_skills
    - resume_skill_list
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"F:\BlueHyre Project Final\04_Matching")

SOURCE_FILE = BASE_DIR / "data" / "job_resume_fit.csv"

OUTPUT_DIR = BASE_DIR / "evaluation"

CLEAN_FILE = OUTPUT_DIR / "job_resume_fit_clean.csv"

REPORT_FILE = OUTPUT_DIR / "Reports" / "cleaning_report.txt"


# ============================================================
# DATASET DEFINITION
# ============================================================

REQUIRED_COLUMNS = [
    "ID",
    "resume_text",
    "job_text",
    "category",
    "job_required_skills",
    "resume_skill_list",
]

EXCLUDED_COLUMNS = [
    "ai_matched_skills",
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score",
]


# ============================================================
# HELPERS
# ============================================================

def write_report(lines):
    """Write the cleaning report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def main():

    print("=" * 72)
    print("BLUEHYRE CLEAN MATCHING DATASET")
    print("=" * 72)

    print()
    print(f"Source : {SOURCE_FILE}")
    print(f"Output : {CLEAN_FILE}")
    print()

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not SOURCE_FILE.exists():
        print("ERROR: Source dataset not found.")
        print()
        print(f"Expected:")
        print(SOURCE_FILE)
        print()
        print("Make sure the file is named:")
        print("job_resume_fit.csv")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("Loading original dataset...")

    df = pd.read_csv(SOURCE_FILE)

    original_rows = len(df)
    original_columns = list(df.columns)

    print(f"Rows    : {original_rows:,}")
    print(f"Columns : {len(original_columns)}")

    print()
    print("Original columns:")
    for column in original_columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_required:

        print()
        print("ERROR: Required columns are missing:")

        for column in missing_required:
            print(f"  - {column}")

        print()
        print("Dataset was NOT modified.")

        return

    # --------------------------------------------------------
    # CHECK EXTERNAL FIELDS
    # --------------------------------------------------------

    present_excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    print()
    print("External/derived fields detected:")

    if present_excluded:
        for column in present_excluded:
            print(f"  - {column}")
    else:
        print("  None")

    # --------------------------------------------------------
    # CREATE CLEAN DATASET
    # --------------------------------------------------------

    clean_df = df[REQUIRED_COLUMNS].copy()

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    print()
    print("Validating clean dataset...")

    missing_counts = clean_df.isna().sum()

    total_missing = int(missing_counts.sum())

    duplicate_rows = int(
        clean_df.duplicated().sum()
    )

    duplicate_ids = int(
        clean_df["ID"].duplicated().sum()
    )

    empty_text_rows = (
        clean_df["resume_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_job_rows = (
        clean_df["job_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_required_skills = (
        clean_df["job_required_skills"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_candidate_skills = (
        clean_df["resume_skill_list"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    clean_df.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    category_counts = (
        clean_df["category"]
        .value_counts()
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = []

    report.append("=" * 72)
    report.append("BLUEHYRE CLEAN MATCHING DATASET REPORT")
    report.append("=" * 72)
    report.append("")

    report.append("SOURCE")
    report.append(str(SOURCE_FILE))
    report.append("")

    report.append("OUTPUT")
    report.append(str(CLEAN_FILE))
    report.append("")

    report.append("DATASET SIZE")
    report.append(f"Original rows : {original_rows:,}")
    report.append(f"Clean rows    : {len(clean_df):,}")
    report.append("")

    report.append("RETAINED FIELDS")
    for column in REQUIRED_COLUMNS:
        report.append(f"  {column}")
    report.append("")

    report.append("EXCLUDED EXTERNAL / DERIVED FIELDS")
    for column in EXCLUDED_COLUMNS:
        report.append(f"  {column}")
    report.append("")

    report.append("DATA QUALITY")
    report.append(f"Total missing values       : {total_missing:,}")
    report.append(f"Duplicate complete rows    : {duplicate_rows:,}")
    report.append(f"Duplicate IDs              : {duplicate_ids:,}")
    report.append(f"Empty resume text          : {empty_text_rows:,}")
    report.append(f"Empty job text             : {empty_job_rows:,}")
    report.append(f"Empty required skills      : {empty_required_skills:,}")
    report.append(f"Empty candidate skills     : {empty_candidate_skills:,}")
    report.append("")

    report.append("CATEGORY DISTRIBUTION")
    for category, count in category_counts.items():
        report.append(
            f"  {str(category):30s} {int(count):6d}"
        )

    report.append("")
    report.append("FINAL COLUMNS")
    for column in clean_df.columns:
        report.append(f"  {column}")

    report.append("")
    report.append("METHODOLOGY NOTE")
    report.append(
        "External AI-generated matching scores and matched-skill "
        "outputs were excluded from the BlueHyre matching dataset."
    )
    report.append(
        "The retained fields represent candidate/job evidence "
        "and job-category context."
    )
    report.append(
        "The original downloaded dataset remains unchanged."
    )

    write_report(report)

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BLUEHYRE CLEAN DATASET CREATED")
    print("=" * 72)

    print()
    print(f"Original rows : {original_rows:,}")
    print(f"Clean rows    : {len(clean_df):,}")

    print()
    print("Retained fields:")
    for column in REQUIRED_COLUMNS:
        print(f"  {column}")

    print()
    print("Excluded fields:")
    for column in EXCLUDED_COLUMNS:
        print(f"  {column}")

    print()
    print("DATA QUALITY")
    print(f"Missing values       : {total_missing:,}")
    print(f"Duplicate rows       : {duplicate_rows:,}")
    print(f"Duplicate IDs        : {duplicate_ids:,}")
    print(f"Empty resume text    : {empty_text_rows:,}")
    print(f"Empty job text       : {empty_job_rows:,}")
    print(f"Empty required skills: {empty_required_skills:,}")
    print(f"Empty candidate skills: {empty_candidate_skills:,}")

    print()
    print("OUTPUT FILE:")
    print(CLEAN_FILE)

    print()
    print("REPORT:")
    print(REPORT_FILE)

    print()
    print("=" * 72)
    print("CLEANING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()