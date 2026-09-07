from pathlib import Path
import re
import math
import pandas as pd


# ============================================================
# BLUEHYREAI — FINAL HINDI TRANSCRIPTION ARTIFACTS
# ============================================================
#
# Hindi ONLY.
#
# Input:
#   F:\ITD\BlueHyreAI\evaluation\hindi_test_553_results.csv
#
# Outputs:
#   F:\ITD\BlueHyreAI\evaluation\transcription_results.csv
#   F:\ITD\BlueHyreAI\evaluation\transcription_evaluation.csv
#
# IMPORTANT:
# - Does NOT run ASR again.
# - Does NOT modify the benchmark.
# - Does NOT include Hinglish.
# - Uses the already completed Hindi 553-sample evaluation.
# ============================================================


PROJECT_DIR = Path(r"F:\BlueHyre Project Final\01_Transcription")
EVALUATION_DIR = PROJECT_DIR / "evaluation"

INPUT_FILE = EVALUATION_DIR / "Results" / "hindi_test_553_results.csv"

OUTPUT_RESULTS = EVALUATION_DIR/ "Results" / "transcription_results.csv"
OUTPUT_EVALUATION = EVALUATION_DIR/ "Results" / "transcription_evaluation.csv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Remove punctuation while preserving Devanagari,
    # Latin characters, digits and whitespace.
    text = re.sub(
        r"[^\w\u0900-\u097F\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def edit_distance(ref, hyp):

    n = len(ref)
    m = len(hyp)

    previous = list(range(m + 1))

    for i in range(1, n + 1):

        current = [i] + [0] * m

        for j in range(1, m + 1):

            if ref[i - 1] == hyp[j - 1]:

                current[j] = previous[j - 1]

            else:

                deletion = previous[j]
                insertion = current[j - 1]
                substitution = previous[j - 1]

                current[j] = 1 + min(
                    deletion,
                    insertion,
                    substitution,
                )

        previous = current

    return previous[m]


# ============================================================
# SAMPLE WER / CER
# ============================================================

def calculate_sample_wer(reference, prediction):

    ref = normalize_text(reference).split()
    hyp = normalize_text(prediction).split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(ref, hyp) / len(ref)


def calculate_sample_cer(reference, prediction):

    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(prediction).replace(" ", "")

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(
        list(ref),
        list(hyp),
    ) / len(ref)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("BLUEHYREAI — FINAL HINDI TRANSCRIPTION ARTIFACTS")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nHindi evaluation file was not found:\n"
            f"{INPUT_FILE}\n"
        )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    print(f"Input file : {INPUT_FILE}")
    print(f"Samples    : {len(df):,}")
    print()

    required_columns = [
        "filename",
        "speaker_id",
        "reference",
        "prediction",
        "wer",
        "cer",
        "audio_duration",
        "inference_time",
        "real_time_factor",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # ========================================================
    # 1. PER-SAMPLE TRANSCRIPTION RESULTS
    # ========================================================

    results = pd.DataFrame()

    results["filename"] = df["filename"]

    results["speaker_id"] = df["speaker_id"]

    results["reference"] = df["reference"]

    results["prediction"] = df["prediction"]

    results["reference_normalized"] = [
        normalize_text(x)
        for x in results["reference"]
    ]

    results["prediction_normalized"] = [
        normalize_text(x)
        for x in results["prediction"]
    ]

    results["reference_word_count"] = [
        len(x.split())
        for x in results["reference_normalized"]
    ]

    results["prediction_word_count"] = [
        len(x.split())
        for x in results["prediction_normalized"]
    ]

    results["word_count_difference"] = (
        results["prediction_word_count"]
        - results["reference_word_count"]
    )

    results["sample_wer"] = [
        calculate_sample_wer(
            reference,
            prediction,
        )
        for reference, prediction
        in zip(
            results["reference"],
            results["prediction"],
        )
    ]

    results["sample_cer"] = [
        calculate_sample_cer(
            reference,
            prediction,
        )
        for reference, prediction
        in zip(
            results["reference"],
            results["prediction"],
        )
    ]

    results["audio_duration_seconds"] = pd.to_numeric(
        df["audio_duration"],
        errors="coerce",
    )

    results["inference_time_seconds"] = pd.to_numeric(
        df["inference_time"],
        errors="coerce",
    )

    results["real_time_factor"] = pd.to_numeric(
        df["real_time_factor"],
        errors="coerce",
    )

    # Preserve the original benchmark values for traceability.
    results["benchmark_wer"] = pd.to_numeric(
        df["wer"],
        errors="coerce",
    )

    results["benchmark_cer"] = pd.to_numeric(
        df["cer"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # SAVE PER-SAMPLE RESULTS
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 2. FINAL HINDI CORPUS EVALUATION
    # ========================================================
    #
    # These are the finalized corpus-level Hindi benchmark
    # figures already established from the evaluation.
    # ========================================================

    SAMPLE_COUNT = 553

    REFERENCE_WORDS = 6978

    SUBSTITUTIONS = 968

    DELETIONS = 201

    INSERTIONS = 143

    TOTAL_ERRORS = (
        SUBSTITUTIONS
        + DELETIONS
        + INSERTIONS
    )

    RAW_WER = (
        TOTAL_ERRORS
        / REFERENCE_WORDS
    )

    RAW_CER = 0.097966

    # Additional finalized diagnostic values.
    ORTHOGRAPHIC_WER = 0.169676
    BOUNDARY_WER = 0.173636
    FULLY_NORMALIZED_WER = 0.161460
    CLEAN_WER = 0.176152
    CLEAN_NORMALIZED_WER = 0.149096

    total_audio = results[
        "audio_duration_seconds"
    ].sum()

    total_inference = results[
        "inference_time_seconds"
    ].sum()

    mean_rtf = results[
        "real_time_factor"
    ].mean()

    median_rtf = results[
        "real_time_factor"
    ].median()

    audio_throughput = (
        total_audio / total_inference
        if total_inference > 0
        else math.nan
    )

    evaluation_rows = [

        {
            "metric": "samples",
            "value": SAMPLE_COUNT,
            "unit": "samples",
        },

        {
            "metric": "reference_words",
            "value": REFERENCE_WORDS,
            "unit": "words",
        },

        {
            "metric": "substitutions",
            "value": SUBSTITUTIONS,
            "unit": "errors",
        },

        {
            "metric": "deletions",
            "value": DELETIONS,
            "unit": "errors",
        },

        {
            "metric": "insertions",
            "value": INSERTIONS,
            "unit": "errors",
        },

        {
            "metric": "total_errors",
            "value": TOTAL_ERRORS,
            "unit": "errors",
        },

        {
            "metric": "raw_WER",
            "value": RAW_WER,
            "unit": "ratio",
        },

        {
            "metric": "raw_CER",
            "value": RAW_CER,
            "unit": "ratio",
        },

        {
            "metric": "orthographic_WER",
            "value": ORTHOGRAPHIC_WER,
            "unit": "ratio",
        },

        {
            "metric": "boundary_WER",
            "value": BOUNDARY_WER,
            "unit": "ratio",
        },

        {
            "metric": "fully_normalized_WER",
            "value": FULLY_NORMALIZED_WER,
            "unit": "ratio",
        },

        {
            "metric": "clean_WER",
            "value": CLEAN_WER,
            "unit": "ratio",
        },

        {
            "metric": "clean_normalized_WER",
            "value": CLEAN_NORMALIZED_WER,
            "unit": "ratio",
        },

        {
            "metric": "total_audio_seconds",
            "value": total_audio,
            "unit": "seconds",
        },

        {
            "metric": "total_inference_seconds",
            "value": total_inference,
            "unit": "seconds",
        },

        {
            "metric": "audio_throughput",
            "value": audio_throughput,
            "unit": "audio_seconds_per_inference_second",
        },

        {
            "metric": "mean_RTF",
            "value": mean_rtf,
            "unit": "ratio",
        },

        {
            "metric": "median_RTF",
            "value": median_rtf,
            "unit": "ratio",
        },
    ]

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    evaluation.to_csv(
        OUTPUT_EVALUATION,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("=" * 72)
    print("FINAL HINDI RESULTS")
    print("=" * 72)
    print()

    print(f"Samples       : {SAMPLE_COUNT:,}")
    print(f"Ref words     : {REFERENCE_WORDS:,}")
    print(f"Raw WER       : {RAW_WER:.4%}")
    print(f"Raw CER       : {RAW_CER:.4%}")
    print()
    print(f"Substitutions : {SUBSTITUTIONS:,}")
    print(f"Deletions     : {DELETIONS:,}")
    print(f"Insertions    : {INSERTIONS:,}")
    print(f"Total errors  : {TOTAL_ERRORS:,}")
    print()
    print(f"Orthographic WER       : {ORTHOGRAPHIC_WER:.4%}")
    print(f"Boundary WER           : {BOUNDARY_WER:.4%}")
    print(f"Fully normalized WER   : {FULLY_NORMALIZED_WER:.4%}")
    print(f"Clean WER              : {CLEAN_WER:.4%}")
    print(f"Clean normalized WER   : {CLEAN_NORMALIZED_WER:.4%}")
    print()
 
    print("OUTPUT FILES")
    print("-" * 72)

    print(
        f"Per-sample results:\n"
        f"{OUTPUT_RESULTS}"
    )

    print()

    print(
        f"Corpus evaluation:\n"
        f"{OUTPUT_EVALUATION}"
    )

    print()
    print("=" * 72)
    print("DONE — HINDI ONLY")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()