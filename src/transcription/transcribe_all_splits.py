import os
import csv
import json
import time
import re

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"F:\BlueHyre Project Final\01_Transcription"

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

TARGET_SR = 16000
DECODER = "ctc"
LANGUAGE = "hi"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SPLIT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "splits",
    "hindi"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "evaluation",
    "all_splits"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SPLITS = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv"
}


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("INDICCONFORMER FP16 - ALL HINDI SPLITS")
print("=" * 75)

print(
    f"Device : {DEVICE}"
)

if DEVICE == "cuda":

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Model  : {MODEL_NAME}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading IndicConformer..."
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "Model loaded successfully."
)

print()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip().lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein(
    reference,
    hypothesis
):

    n = len(reference)
    m = len(hypothesis)

    previous = list(
        range(m + 1)
    )

    for i in range(
        1,
        n + 1
    ):

        current = [
            i
        ] + [
            0
        ] * m

        for j in range(
            1,
            m + 1
        ):

            if (
                reference[i - 1]
                == hypothesis[j - 1]
            ):

                current[j] = (
                    previous[j - 1]
                )

            else:

                current[j] = (
                    1
                    + min(
                        previous[j],
                        current[j - 1],
                        previous[j - 1]
                    )
                )

        previous = current

    return previous[m]


# ============================================================
# WER
# ============================================================

def calculate_wer(
    reference,
    hypothesis
):

    reference = normalize_text(
        reference
    )

    hypothesis = normalize_text(
        hypothesis
    )

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:

        return (
            0.0
            if not hyp_words
            else 1.0
        )

    return (
        levenshtein(
            ref_words,
            hyp_words
        )
        / len(ref_words)
    )


# ============================================================
# CER
# ============================================================

def calculate_cer(
    reference,
    hypothesis
):

    reference = (
        normalize_text(
            reference
        )
        .replace(" ", "")
    )

    hypothesis = (
        normalize_text(
            hypothesis
        )
        .replace(" ", "")
    )

    if not reference:

        return (
            0.0
            if not hypothesis
            else 1.0
        )

    return (
        levenshtein(
            list(reference),
            list(hypothesis)
        )
        / len(reference)
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(
    audio_path
):

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    # Stereo -> mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    waveform = torch.from_numpy(
        audio
    )

    # Resample to 16 kHz
    if sample_rate != TARGET_SR:

        new_length = int(
            len(waveform)
            * TARGET_SR
            / sample_rate
        )

        waveform = F.interpolate(
            waveform
            .unsqueeze(0)
            .unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False
        ).squeeze()

    return waveform.unsqueeze(
        0
    ).to(
        DEVICE
    )


# ============================================================
# TRANSCRIBE ONE FILE
# ============================================================

def transcribe(
    audio_path
):

    waveform = load_audio(
        audio_path
    )

    if DEVICE == "cuda":

        with torch.inference_mode():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                prediction = model(
                    waveform,
                    LANGUAGE,
                    DECODER
                )

    else:

        with torch.inference_mode():

            prediction = model(
                waveform,
                LANGUAGE,
                DECODER
            )

    if isinstance(
        prediction,
        (list, tuple)
    ):

        prediction = prediction[0]

    return str(
        prediction
    )


# ============================================================
# GPU WARMUP
# ============================================================

if DEVICE == "cuda":

    print(
        "Warming up GPU..."
    )

    dummy = torch.zeros(
        1,
        TARGET_SR,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            model(
                dummy,
                LANGUAGE,
                DECODER
            )

    torch.cuda.synchronize()

    print(
        "GPU warm-up complete."
    )

    print()


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    manifest_name
):

    manifest_path = os.path.join(
        SPLIT_DIR,
        manifest_name
    )

    if not os.path.exists(
        manifest_path
    ):

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{manifest_path}"
        )


    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )


    print()
    print("=" * 75)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 75)

    print(
        f"Samples: {len(rows)}"
    )

    print()


    results = []

    failed = 0
    missing_audio = 0
    missing_reference = 0

    total_inference_time = 0.0
    total_audio_duration = 0.0

    start_time = time.perf_counter()


    for index, row in enumerate(
        rows,
        start=1
    ):

        audio_path = row.get(
            "audio",
            ""
        )

        reference = row.get(
            "text",
            ""
        )


        # ----------------------------------------------------
        # Resolve audio
        # ----------------------------------------------------

        if not os.path.isabs(
            audio_path
        ):

            audio_path = os.path.join(
                PROJECT_DIR,
                audio_path
            )


        if not os.path.exists(
            audio_path
        ):

            missing_audio += 1

            print(
                f"[{index}/{len(rows)}] "
                "MISSING AUDIO"
            )

            continue


        if not reference.strip():

            missing_reference += 1

            print(
                f"[{index}/{len(rows)}] "
                "MISSING REFERENCE"
            )

            continue


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        try:

            info = sf.info(
                audio_path
            )

            duration = (
                info.frames
                / info.samplerate
            )

        except Exception:

            duration = 0.0


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        if DEVICE == "cuda":

            torch.cuda.synchronize()


        inference_start = (
            time.perf_counter()
        )


        try:

            prediction = transcribe(
                audio_path
            )

        except Exception as e:

            failed += 1

            print()
            print(
                f"FAILED [{index}/{len(rows)}]"
            )

            print(
                f"File: {audio_path}"
            )

            print(
                f"Error: {e}"
            )

            continue


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        inference_time = (
            time.perf_counter()
            - inference_start
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        wer = calculate_wer(
            reference,
            prediction
        )

        cer = calculate_cer(
            reference,
            prediction
        )


        total_inference_time += (
            inference_time
        )

        total_audio_duration += (
            duration
        )


        results.append({

            "split":
                split_name,

            "filename":
                row.get(
                    "audio",
                    os.path.basename(
                        audio_path
                    )
                ),

            "speaker_id":
                row.get(
                    "speaker_id",
                    ""
                ),

            "reference":
                reference,

            "prediction":
                prediction,

            "wer":
                wer,

            "cer":
                cer,

            "audio_duration":
                duration,

            "inference_time":
                inference_time,

            "real_time_factor":
                (
                    inference_time
                    / duration
                    if duration > 0
                    else None
                )
        })


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 100 == 0
            or index == len(rows)
        ):

            print(
                f"{index}/{len(rows)} | "
                f"Latest WER: {wer * 100:.2f}% | "
                f"Latest CER: {cer * 100:.2f}%"
            )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    processing_time = (
        time.perf_counter()
        - start_time
    )

    successful = len(
        results
    )


    if successful:

        average_wer = (
            sum(
                r["wer"]
                for r in results
            )
            / successful
        )

        average_cer = (
            sum(
                r["cer"]
                for r in results
            )
            / successful
        )

        average_inference = (
            total_inference_time
            / successful
        )

        average_audio = (
            total_audio_duration
            / successful
        )

        real_time_factor = (

            total_inference_time
            / total_audio_duration

            if total_audio_duration > 0

            else None
        )

    else:

        average_wer = None
        average_cer = None
        average_inference = None
        average_audio = None
        real_time_factor = None


    # ========================================================
    # SAVE SPLIT RESULTS
    # ========================================================

    results_csv = os.path.join(
        OUTPUT_DIR,
        "evaluation",
        "Results"
        f"{split_name}_results.csv"
    )

    fields = [

        "split",
        "filename",
        "speaker_id",
        "reference",
        "prediction",
        "wer",
        "cer",
        "audio_duration",
        "inference_time",
        "real_time_factor"
    ]


    with open(
        results_csv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)


    summary = {

        "split":
            split_name,

        "samples_in_manifest":
            len(rows),

        "samples_evaluated":
            successful,

        "failed":
            failed,

        "missing_audio":
            missing_audio,

        "missing_reference":
            missing_reference,

        "average_wer":
            average_wer,

        "average_cer":
            average_cer,

        "average_inference_seconds":
            average_inference,

        "average_audio_seconds":
            average_audio,

        "real_time_factor":
            real_time_factor,

        "total_audio_seconds":
            total_audio_duration,

        "total_inference_seconds":
            total_inference_time,

        "processing_time_seconds":
            processing_time
    }


    summary_json = os.path.join(
        OUTPUT_DIR,
        "evaluation",
        "Summary"
        f"{split_name}_summary.json"
    )


    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # SPLIT REPORT
    # ========================================================

    print()
    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        f"Samples evaluated : "
        f"{successful}"
    )

    print(
        f"Failed            : "
        f"{failed}"
    )

    print(
        f"Average WER       : "
        f"{average_wer * 100:.2f}%"
        if average_wer is not None
        else "Average WER       : N/A"
    )

    print(
        f"Average CER       : "
        f"{average_cer * 100:.2f}%"
        if average_cer is not None
        else "Average CER       : N/A"
    )

    print(
        f"Avg inference     : "
        f"{average_inference:.4f} sec/sample"
        if average_inference is not None
        else "Avg inference     : N/A"
    )

    print(
        f"Avg audio         : "
        f"{average_audio:.4f} sec/sample"
        if average_audio is not None
        else "Avg audio         : N/A"
    )

    print(
        f"Real-time factor  : "
        f"{real_time_factor:.4f}"
        if real_time_factor is not None
        else "Real-time factor  : N/A"
    )

    print()

    return summary


# ============================================================
# RUN ALL THREE
# ============================================================

all_summaries = {}

overall_start = time.perf_counter()


for split_name, manifest_name in SPLITS.items():

    all_summaries[
        split_name
    ] = process_split(
        split_name,
        manifest_name
    )


overall_processing_time = (
    time.perf_counter()
    - overall_start
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

combined_summary = {

    "model":
        MODEL_NAME,

    "device":
        DEVICE,

    "device_name":
        (
            torch.cuda.get_device_name(0)
            if DEVICE == "cuda"
            else "CPU"
        ),

    "precision":
        "FP16 AMP"
        if DEVICE == "cuda"
        else "FP32",

    "language":
        LANGUAGE,

    "decoder":
        DECODER,

    "splits":
        all_summaries,

    "total_processing_time_seconds":
        overall_processing_time
}


combined_path = os.path.join(
    OUTPUT_DIR,
    "evaluation",
    "Summary"
    "combined_summary.json"
)


with open(
    combined_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        combined_summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 75)
print("ALL HINDI SPLITS - FINAL SUMMARY")
print("=" * 75)

print(
    f"{'SPLIT':<12}"
    f"{'N':>8}"
    f"{'WER':>12}"
    f"{'CER':>12}"
    f"{'RTF':>12}"
)

print("-" * 75)


for split_name in [
    "train",
    "validation",
    "test"
]:

    s = all_summaries[
        split_name
    ]

    wer = s["average_wer"]
    cer = s["average_cer"]
    rtf = s["real_time_factor"]

    print(
        f"{split_name:<12}"
        f"{s['samples_evaluated']:>8}"
        f"{wer * 100:>11.2f}%"
        f"{cer * 100:>11.2f}%"
        f"{rtf:>12.4f}"
    )


print()
print(
    f"Overall processing time: "
    f"{overall_processing_time:.2f} seconds"
)

print()
print(
    "Results directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Combined summary:"
)

print(
    combined_path
)

print("=" * 75)