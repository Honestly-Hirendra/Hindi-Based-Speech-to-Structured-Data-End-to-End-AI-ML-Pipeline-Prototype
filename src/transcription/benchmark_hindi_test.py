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

MODEL_NAME = (
    "ai4bharat/indic-conformer-600m-multilingual"
)

TARGET_SR = 16000
DECODER = "ctc"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEST_CSV = os.path.join(
    PROJECT_DIR,
    "evaluation",
    "all_splits",
    "test_results.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "evaluation",
)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "evaluation",
    "Results",
    "hindi_test_553_results.csv"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "evaluation",
    "Summary",
    "hindi_test_553_summary.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("INDICCONFORMER FP16 - HINDI TEST SET EVALUATION")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if DEVICE == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Test manifest: {TEST_CSV}"
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

        current = [i] + [
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

    distance = levenshtein(
        ref_words,
        hyp_words
    )

    return (
        distance
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

    distance = levenshtein(
        list(reference),
        list(hypothesis)
    )

    return (
        distance
        / len(reference)
    )


# ============================================================
# LOAD TEST MANIFEST
# ============================================================

if not os.path.exists(
    TEST_CSV
):

    raise FileNotFoundError(
        f"Test manifest not found:\n"
        f"{TEST_CSV}"
    )


with open(
    TEST_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    test_rows = list(
        csv.DictReader(f)
    )


print(
    f"Test samples: {len(test_rows)}"
)

print()


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(
    path
):

    audio, sample_rate = sf.read(
        path,
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

    waveform = waveform.unsqueeze(
        0
    )

    return waveform.to(
        DEVICE
    )


# ============================================================
# TRANSCRIPTION
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
                    "hi",
                    DECODER
                )

    else:

        with torch.inference_mode():

            prediction = model(
                waveform,
                "hi",
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
# GPU WARM-UP
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

            try:

                model(
                    dummy,
                    "hi",
                    DECODER
                )

            except Exception as e:

                print(
                    "Warm-up warning:",
                    e
                )

    torch.cuda.synchronize()

    print(
        "GPU warm-up complete."
    )

    print()


# ============================================================
# EVALUATION
# ============================================================

results = []

failed = 0
missing_audio = 0
missing_reference = 0

total_inference_time = 0.0
total_audio_duration = 0.0

benchmark_start = time.perf_counter()


print("=" * 70)
print("STARTING TEST SET EVALUATION")
print("=" * 70)
print()


for index, row in enumerate(
    test_rows,
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


    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

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
            f"[{index}/{len(test_rows)}] "
            f"MISSING AUDIO: "
            f"{audio_path}"
        )

        continue


    if not reference.strip():

        missing_reference += 1

        print(
            f"[{index}/{len(test_rows)}] "
            "MISSING REFERENCE"
        )

        continue


    # --------------------------------------------------------
    # Audio duration
    # --------------------------------------------------------

    try:

        audio_info = sf.info(
            audio_path
        )

        duration = (
            audio_info.frames
            / audio_info.samplerate
        )

    except Exception:

        duration = 0.0


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

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
            f"FAILED [{index}/{len(test_rows)}]"
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


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    wer = calculate_wer(
        reference,
        prediction
    )

    cer = calculate_cer(
        reference,
        prediction
    )


    results.append({

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


    total_inference_time += (
        inference_time
    )

    total_audio_duration += (
        duration
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index == 1
        or index % 50 == 0
        or index == len(test_rows)
    ):

        print(
            f"{index}/{len(test_rows)} processed | "
            f"WER: {wer * 100:.2f}% | "
            f"CER: {cer * 100:.2f}%"
        )


# ============================================================
# FINAL METRICS
# ============================================================

processing_time = (
    time.perf_counter()
    - benchmark_start
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


# ============================================================
# GPU MEMORY
# ============================================================

if DEVICE == "cuda":

    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

else:

    peak_gpu_memory_gb = 0.0


# ============================================================
# SAVE RESULTS CSV
# ============================================================

fields = [

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
    RESULTS_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

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

    "decoder":
        DECODER,

    "test_samples":
        len(test_rows),

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
        processing_time,

    "peak_gpu_memory_gb":
        peak_gpu_memory_gb
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("=" * 70)
print("HINDI TEST SET EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Test samples         : "
    f"{len(test_rows)}"
)

print(
    f"Samples evaluated    : "
    f"{successful}"
)

print(
    f"Failed               : "
    f"{failed}"
)

print(
    f"Missing audio        : "
    f"{missing_audio}"
)

print(
    f"Missing reference    : "
    f"{missing_reference}"
)

if average_wer is not None:

    print()

    print(
        f"Average WER         : "
        f"{average_wer * 100:.2f}%"
    )

    print(
        f"Average CER         : "
        f"{average_cer * 100:.2f}%"
    )

    print(
        f"Avg inference       : "
        f"{average_inference:.4f} sec/sample"
    )

    print(
        f"Avg audio           : "
        f"{average_audio:.4f} sec/sample"
    )

    print(
        f"Real-time factor    : "
        f"{real_time_factor:.4f}"
    )


print()

print(
    f"Total audio         : "
    f"{total_audio_duration:.2f} sec"
)

print(
    f"Total inference     : "
    f"{total_inference_time:.2f} sec"
)

print(
    f"Processing time     : "
    f"{processing_time:.2f} sec"
)

print(
    f"Peak GPU memory     : "
    f"{peak_gpu_memory_gb:.2f} GB"
)

print()

print(
    "Results CSV:"
)

print(
    RESULTS_CSV
)

print()

print(
    "Summary JSON:"
)

print(
    SUMMARY_JSON
)

print("=" * 70)