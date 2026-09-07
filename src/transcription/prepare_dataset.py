import csv
import json
import os
import random
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

PROJECT_DIR = r"F:\BlueHyre Project Final\01_Transcription"

METADATA_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "indicvoices_hindi",
    "metadata.json"
)

AUDIO_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "indicvoices_hindi",
    "audio"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "splits",
    "hindi"
)


# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not os.path.exists(
        METADATA_FILE
    ):

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "metadata.json must contain a list."
        )

    return data


# ============================================================
# SPEAKER ID
# ============================================================

def get_speaker_id(record):

    speaker = record.get(
        "speaker_id"
    )

    if speaker is None:

        raise ValueError(
            "Record has no speaker_id:\n"
            f"{record}"
        )

    return str(
        speaker
    )


# ============================================================
# AUDIO PATH
# ============================================================

def get_audio_path(record):

    filename = record.get(
        "filename"
    )

    if not filename:

        raise ValueError(
            "Record has no filename."
        )

    return os.path.join(
        AUDIO_DIR,
        filename
    )


# ============================================================
# VALIDATE RECORDS
# ============================================================

def validate_records(
    records
):

    valid = []

    missing_audio = 0

    missing_text = 0

    duplicate_filenames = []

    duplicate_seen = set()


    for record in records:

        filename = record.get(
            "filename"
        )

        if not filename:

            continue


        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        if filename in duplicate_seen:

            duplicate_filenames.append(
                filename
            )

            continue

        duplicate_seen.add(
            filename
        )


        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        audio_path = get_audio_path(
            record
        )

        if not os.path.exists(
            audio_path
        ):

            missing_audio += 1

            continue


        # ----------------------------------------------------
        # Transcript
        # ----------------------------------------------------

        text = record.get(
            "normalized"
        )

        if not text:

            text = record.get(
                "text"
            )

        if not text:

            text = record.get(
                "verbatim"
            )

        if not text:

            missing_text += 1

            continue


        valid.append(
            record
        )


    return (
        valid,
        missing_audio,
        missing_text,
        duplicate_filenames
    )


# ============================================================
# SPEAKER-INDEPENDENT SPLIT
# ============================================================

def create_speaker_split(
    records
):

    speakers = defaultdict(
        list
    )


    # Group recordings by speaker
    for record in records:

        speaker = get_speaker_id(
            record
        )

        speakers[
            speaker
        ].append(
            record
        )


    speaker_items = list(
        speakers.items()
    )


    # Shuffle speaker order so the split is reproducible
    random.shuffle(
        speaker_items
    )


    total_samples = len(
        records
    )


    target_train = (
        total_samples
        * TRAIN_RATIO
    )

    target_val = (
        total_samples
        * VAL_RATIO
    )

    target_test = (
        total_samples
        * TEST_RATIO
    )


    targets = {

        "train":
            target_train,

        "validation":
            target_val,

        "test":
            target_test
    }


    splits = {

        "train": [],

        "validation": [],

        "test": []
    }


    counts = {

        "train": 0,

        "validation": 0,

        "test": 0
    }


    # --------------------------------------------------------
    # Assign complete speakers
    # --------------------------------------------------------

    # Largest speakers first helps make the resulting
    # sample counts reasonably close to 80/10/10.
    speaker_items.sort(
        key=lambda item: len(
            item[1]
        ),
        reverse=True
    )


    for speaker, speaker_records in speaker_items:

        # Choose the split currently furthest below
        # its desired proportion.
        best_split = min(

            counts.keys(),

            key=lambda split:
                counts[split]
                / targets[split]
                if targets[split] > 0
                else float("inf")
        )


        splits[
            best_split
        ].extend(
            speaker_records
        )

        counts[
            best_split
        ] += len(
            speaker_records
        )


    return (
        splits,
        counts,
        speakers
    )


# ============================================================
# SPEAKER LEAKAGE CHECK
# ============================================================

def check_speaker_leakage(
    splits
):

    speaker_sets = {}

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        speaker_sets[
            split_name
        ] = {
            get_speaker_id(record)

            for record
            in splits[split_name]
        }


    train = speaker_sets[
        "train"
    ]

    validation = speaker_sets[
        "validation"
    ]

    test = speaker_sets[
        "test"
    ]


    overlap = {

        "train_validation":
            train & validation,

        "train_test":
            train & test,

        "validation_test":
            validation & test
    }


    leakage = any(
        overlap.values()
    )


    if leakage:

        raise RuntimeError(
            "SPEAKER LEAKAGE DETECTED:\n"
            f"{overlap}"
        )


    return speaker_sets


# ============================================================
# WRITE MANIFEST
# ============================================================

def write_manifest(
    records,
    output_path,
    split_name
):

    fields = [

        "audio",

        "text",

        "language",

        "speaker_id",

        "split",

        "duration",

        "gender",

        "age_group",

        "area",

        "district",

        "state",

        "occupation"
    ]


    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()


        for record in records:

            audio_path = get_audio_path(
                record
            )


            text = (

                record.get(
                    "normalized"
                )

                or record.get(
                    "text"
                )

                or record.get(
                    "verbatim"
                )

                or ""
            )


            writer.writerow({

                "audio":
                    audio_path,

                "text":
                    text,

                "language":
                    "hi",

                "speaker_id":
                    get_speaker_id(
                        record
                    ),

                "split":
                    split_name,

                "duration":
                    record.get(
                        "duration",
                        ""
                    ),

                "gender":
                    record.get(
                        "gender",
                        ""
                    ),

                "age_group":
                    record.get(
                        "age_group",
                        ""
                    ),

                "area":
                    record.get(
                        "area",
                        ""
                    ),

                "district":
                    record.get(
                        "district",
                        ""
                    ),

                "state":
                    record.get(
                        "state",
                        ""
                    ),

                "occupation":
                    record.get(
                        "occupation",
                        ""
                    )
            })


# ============================================================
# DURATION
# ============================================================

def total_duration(
    records
):

    total = 0.0

    for record in records:

        try:

            total += float(
                record.get(
                    "duration",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            pass

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HINDI DATASET PREPARATION")
    print("=" * 70)

    print(
        f"Metadata : {METADATA_FILE}"
    )

    print(
        f"Audio    : {AUDIO_DIR}"
    )

    print(
        f"Output   : {OUTPUT_DIR}"
    )

    print()


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_metadata()

    print(
        f"Metadata records: "
        f"{len(records)}"
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    (
        valid_records,
        missing_audio,
        missing_text,
        duplicates
    ) = validate_records(
        records
    )


    print(
        f"Valid records   : "
        f"{len(valid_records)}"
    )

    print(
        f"Missing audio   : "
        f"{missing_audio}"
    )

    print(
        f"Missing text    : "
        f"{missing_text}"
    )

    print(
        f"Duplicate files : "
        f"{len(duplicates)}"
    )

    print()


    if not valid_records:

        raise RuntimeError(
            "No valid records available."
        )


    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    total_seconds = total_duration(
        valid_records
    )

    print(
        f"Total audio     : "
        f"{total_seconds / 3600:.2f} hours"
    )

    print()


    # --------------------------------------------------------
    # Speakers
    # --------------------------------------------------------

    speaker_groups = defaultdict(
        list
    )

    for record in valid_records:

        speaker_groups[
            get_speaker_id(record)
        ].append(
            record
        )


    print(
        f"Unique speakers : "
        f"{len(speaker_groups)}"
    )

    print()


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        splits,
        counts,
        speakers
    ) = create_speaker_split(
        valid_records
    )


    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    speaker_sets = check_speaker_leakage(
        splits
    )


    print(
        "Speaker leakage: NONE"
    )

    print()


    # --------------------------------------------------------
    # Report split sizes
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        records_in_split = splits[
            split_name
        ]

        split_duration = total_duration(
            records_in_split
        )

        print(
            f"{split_name:12s}: "
            f"{len(records_in_split):4d} samples | "
            f"{len(speaker_sets[split_name]):3d} speakers | "
            f"{split_duration / 3600:.2f} hours"
        )


    print()


    # --------------------------------------------------------
    # Write manifests
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{split_name}.csv"
        )


        write_manifest(
            splits[split_name],
            output_path,
            split_name
        )


        print(
            "Created:",
            output_path
        )


    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {

        "dataset":
            "ai4bharat/IndicVoices",

        "config":
            "hindi",

        "split_source":
            "valid",

        "total_samples":
            len(valid_records),

        "total_speakers":
            len(speakers),

        "total_audio_hours":
            total_seconds / 3600,

        "train": {

            "samples":
                len(
                    splits["train"]
                ),

            "speakers":
                len(
                    speaker_sets["train"]
                ),

            "hours":
                total_duration(
                    splits["train"]
                ) / 3600
        },

        "validation": {

            "samples":
                len(
                    splits["validation"]
                ),

            "speakers":
                len(
                    speaker_sets["validation"]
                ),

            "hours":
                total_duration(
                    splits["validation"]
                ) / 3600
        },

        "test": {

            "samples":
                len(
                    splits["test"]
                ),

            "speakers":
                len(
                    speaker_sets["test"]
                ),

            "hours":
                total_duration(
                    splits["test"]
                ) / 3600
        },

        "speaker_leakage":
            False,

        "random_seed":
            RANDOM_SEED
    }


    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary.json"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )


    print()
    print(
        "Created:",
        summary_path
    )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HINDI DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Total samples : "
        f"{len(valid_records)}"
    )

    print(
        f"Total speakers: "
        f"{len(speakers)}"
    )

    print(
        f"Total audio   : "
        f"{total_seconds / 3600:.2f} hours"
    )

    print()
    print(
        "Speaker-independent train/validation/test "
        "manifests are ready."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()