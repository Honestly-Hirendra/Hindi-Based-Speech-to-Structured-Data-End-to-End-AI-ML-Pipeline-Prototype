import os
import json
import re
import gc
import time

import pandas as pd
import torch

from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel


# ============================================================
# BLUEHYREAI - GEMMA 3 1B RUN 5 EVALUATION
# ============================================================
#
# MODEL:
#   Gemma 3 1B + Run 5 QLoRA
#
# DATASET:
#   llama_extraction_v5_1
#
# TEST SET:
#   llama_extraction_v5_1/test.jsonl
#
# TEST EXAMPLES:
#   888
#
# NO TRAINING
# NO API
# LOCAL ONLY
# RESUMABLE
#
# Predictions are saved after every batch.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"F:\BlueHyre Project Final\03_Extraction"

TEST_FILE = os.path.join(
    BASE_DIR,
    "Results",
    "01_extraction_v5_1",
    "test.jsonl"
)

BASE_MODEL = "google/gemma-3-1b-it"

RUN5_ADAPTER = os.path.join(
    BASE_DIR,
    "Results",
    "02_gemma3_1b_qlora_run5",
    "final_adapter"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "03_run_5"
)

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "run5_predictions.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "run5_metrics.json"
)

FIELD_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "run5_field_metrics.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_SEQ_LENGTH = 512

MAX_NEW_TOKENS = 256

# Safe setting for your RTX 4060 Laptop 8 GB.
BATCH_SIZE = 10



# ============================================================
# FINAL HIERARCHICAL SCHEMA
# ============================================================

TOP_LEVEL_FIELDS = {
    "profile",
    "behavioral",
    "availability",
    "motivation",
    "role_fit",
    "other_explicit_facts"
}

PROFILE_FIELDS = {
    "experience",
    "previous_responsibilities",
    "skills",
    "education_certifications"
}

BEHAVIORAL_FIELDS = {
    "problem_solving",
    "teamwork",
    "communication",
    "conflict_resolution",
    "prioritization"
}

AVAILABILITY_FIELDS = {
    "availability",
    "notice_period",
    "salary_expectations"
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
आप BlueHyreAI के उम्मीदवार सूचना निष्कर्षण मॉडल हैं।

इंटरव्यू प्रश्न और उम्मीदवार के उत्तर से केवल वही जानकारी
निकालें जो उत्तर में स्पष्ट रूप से दी गई है।

सभी स्पष्ट तथ्यों को सुरक्षित रखें। अनुमान या unsupported
information न जोड़ें।

Hierarchical schema का उपयोग करें।

BEHAVIORAL FIELD BOUNDARIES:

problem_solving:
समस्या की पहचान, investigation, diagnosis, troubleshooting,
fix, resolution, mitigation, recovery, implementation या
solution.

teamwork:
टीम के साथ काम करना, collaboration, coordination, delegation,
task assignment/division, leadership, coaching, mentoring,
team management.

communication:
inform करना, update देना, report करना, explain करना,
notify करना, feedback देना, stakeholder/client communication,
status/risk communicate करना।

conflict_resolution:
disagreement, dispute, negotiation, mediation, compromise,
persuasion या interpersonal conflict resolve करना।

prioritization:
किसे पहले करना है, priority order, urgency, deferment,
trade-off, selection या resource priority तय करना।

महत्वपूर्ण:
यदि communication का action किसी problem scenario में है,
तो उसे केवल problem_solving में न डालें।
यदि team coordination किसी problem के दौरान है, तो केवल
problem_solving में न डालें।
यदि कोई action यह तय करता है कि क्या पहले होगा, तो
prioritization का उपयोग करें।
यदि कोई action disagreement/negotiation से संबंधित है,
तो conflict_resolution का उपयोग करें।

एक उत्तर में अलग-अलग semantic facts हों तो उन्हें अलग-अलग
fields में रखें।

केवल वैध hierarchical JSON object लौटाएं।
"""


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def is_empty(value):

    if value is None:
        return True

    if value == "":
        return True

    if isinstance(value, list):
        return len(value) == 0

    if isinstance(value, dict):
        return len(value) == 0

    return False


def normalize_text(text):

    text = clean(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def extract_json(text):

    text = clean(text)

    if not text:
        return None


    # Direct JSON
    try:

        obj = json.loads(
            text
        )

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass


    # Markdown fenced JSON
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    ).strip()


    try:

        obj = json.loads(
            cleaned
        )

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass


    # Balanced JSON object extraction
    start = text.find("{")

    if start == -1:
        return None


    depth = 0
    in_string = False
    escaped = False


    for i in range(
        start,
        len(text)
    ):

        char = text[i]


        if escaped:

            escaped = False
            continue


        if char == "\\":

            escaped = True
            continue


        if char == '"':

            in_string = not in_string
            continue


        if in_string:
            continue


        if char == "{":

            depth += 1


        elif char == "}":

            depth -= 1


            if depth == 0:

                candidate = text[
                    start:i + 1
                ]


                try:

                    obj = json.loads(
                        candidate
                    )

                    if isinstance(obj, dict):
                        return obj

                except Exception:
                    return None


    return None


# ============================================================
# HIERARCHICAL FIELD HELPERS
# ============================================================

def collect_field_values(
    target
):

    """
    Return:
        {
            flat_field_name: value
        }

    from the V5.1 hierarchical schema.
    """

    if not isinstance(
        target,
        dict
    ):

        return {}


    output = {}


    profile = target.get(
        "profile",
        {}
    )


    if isinstance(
        profile,
        dict
    ):

        for field in PROFILE_FIELDS:

            output[field] = profile.get(
                field
            )


    behavioral = target.get(
        "behavioral",
        {}
    )


    if isinstance(
        behavioral,
        dict
    ):

        for field in BEHAVIORAL_FIELDS:

            output[field] = behavioral.get(
                field
            )


    availability = target.get(
        "availability",
        {}
    )


    if isinstance(
        availability,
        dict
    ):

        for field in AVAILABILITY_FIELDS:

            output[field] = availability.get(
                field
            )


    output["motivation"] = target.get(
        "motivation"
    )

    output["role_specific_suitability"] = target.get(
        "role_fit"
    )

    output["other_explicit_facts"] = target.get(
        "other_explicit_facts",
        {}
    )


    return output


def non_empty_flat_fields(
    target
):

    flat = collect_field_values(
        target
    )


    return {
        field

        for field, value
        in flat.items()

        if not is_empty(value)
    }


# ============================================================
# EXACT SCHEMA VALIDATION
# ============================================================

def schema_is_valid(
    prediction
):

    if not isinstance(
        prediction,
        dict
    ):

        return False


    if set(
        prediction.keys()
    ) != TOP_LEVEL_FIELDS:

        return False


    profile = prediction.get(
        "profile"
    )

    if not isinstance(
        profile,
        dict
    ):

        return False


    if set(
        profile.keys()
    ) != PROFILE_FIELDS:

        return False


    behavioral = prediction.get(
        "behavioral"
    )

    if not isinstance(
        behavioral,
        dict
    ):

        return False


    if set(
        behavioral.keys()
    ) != BEHAVIORAL_FIELDS:

        return False


    availability = prediction.get(
        "availability"
    )

    if not isinstance(
        availability,
        dict
    ):

        return False


    if set(
        availability.keys()
    ) != AVAILABILITY_FIELDS:

        return False


    if not isinstance(
        prediction.get(
            "other_explicit_facts"
        ),
        dict
    ):

        return False


    return True


# ============================================================
# VALUE FLATTENING
# ============================================================

def flatten_value(
    value
):

    if is_empty(value):
        return []


    if isinstance(
        value,
        list
    ):

        output = []

        for item in value:

            output.extend(
                flatten_value(item)
            )

        return output


    if isinstance(
        value,
        dict
    ):

        output = []

        for key, item in value.items():

            if is_empty(item):
                continue


            children = flatten_value(
                item
            )


            for child in children:

                output.append(
                    f"{key}: {child}"
                )

        return output


    return [
        clean(value)
    ]


# ============================================================
# TEXT SIMILARITY
# ============================================================

def levenshtein_similarity(
    a,
    b
):

    a = normalize_text(a)
    b = normalize_text(b)


    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0


    previous = list(
        range(
            len(b) + 1
        )
    )


    for i, ca in enumerate(
        a,
        start=1
    ):

        current = [i]


        for j, cb in enumerate(
            b,
            start=1
        ):

            insertion = (
                current[j - 1]
                + 1
            )

            deletion = (
                previous[j]
                + 1
            )

            substitution = (
                previous[j - 1]
                +
                (
                    0
                    if ca == cb
                    else 1
                )
            )


            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )


        previous = current


    distance = previous[-1]

    max_len = max(
        len(a),
        len(b)
    )


    return (
        1
        -
        distance / max_len
    )


def token_similarity(
    a,
    b
):

    tokens_a = set(
        normalize_text(a).split()
    )

    tokens_b = set(
        normalize_text(b).split()
    )


    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0


    intersection = (
        tokens_a
        &
        tokens_b
    )

    union = (
        tokens_a
        |
        tokens_b
    )


    jaccard = (
        len(intersection)
        /
        len(union)
    )


    containment = max(

        len(intersection)
        /
        len(tokens_a),

        len(intersection)
        /
        len(tokens_b)
    )


    return max(
        jaccard,
        containment
    )


def value_similarity(
    predicted,
    target
):

    predicted_values = flatten_value(
        predicted
    )

    target_values = flatten_value(
        target
    )


    if not predicted_values and not target_values:
        return 1.0

    if not predicted_values or not target_values:
        return 0.0


    matches = []

    used = set()


    for predicted_value in (
        predicted_values
    ):

        best_score = 0.0
        best_index = None


        for i, target_value in enumerate(
            target_values
        ):

            if i in used:
                continue


            token_score = token_similarity(
                predicted_value,
                target_value
            )


            char_score = levenshtein_similarity(
                predicted_value,
                target_value
            )


            score = (
                0.50 * token_score
                +
                0.50 * char_score
            )


            if score > best_score:

                best_score = score
                best_index = i


        if best_index is not None:

            used.add(
                best_index
            )


        matches.append(
            best_score
        )


    precision_like = (
        sum(matches)
        /
        len(matches)
    )


    recall_like = (
        len(used)
        /
        len(target_values)
    )


    return (
        0.50 * precision_like
        +
        0.50 * recall_like
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print(
    "BLUEHYREAI - RUN 5 TEST EVALUATION"
)
print("=" * 80)
print()


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.exists(
    TEST_FILE
):

    raise FileNotFoundError(
        f"V5.1 test file not found:\n"
        f"{TEST_FILE}"
    )


if not os.path.exists(
    RUN5_ADAPTER
):

    raise FileNotFoundError(
        f"Run 5 adapter not found:\n"
        f"{RUN5_ADAPTER}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files=TEST_FILE,
    split="train"
)


TEST_COUNT = len(
    dataset
)


print(
    f"Test examples: {TEST_COUNT:,}"
)


if TEST_COUNT != 888:

    raise RuntimeError(
        f"Expected 888 test examples, "
        f"found {TEST_COUNT}."
    )


# ============================================================
# PREPARE RECORDS
# ============================================================

records = []


for index, row in enumerate(
    dataset
):

    messages = row.get(
        "messages",
        []
    )


    user_content = ""

    target_content = ""


    for message in messages:

        role = message.get(
            "role",
            ""
        )

        content = clean(
            message.get(
                "content",
                ""
            )
        )


        if role == "user":

            user_content = content


        elif role == "assistant":

            target_content = content


    target = extract_json(
        target_content
    )


    if target is None:

        raise RuntimeError(
            f"Invalid target JSON "
            f"at test row {index}."
        )


    if not schema_is_valid(
        target
    ):

        raise RuntimeError(
            f"Target schema invalid "
            f"at test row {index}."
        )


    records.append({

        "index":
            index,

        "user":
            user_content,

        "target":
            target
    })


# ============================================================
# LOAD BASE MODEL
# ============================================================

print()
print(
    "Loading base Gemma 3 1B..."
)


model, tokenizer = (
    FastLanguageModel.from_pretrained(

        model_name=BASE_MODEL,

        max_seq_length=MAX_SEQ_LENGTH,

        load_in_4bit=True,

        dtype=None
    )
)


print(
    "Base model loaded."
)


# ============================================================
# LOAD RUN 5 ADAPTER
# ============================================================

print()
print(
    "Loading Run 5 QLoRA adapter..."
)

print(
    RUN5_ADAPTER
)


model = PeftModel.from_pretrained(

    model,

    RUN5_ADAPTER,

    is_trainable=False
)


# ============================================================
# INFERENCE MODE
# ============================================================

try:

    model = FastLanguageModel.for_inference(
        model
    )

except Exception:
    pass


model.eval()


try:

    model.generation_config.max_length = None

except Exception:
    pass


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

all_rows = []


if os.path.exists(
    PREDICTION_FILE
):

    existing_df = pd.read_csv(

        PREDICTION_FILE,

        encoding="utf-8-sig"
    )


    if "index" in existing_df.columns:

        all_rows = (
            existing_df
            .to_dict(
                "records"
            )
        )


    print()
    print(
        f"Existing predictions : "
        f"{len(all_rows):,}/{TEST_COUNT:,}"
    )


completed_indexes = {

    int(row["index"])

    for row in all_rows

}


remaining = [

    record

    for record in records

    if record["index"]
    not in completed_indexes

]


print(
    f"Remaining predictions: "
    f"{len(remaining):,}"
)


# ============================================================
# GENERATION
# ============================================================

if remaining:

    total_batches = (

        len(remaining)
        +
        BATCH_SIZE
        -
        1

    ) // BATCH_SIZE


    generation_start = time.time()


    print()
    print("=" * 80)
    print(
        "GENERATING RUN 5 PREDICTIONS"
    )
    print("=" * 80)
    print()


    for batch_number, start in enumerate(

        range(
            0,
            len(remaining),
            BATCH_SIZE
        ),

        start=1
    ):

        batch = remaining[
            start:
            start + BATCH_SIZE
        ]


        prompts = []


        for record in batch:

            messages = [

                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT.strip()
                },

                {
                    "role":
                        "user",

                    "content":
                        record["user"]
                }

            ]


            prompt = (
                tokenizer.apply_chat_template(

                    messages,

                    tokenize=False,

                    add_generation_prompt=True
                )
            )


            prompts.append(
                prompt
            )


        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = tokenizer(

            prompts,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=MAX_SEQ_LENGTH
        )


        inputs = {

            key:
                value.cuda()

            for key, value
            in inputs.items()
        }


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        with torch.inference_mode():

            generated = model.generate(

                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=False,

                num_beams=1
            )


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        input_lengths = (

            inputs[
                "attention_mask"
            ]
            .sum(
                dim=1
            )
            .tolist()

        )


        batch_rows = []


        for i, output in enumerate(
            generated
        ):

            new_tokens = output[
                int(
                    input_lengths[i]
                ):
            ]


            text = tokenizer.decode(

                new_tokens,

                skip_special_tokens=True
            ).strip()


            parsed = extract_json(
                text
            )


            record = batch[i]


            batch_rows.append({

                "index":
                    record["index"],

                "question_and_answer":
                    record["user"],

                "ground_truth":
                    json.dumps(
                        record["target"],
                        ensure_ascii=False
                    ),

                "model_output":
                    text,

                "parsed_json":
                    json.dumps(
                        parsed,
                        ensure_ascii=False
                    )
                    if parsed is not None
                    else "",

                "json_valid":
                    parsed is not None,

                "schema_valid":
                    schema_is_valid(
                        parsed
                    )
            })


        # ----------------------------------------------------
        # SAVE AFTER EVERY BATCH
        # ----------------------------------------------------

        all_rows.extend(
            batch_rows
        )


        all_rows.sort(
            key=lambda row:
                int(
                    row["index"]
                )
        )


        pd.DataFrame(
            all_rows
        ).to_csv(

            PREDICTION_FILE,

            index=False,

            encoding="utf-8-sig"
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        completed = len(
            all_rows
        )


        elapsed = (
            time.time()
            -
            generation_start
        )


        new_examples = (
            completed
            -
            len(existing_rows)
            if "existing_rows" in locals()
            else completed
        )


        rate = (

            new_examples
            /
            elapsed

            if elapsed > 0

            else 0
        )


        percent = (

            completed
            /
            TEST_COUNT
            *
            100
        )


        print(

            f"Batch "
            f"{batch_number:>3}/"
            f"{total_batches:<3} | "
            f"Examples "
            f"{completed:>3}/"
            f"{TEST_COUNT} | "
            f"{percent:>6.2f}% | "
            f"{rate:>5.2f} ex/s"
        )


        del inputs
        del generated

        torch.cuda.empty_cache()


# ============================================================
# FINAL COMPLETENESS CHECK
# ============================================================

final_df = pd.read_csv(

    PREDICTION_FILE,

    encoding="utf-8-sig"
)


final_df = (

    final_df
    .sort_values(
        "index"
    )
    .reset_index(
        drop=True
    )

)


if len(final_df) != TEST_COUNT:

    raise RuntimeError(

        f"Evaluation incomplete.\n"

        f"Expected: {TEST_COUNT}\n"

        f"Found: {len(final_df)}"
    )


print()
print("=" * 80)
print(
    "RUN 5 GENERATION COMPLETE"
)
print("=" * 80)
print()

print(
    f"Predictions: "
    f"{len(final_df):,}/{TEST_COUNT:,}"
)


# ============================================================
# FREE MODEL
# ============================================================

del model
del tokenizer

gc.collect()

torch.cuda.empty_cache()


# ============================================================
# METRIC COUNTERS
# ============================================================

json_valid = 0

schema_valid = 0

exact_json = 0

empty_outputs = 0

field_tp = 0

field_fp = 0

field_fn = 0

value_correct = 0

value_incorrect = 0


# IMPORTANT:
# Use FLAT metric field names for comparability with Runs 1-4.

METRIC_FIELDS = [

    "experience",

    "previous_responsibilities",

    "skills",

    "education_certifications",

    "problem_solving",

    "teamwork",

    "communication",

    "conflict_resolution",

    "prioritization",

    "availability",

    "notice_period",

    "salary_expectations",

    "motivation",

    "role_specific_suitability",

    "other_explicit_facts"
]


field_stats = {

    field: {

        "actual":
            0,

        "predicted":
            0,

        "presence_correct":
            0,

        "value_correct":
            0,

        "value_incorrect":
            0

    }

    for field in METRIC_FIELDS
}


# ============================================================
# EVALUATE
# ============================================================

print()
print(
    "Calculating metrics..."
)


for _, row in final_df.iterrows():

    prediction = extract_json(
        row[
            "parsed_json"
        ]
    )

    target = extract_json(
        row[
            "ground_truth"
        ]
    )


    # --------------------------------------------------------
    # JSON validity
    # --------------------------------------------------------

    if isinstance(
        prediction,
        dict
    ):

        json_valid += 1


    # --------------------------------------------------------
    # Exact hierarchical schema
    # --------------------------------------------------------

    if schema_is_valid(
        prediction
    ):

        schema_valid += 1


    # --------------------------------------------------------
    # Exact JSON
    # --------------------------------------------------------

    if (

        isinstance(
            prediction,
            dict
        )

        and

        isinstance(
            target,
            dict
        )

        and

        json.dumps(
            prediction,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        )

        ==

        json.dumps(
            target,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        )

    ):

        exact_json += 1


    # --------------------------------------------------------
    # FLAT FIELD PRESENCE
    # --------------------------------------------------------

    predicted_fields = (
        non_empty_flat_fields(
            prediction
        )
    )

    actual_fields = (
        non_empty_flat_fields(
            target
        )
    )


    if not predicted_fields:

        empty_outputs += 1


    common_fields = (

        predicted_fields
        &
        actual_fields

    )


    extra_fields = (

        predicted_fields
        -
        actual_fields

    )


    missing_fields = (

        actual_fields
        -
        predicted_fields

    )


    field_tp += len(
        common_fields
    )

    field_fp += len(
        extra_fields
    )

    field_fn += len(
        missing_fields
    )


    # --------------------------------------------------------
    # Field statistics
    # --------------------------------------------------------

    prediction_flat = (
        collect_field_values(
            prediction
        )
    )

    target_flat = (
        collect_field_values(
            target
        )
    )


    for field in actual_fields:

        field_stats[
            field
        ]["actual"] += 1


    for field in predicted_fields:

        field_stats[
            field
        ]["predicted"] += 1


    for field in common_fields:

        field_stats[
            field
        ]["presence_correct"] += 1


        score = value_similarity(

            prediction_flat.get(
                field
            ),

            target_flat.get(
                field
            )
        )


        if score >= 0.75:

            value_correct += 1

            field_stats[
                field
            ]["value_correct"] += 1

        else:

            value_incorrect += 1

            field_stats[
                field
            ]["value_incorrect"] += 1


# ============================================================
# OVERALL METRICS
# ============================================================

total = TEST_COUNT


field_precision = (

    field_tp
    /
    (
        field_tp
        +
        field_fp
    )

    if (
        field_tp
        +
        field_fp
    )

    else 0
)


field_recall = (

    field_tp
    /
    (
        field_tp
        +
        field_fn
    )

    if (
        field_tp
        +
        field_fn
    )

    else 0
)


field_f1 = (

    2
    *
    field_precision
    *
    field_recall
    /
    (
        field_precision
        +
        field_recall
    )

    if (
        field_precision
        +
        field_recall
    )

    else 0
)


value_precision = (

    value_correct
    /
    (
        value_correct
        +
        value_incorrect
    )

    if (
        value_correct
        +
        value_incorrect
    )

    else 0
)


# ============================================================
# RESULT OBJECT
# ============================================================

metrics = {

    "experiment":
        "run5",

    "model":
        "Gemma 3 1B + Run 5 QLoRA",

    "dataset":
        "llama_extraction_v5_1",

    "schema_type":
        "hierarchical_context_corrected",

    "examples":
        total,

    "json_valid":
        json_valid,

    "json_valid_rate":
        json_valid / total,

    "schema_valid":
        schema_valid,

    "schema_valid_rate":
        schema_valid / total,

    "exact_json":
        exact_json,

    "exact_json_rate":
        exact_json / total,

    "empty_outputs":
        empty_outputs,

    "empty_output_rate":
        empty_outputs / total,

    "field_tp":
        field_tp,

    "field_fp":
        field_fp,

    "field_fn":
        field_fn,

    "field_presence_precision":
        field_precision,

    "field_presence_recall":
        field_recall,

    "field_presence_f1":
        field_f1,

    "value_correct":
        value_correct,

    "value_incorrect":
        value_incorrect,

    "value_precision_under_heuristic":
        value_precision,

    "fields":
        {}
}


# ============================================================
# PER-FIELD METRICS
# ============================================================

field_rows = []


for field in METRIC_FIELDS:

    stats = field_stats[
        field
    ]


    actual = stats[
        "actual"
    ]

    predicted = stats[
        "predicted"
    ]

    presence_correct = stats[
        "presence_correct"
    ]

    correct_values = stats[
        "value_correct"
    ]


    presence_precision = (

        presence_correct
        /
        predicted

        if predicted

        else 0
    )


    presence_recall = (

        presence_correct
        /
        actual

        if actual

        else 0
    )


    presence_f1 = (

        2
        *
        presence_precision
        *
        presence_recall
        /
        (
            presence_precision
            +
            presence_recall
        )

        if (

            presence_precision
            +
            presence_recall

        )

        else 0
    )


    value_precision_field = (

        correct_values
        /
        predicted

        if predicted

        else 0
    )


    value_recall_field = (

        correct_values
        /
        actual

        if actual

        else 0
    )


    value_f1_field = (

        2
        *
        value_precision_field
        *
        value_recall_field
        /
        (
            value_precision_field
            +
            value_recall_field
        )

        if (

            value_precision_field
            +
            value_recall_field

        )

        else 0
    )


    metrics[
        "fields"
    ][
        field
    ] = {

        "actual":
            actual,

        "predicted":
            predicted,

        "presence_correct":
            presence_correct,

        "value_correct":
            correct_values,

        "presence_precision":
            presence_precision,

        "presence_recall":
            presence_recall,

        "presence_f1":
            presence_f1,

        "value_precision":
            value_precision_field,

        "value_recall":
            value_recall_field,

        "value_f1":
            value_f1_field
    }


    field_rows.append({

        "field":
            field,

        "actual":
            actual,

        "predicted":
            predicted,

        "presence_correct":
            presence_correct,

        "value_correct":
            correct_values,

        "presence_precision":
            presence_precision,

        "presence_recall":
            presence_recall,

        "presence_f1":
            presence_f1,

        "value_precision":
            value_precision_field,

        "value_recall":
            value_recall_field,

        "value_f1":
            value_f1_field
    })


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        ensure_ascii=False,
        indent=2
    )


pd.DataFrame(
    field_rows
).to_csv(

    FIELD_METRICS_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print(
    "BLUEHYREAI - RUN 5 FINAL EVALUATION"
)
print("=" * 80)
print()

print(
    f"Model               : "
    f"{metrics['model']}"
)

print(
    f"Test set            : "
    f"{metrics['dataset']}"
)

print(
    f"Schema              : "
    f"{metrics['schema_type']}"
)

print(
    f"Examples            : "
    f"{total:,}"
)

print()

print(
    f"JSON validity       : "
    f"{metrics['json_valid_rate']:.2%}"
)

print(
    f"Schema validity     : "
    f"{metrics['schema_valid_rate']:.2%}"
)

print(
    f"Exact JSON          : "
    f"{metrics['exact_json_rate']:.2%}"
)

print(
    f"Field presence P    : "
    f"{metrics['field_presence_precision']:.2%}"
)

print(
    f"Field presence R    : "
    f"{metrics['field_presence_recall']:.2%}"
)

print(
    f"Field presence F1   : "
    f"{metrics['field_presence_f1']:.2%}"
)

print(
    f"Value heuristic P   : "
    f"{metrics['value_precision_under_heuristic']:.2%}"
)

print(
    f"Empty outputs       : "
    f"{metrics['empty_output_rate']:.2%}"
)

print()

print(
    "PREDICTIONS:"
)

print(
    PREDICTION_FILE
)

print()

print(
    "METRICS:"
)

print(
    METRICS_FILE
)

print()

print(
    "FIELD METRICS:"
)

print(
    FIELD_METRICS_FILE
)

print()
print("=" * 80)
print(
    "RUN 5 EVALUATION COMPLETE"
)
print("=" * 80)