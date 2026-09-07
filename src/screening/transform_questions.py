import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"F:\BlueHyre Project Final\02_Screening"

# IMPORTANT:
# THIS IS THE SAME VERIFIED CSV WE ALREADY CREATED.
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "screening_interviews_hi_verified.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Result",
    "screening_interviews_final.csv"
)

CHECKPOINT_FILE = os.path.join(
    BASE_DIR,
    "screening_interviews_transform_checkpoint_v2.csv"
)

ERROR_FILE = os.path.join(
    BASE_DIR,
    "screening_interviews_transform_errors_v2.csv"
)

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

# ------------------------------------------------------------
# BATCH
# ------------------------------------------------------------

BATCH_SIZE = 100

# PILOT FIRST
#
# Keep this at 100 for the first run.
#
# After checking the result, change to:
#
# MAX_ROWS = None
#
# for the complete 4,437 rows.
MAX_ROWS = None

MODEL_NAME = "gemini-3.1-flash-lite"

MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 5


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        f"GEMINI_API_KEY not found:\n{ENV_FILE}"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

class TransformationRecord(BaseModel):

    record_id: str = Field(
        description="Exact input record ID. Must never change."
    )

    is_transform: bool = Field(
        description=(
            "True only when the question is materially "
            "technical and should be rewritten into a "
            "general operational/frontline workplace question."
        )
    )

    transformation_reason: str = Field(
        description=(
            "Short explanation. Use 'already suitable for "
            "general screening' when is_transform is false."
        )
    )

    question_en_transformed: str = Field(
        description=(
            "If is_transform is true, provide the transformed "
            "English screening question. If false, repeat "
            "the original English question exactly."
        )
    )

    question_hi_transformed: str = Field(
        description=(
            "If is_transform is true, provide the transformed "
            "Hindi screening question. If false, repeat "
            "the original Hindi question exactly."
        )
    )


class TransformationBatch(BaseModel):

    records: list[TransformationRecord] = Field(
        description=(
            "Exactly one result for every input record, "
            "with no omissions or additions."
        )
    )


# ============================================================
# LOAD INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print()
print("=" * 75)
print("BLUEHYREAI - TECHNICAL QUESTION TRANSFORMATION")
print("=" * 75)

print(
    f"Input rows : {len(df)}"
)

print(
    f"Batch size : {BATCH_SIZE}"
)

print(
    f"Model      : {MODEL_NAME}"
)

print(
    f"Max rows   : {MAX_ROWS}"
)

print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "question",
    "question_hi"
]

for column in required_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Required column missing: {column}"
        )


# ============================================================
# STABLE RECORD ID
# ============================================================

if "record_id" not in df.columns:

    df.insert(
        0,
        "record_id",
        [
            f"BH-{i:05d}"
            for i in range(
                1,
                len(df) + 1
            )
        ]
    )


# ============================================================
# CLEAN
# ============================================================

for column in [
    "record_id",
    "question",
    "question_hi"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


if "bluehyre_category" not in df.columns:

    df["bluehyre_category"] = ""


df["bluehyre_category"] = (
    df["bluehyre_category"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# WORKING SET
# ============================================================

if MAX_ROWS is None:

    work_df = df.copy()

else:

    work_df = df.head(
        MAX_ROWS
    ).copy()


print(
    f"Rows to process: {len(work_df)}"
)

print()


# ============================================================
# PROMPT
# ============================================================

def build_prompt(batch):

    records = []

    for _, row in batch.iterrows():

        records.append({

            "record_id":
                str(row["record_id"]),

            "question_en":
                str(row["question"]),

            "question_hi":
                str(row["question_hi"]),

            "bluehyre_category":
                str(
                    row["bluehyre_category"]
                ),

            "role":
                str(
                    row.get(
                        "role",
                        ""
                    )
                ),

            "sector":
                str(
                    row.get(
                        "sector",
                        ""
                    )
                )
        })


    payload = json.dumps(
        records,
        ensure_ascii=False,
        indent=2
    )


    return f"""
You are preparing interview questions for a
Hindi-first voice-based employment screening system.

You are reviewing questions that have ALREADY been
translated and verified.

Your job is NOT to rewrite everything.

For every question, decide:

KEEP
or
TRANSFORM

------------------------------------------------------------
KEEP
------------------------------------------------------------

Keep the question unchanged when it is already a useful
general employment-screening question.

Examples:

"Tell me about your previous work experience."

"How do you handle disagreements with coworkers?"

"What motivates you to take this job?"

"When would you be available to join?"

"What salary are you expecting?"

For KEEP:

- Do not rewrite the English question.
- Do not rewrite the Hindi question.
- Return them exactly as provided.

------------------------------------------------------------
TRANSFORM
------------------------------------------------------------

Transform ONLY when the question is materially dependent
on technical/specialist subject matter that would make it
unsuitable for a general operational/frontline screening
conversation.

Examples of technical context:

- programming
- software frameworks
- databases
- APIs
- cloud infrastructure
- networking
- cybersecurity
- machine learning implementation
- system architecture
- software architecture
- technical implementation
- highly specialized engineering tools
- specialist engineering infrastructure

IMPORTANT:

Do NOT merely delete technical words.

Identify the underlying HUMAN/EMPLOYMENT COMPETENCY and
preserve it.

Then replace the technical context with a realistic,
general workplace/operational situation.

------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

SOURCE:

"How would you troubleshoot a Kafka cluster failure?"

UNDERLYING COMPETENCY:

Problem solving.

TRANSFORM:

"If an important piece of equipment suddenly stops
working during your shift, how would you identify the
problem and decide what to do next?"

------------------------------------------------------------

SOURCE:

"How would you prioritize multiple production incidents
with conflicting deadlines?"

UNDERLYING COMPETENCY:

Prioritization.

TRANSFORM:

"If you are given several urgent tasks at the same time
and cannot complete all of them immediately, how would
you decide which task to handle first?"

------------------------------------------------------------

SOURCE:

"How would you resolve a disagreement with another
engineer about system architecture?"

UNDERLYING COMPETENCY:

Conflict resolution.

TRANSFORM:

"If you disagree with a coworker about how a task should
be done, how would you handle the disagreement?"

------------------------------------------------------------

SOURCE:

"How would you explain a complex technical issue to a
non-technical stakeholder?"

UNDERLYING COMPETENCY:

Communication.

TRANSFORM:

"If you need to explain a difficult work-related issue
to a customer or supervisor who is not familiar with
the details, how would you explain it?"

------------------------------------------------------------
BLUE-COLLAR / OPERATIONAL CONTEXT
------------------------------------------------------------

When transforming, make the situation broadly relevant
to workers such as:

- delivery workers
- warehouse workers
- retail workers
- field service workers
- manufacturing workers
- security workers
- hospitality workers
- drivers
- maintenance workers
- customer-facing workers
- general frontline workers

Do NOT force a specific occupation unless the original
question already provides sufficient context.

The transformed question should test the SAME underlying
competency as the original.

Do not turn:

problem solving → teamwork

or:

conflict resolution → motivation

or:

prioritization → self introduction.

Preserve the provided BlueHyre category.

------------------------------------------------------------
HINDI
------------------------------------------------------------

For TRANSFORM:

Create natural spoken Indian Hindi suitable for a recruiter
speaking over a phone call.

Do not use unnecessarily literary Hindi.

Common English workplace words may remain in English where
that is natural.

For KEEP:

Return the original Hindi EXACTLY.

------------------------------------------------------------
IMPORTANT SAFETY / QUALITY RULES
------------------------------------------------------------

Do NOT:

- invent candidate facts
- invent job requirements
- introduce discriminatory criteria
- introduce medical requirements
- introduce age/race/religion/caste/gender screening
- change the category
- change the interview objective
- add unrelated skills
- create a completely different question

Every input record MUST appear exactly once.

No omitted records.
No extra records.
No duplicate records.

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

Return exactly one structured record for each input.

For KEEP:

is_transform = false

For TRANSFORM:

is_transform = true

The output English and Hindi fields must ALWAYS be non-empty.

INPUT:

{payload}
"""


# ============================================================
# GEMINI CALL WITH STRUCTURED OUTPUT
# ============================================================

def process_batch(
    batch
):

    expected_ids = {
        str(record_id)
        for record_id in
        batch["record_id"].tolist()
    }


    prompt = build_prompt(
        batch
    )


    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = (
                client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config={
                        "response_mime_type":
                            "application/json",

                        "response_json_schema":
                            TransformationBatch
                            .model_json_schema()
                    }
                )
            )


            # ------------------------------------------------
            # Parse structured JSON
            # ------------------------------------------------

            if (
                hasattr(response, "parsed")
                and response.parsed
            ):

                parsed = response.parsed

                if isinstance(
                    parsed,
                    TransformationBatch
                ):

                    result = parsed.records

                else:

                    result = (
                        TransformationBatch
                        .model_validate(
                            parsed
                        )
                        .records
                    )

            else:

                # Fallback
                raw = json.loads(
                    response.text
                )

                result = (
                    TransformationBatch
                    .model_validate(
                        raw
                    )
                    .records
                )


            # ------------------------------------------------
            # Convert Pydantic objects
            # ------------------------------------------------

            results = [
                item.model_dump()
                for item in result
            ]


            # ------------------------------------------------
            # Validate IDs
            # ------------------------------------------------

            returned_ids = {
                str(item["record_id"])
                for item in results
            }


            if returned_ids != expected_ids:

                missing = (
                    expected_ids
                    - returned_ids
                )

                extra = (
                    returned_ids
                    - expected_ids
                )

                raise ValueError(
                    "Record ID mismatch. "
                    f"Missing={sorted(missing)} "
                    f"Extra={sorted(extra)}"
                )


            if len(results) != len(batch):

                raise ValueError(
                    f"Expected {len(batch)} records, "
                    f"got {len(results)}."
                )


            # ------------------------------------------------
            # Validate fields
            # ------------------------------------------------

            for item in results:

                if not item[
                    "question_en_transformed"
                ].strip():

                    raise ValueError(
                        "Empty transformed English question."
                    )

                if not item[
                    "question_hi_transformed"
                ].strip():

                    raise ValueError(
                        "Empty transformed Hindi question."
                    )


            return results


        except Exception as e:

            last_error = e

            print(
                f"    Attempt "
                f"{attempt}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS
                    * attempt
                )


    raise RuntimeError(
        "Gemini batch failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# SAFE CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    records
):

    if not records:
        return

    checkpoint_df = (
        pd.DataFrame(
            records
        )
    )

    temp_file = (
        CHECKPOINT_FILE
        + ".tmp"
    )


    try:

        if os.path.exists(
            temp_file
        ):

            os.remove(
                temp_file
            )

    except PermissionError:
        pass


    checkpoint_df.to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig"
    )


    os.replace(
        temp_file,
        CHECKPOINT_FILE
    )


# ============================================================
# PROCESS
# ============================================================

all_results = []
errors = []

total_batches = (
    (
        len(work_df)
        + BATCH_SIZE
        - 1
    )
    // BATCH_SIZE
)


for batch_number, start in enumerate(
    range(
        0,
        len(work_df),
        BATCH_SIZE
    ),
    start=1
):

    end = min(
        start + BATCH_SIZE,
        len(work_df)
    )

    batch = work_df.iloc[
        start:end
    ].copy()


    print()
    print(
        f"Batch {batch_number}/"
        f"{total_batches}"
    )

    print(
        f"Rows: {start + 1}-{end}"
    )


    try:

        results = process_batch(
            batch
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # For KEEP records, preserve the original English
        # and Hindi locally.
        # ----------------------------------------------------

        batch_result_map = {

            str(item["record_id"]):
                item

            for item in results
        }


        for _, row in batch.iterrows():

            record_id = str(
                row["record_id"]
            )

            item = (
                batch_result_map[
                    record_id
                ]
            )


            is_transform = bool(
                item["is_transform"]
            )


            if is_transform:

                final_en = (
                    item[
                        "question_en_transformed"
                    ]
                )

                final_hi = (
                    item[
                        "question_hi_transformed"
                    ]
                )

                reason = (
                    item[
                        "transformation_reason"
                    ]
                )

            else:

                # Preserve source exactly.
                final_en = str(
                    row["question"]
                )

                final_hi = str(
                    row["question_hi"]
                )

                reason = (
                    "already suitable for "
                    "general screening"
                )


            all_results.append({

                "record_id":
                    record_id,

                "is_transformed":
                    is_transform,

                "transformation_reason":
                    reason,

                "question_en_final":
                    final_en,

                "question_hi_final":
                    final_hi,

                "bluehyre_category":
                    str(
                        row[
                            "bluehyre_category"
                        ]
                    )
            })


        transformed = sum(
            1
            for item in results
            if item["is_transform"]
        )


        unchanged = (
            len(results)
            - transformed
        )


        print(
            f"Complete: {len(results)}"
        )

        print(
            f"Transformed: {transformed}"
        )

        print(
            f"Unchanged: {unchanged}"
        )


        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        try:

            save_checkpoint(
                all_results
            )

            print(
                "Checkpoint saved."
            )

        except Exception as checkpoint_error:

            print(
                "WARNING: checkpoint save failed:"
            )

            print(
                f"{type(checkpoint_error).__name__}: "
                f"{checkpoint_error}"
            )


    except Exception as e:

        print()
        print(
            f"BATCH FAILED: "
            f"{type(e).__name__}: {e}"
        )


        # ----------------------------------------------------
        # Important:
        # Save each failed batch's IDs.
        # ----------------------------------------------------

        for record_id in batch[
            "record_id"
        ].tolist():

            errors.append({

                "record_id":
                    str(record_id),

                "batch":
                    batch_number,

                "error":
                    str(e)
            })


# ============================================================
# MERGE ORIGINAL DATA + FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


if not results_df.empty:

    results_df["record_id"] = (
        results_df[
            "record_id"
        ]
        .astype(str)
    )


work_df["record_id"] = (
    work_df[
        "record_id"
    ]
    .astype(str)
)


final_df = work_df.merge(
    results_df[
        [
            "record_id",
            "is_transformed",
            "transformation_reason",
            "question_en_final",
            "question_hi_final"
        ]
    ],
    on="record_id",
    how="left"
)


# ============================================================
# VALIDATION
# ============================================================

missing_en = (
    final_df[
        "question_en_final"
    ]
    .isna()
    .sum()
)

missing_hi = (
    final_df[
        "question_hi_final"
    ]
    .isna()
    .sum()
)

duplicate_ids = (
    final_df[
        "record_id"
    ]
    .duplicated()
    .sum()
)

transformed_count = (
    final_df[
        "is_transformed"
    ]
    .eq(True)
    .sum()
)

unchanged_count = (
    final_df[
        "is_transformed"
    ]
    .eq(False)
    .sum()
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE ERRORS
# ============================================================

if errors:

    pd.DataFrame(
        errors
    ).to_csv(
        ERROR_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("QUESTION TRANSFORMATION COMPLETE")
print("=" * 75)

print(
    f"Input rows            : {len(work_df)}"
)

print(
    f"Final rows            : {len(final_df)}"
)

print(
    f"Transformed           : {transformed_count}"
)

print(
    f"Unchanged             : {unchanged_count}"
)

print(
    f"Missing English       : {missing_en}"
)

print(
    f"Missing Hindi         : {missing_hi}"
)

print(
    f"Duplicate record IDs  : {duplicate_ids}"
)

print(
    f"Failed records        : {len(errors)}"
)

print()

print(
    "Final output:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_FILE
)

if errors:

    print()

    print(
        "Errors:"
    )

    print(
        ERROR_FILE
    )


# ============================================================
# SAMPLE
# ============================================================

print()
print(
    "SAMPLE RESULTS"
)

for _, row in final_df.head(
    10
).iterrows():

    print()
    print(
        f"[{row['record_id']}]"
    )

    print(
        "Transformed:",
        row["is_transformed"]
    )

    print(
        "Reason:",
        row["transformation_reason"]
    )

    print(
        "Original EN:",
        row["question"]
    )

    print(
        "Final EN:",
        row["question_en_final"]
    )

    print(
        "Original HI:",
        row["question_hi"]
    )

    print(
        "Final HI:",
        row["question_hi_final"]
    )


print()
print("=" * 75)