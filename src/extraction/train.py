import os
import json
import math
import time

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


# ============================================================
# BLUEHYREAI - GEMMA 3 1B QLORA - RUN 5
# ============================================================
#
# FINAL TRAINING DATASET:
#   llama_extraction_v5_1
#
# EXPERIMENT:
#   Final training run using the final frozen dataset.
#
# IMPORTANT:
#   - Fresh training run
#   - No checkpoint resume
#   - No Run 3 / Run 4 adapter loading
#   - Test set is NOT used for training
#   - Same hyperparameters as Run 3 / Run 4
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"F:\BlueHyre Project Final\03_Extraction"

DATA_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "01_extraction_v5_1"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.jsonl"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "validation.jsonl"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.jsonl"
)


# Fresh Run 5 output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "02_gemma3_1b_qlora_run5"
)

FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "training_summary.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "google/gemma-3-1b-it"


# ============================================================
# FINAL TRAINING CONFIGURATION
# ============================================================

MAX_SEQ_LENGTH = 512

MICRO_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 6

EFFECTIVE_BATCH_SIZE = (
    MICRO_BATCH_SIZE
    *
    GRADIENT_ACCUMULATION_STEPS
)

NUM_EPOCHS = 1

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.0

SEED = 42


# ============================================================
# CHECKPOINT / LOGGING
# ============================================================

SAVE_STEPS = 100

EVAL_STEPS = 100

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# ============================================================
# FINAL DATASET DESCRIPTION
# ============================================================

DATASET_VERSION = "BlueHyreAI Extraction V5.1"

SCHEMA_TYPE = "hierarchical_context_corrected"


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("BLUEHYREAI - GEMMA 3 1B QLORA - RUN 5")
print("=" * 80)
print()

print(
    f"Dataset      : {DATASET_VERSION}"
)

print(
    f"Schema       : {SCHEMA_TYPE}"
)

print(
    "Training     : FRESH RUN"
)

print(
    "Resume       : FALSE"
)

print()


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available.\n"
        "Activate the CUDA-enabled .venv_gemma environment."
    )


gpu_name = torch.cuda.get_device_name(0)

gpu_memory_gb = (
    torch.cuda.get_device_properties(0).total_memory
    /
    (1024 ** 3)
)


print(
    f"GPU          : {gpu_name}"
)

print(
    f"VRAM         : {gpu_memory_gb:.2f} GB"
)

print(
    f"PyTorch      : {torch.__version__}"
)

print(
    f"CUDA build   : {torch.version.cuda}"
)

print()


# ============================================================
# DATASET FILE CHECKS
# ============================================================

required_files = [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]


for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required V5.1 dataset file not found:\n{path}"
        )


# ============================================================
# FRESH OUTPUT DIRECTORY CHECK
# ============================================================

if os.path.exists(OUTPUT_DIR):

    existing_items = os.listdir(
        OUTPUT_DIR
    )

    if existing_items:

        raise RuntimeError(
            "Run 5 output directory already exists "
            "and is not empty.\n\n"
            f"{OUTPUT_DIR}\n\n"
            "This script intentionally starts a fresh run. "
            "Use a new directory or remove/rename the existing "
            "Run 5 directory intentionally."
        )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASETS
# ============================================================

print(
    "Loading V5.1 datasets..."
)

print()


train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

validation_dataset = load_dataset(
    "json",
    data_files=VAL_FILE,
    split="train"
)

test_dataset = load_dataset(
    "json",
    data_files=TEST_FILE,
    split="train"
)


print(
    f"Train examples      : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation examples : "
    f"{len(validation_dataset):,}"
)

print(
    f"Test examples       : "
    f"{len(test_dataset):,}"
)

print()


# ============================================================
# EXPECTED DATASET SIZE CHECKS
# ============================================================

if len(train_dataset) != 7098:

    raise RuntimeError(
        f"Expected 7,098 training examples, "
        f"found {len(train_dataset):,}."
    )


if len(validation_dataset) != 888:

    raise RuntimeError(
        f"Expected 888 validation examples, "
        f"found {len(validation_dataset):,}."
    )


if len(test_dataset) != 888:

    raise RuntimeError(
        f"Expected 888 test examples, "
        f"found {len(test_dataset):,}."
    )


# ============================================================
# DATASET STRUCTURE CHECK
# ============================================================

for name, dataset in [

    ("train", train_dataset),

    ("validation", validation_dataset),

    ("test", test_dataset)

]:

    if "messages" not in dataset.column_names:

        raise RuntimeError(
            f"{name} dataset does not contain "
            "'messages'."
        )


    if len(dataset) == 0:

        raise RuntimeError(
            f"{name} dataset is empty."
        )


# ============================================================
# FIRST EXAMPLE CHECK
# ============================================================

print(
    "Checking first V5.1 training example..."
)

sample = train_dataset[0]

messages = sample.get(
    "messages",
    []
)


if not messages:

    raise RuntimeError(
        "First training example has no messages."
    )


for message in messages:

    print()

    print(
        f"[{str(message.get('role', '')).upper()}]"
    )

    print(
        str(
            message.get(
                "content",
                ""
            )
        )[:1800]
    )


# ============================================================
# LOAD BASE MODEL
# ============================================================

print()
print(
    "Loading Gemma 3 1B in 4-bit..."
)

print()


model, tokenizer = (
    FastLanguageModel.from_pretrained(

        model_name=MODEL_NAME,

        max_seq_length=MAX_SEQ_LENGTH,

        load_in_4bit=True,

        dtype=None
    )
)


print(
    "Base model loaded."
)


# ============================================================
# ATTACH FRESH QLORA
# ============================================================

print()
print(
    "Attaching fresh QLoRA adapter..."
)


model = FastLanguageModel.get_peft_model(

    model,

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    bias="none",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],

    use_gradient_checkpointing="unsloth",

    random_state=SEED
)


print(
    "Fresh QLoRA adapter ready."
)


# ============================================================
# PARAMETER REPORT
# ============================================================

total_parameters = 0

trainable_parameters = 0


for parameter in model.parameters():

    count = parameter.numel()

    total_parameters += count

    if parameter.requires_grad:

        trainable_parameters += count


trainable_percentage = (

    trainable_parameters
    /
    total_parameters
    *
    100
)


print()
print("=" * 80)
print("PARAMETERS")
print("=" * 80)

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)

print(
    f"Trainable percentage : "
    f"{trainable_percentage:.4f}%"
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=MICRO_BATCH_SIZE,

    per_device_eval_batch_size=MICRO_BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    logging_steps=LOGGING_STEPS,

    logging_first_step=True,

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,

    eval_strategy="steps",

    eval_steps=EVAL_STEPS,

    bf16=True,

    fp16=False,

    tf32=False,

    optim="adamw_8bit",

    gradient_checkpointing=True,

    max_grad_norm=1.0,

    lr_scheduler_type="cosine",

    report_to="none",

    seed=SEED,

    data_seed=SEED,

    remove_unused_columns=False,

    dataloader_num_workers=0,

    dataloader_pin_memory=True
)


# ============================================================
# CREATE TRAINER
# ============================================================

print()
print(
    "Creating SFTTrainer..."
)


trainer = SFTTrainer(

    model=model,

    processing_class=tokenizer,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    args=training_args
)


print(
    "SFTTrainer ready."
)


# ============================================================
# TRAINING PLAN
# ============================================================

optimizer_steps = math.ceil(

    len(train_dataset)
    /
    EFFECTIVE_BATCH_SIZE
)


print()
print("=" * 80)
print("RUN 5 TRAINING PLAN")
print("=" * 80)

print()

print(
    f"Model                  : "
    f"{MODEL_NAME}"
)

print(
    f"Dataset                : "
    f"{DATASET_VERSION}"
)

print(
    f"Train examples         : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation examples    : "
    f"{len(validation_dataset):,}"
)

print(
    f"Test examples          : "
    f"{len(test_dataset):,}"
)

print()

print(
    f"Epochs                 : "
    f"{NUM_EPOCHS}"
)

print(
    f"Micro-batch            : "
    f"{MICRO_BATCH_SIZE}"
)

print(
    f"Gradient accumulation  : "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch        : "
    f"{EFFECTIVE_BATCH_SIZE}"
)

print(
    f"Approx optimizer steps : "
    f"{optimizer_steps:,}"
)

print()

print(
    f"Learning rate          : "
    f"{LEARNING_RATE}"
)

print(
    f"Weight decay           : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Warmup ratio           : "
    f"{WARMUP_RATIO}"
)

print(
    f"Max sequence length    : "
    f"{MAX_SEQ_LENGTH}"
)

print(
    f"LoRA rank              : "
    f"{LORA_R}"
)

print(
    f"LoRA alpha             : "
    f"{LORA_ALPHA}"
)

print(
    f"LoRA dropout           : "
    f"{LORA_DROPOUT}"
)

print()

print(
    "Resume from checkpoint : FALSE"
)

print(
    "Test set used training : FALSE"
)

print("=" * 80)


# ============================================================
# GPU MEMORY RESET
# ============================================================

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 80)
print("STARTING RUN 5 TRAINING")
print("=" * 80)
print()

print(
    "Starting from the base Gemma 3 1B model."
)

print(
    "No previous QLoRA adapter will be loaded."
)

print(
    "The 888-example test set remains untouched."
)

print()


start_time = time.time()


try:

    # --------------------------------------------------------
    # IMPORTANT:
    # No resume_from_checkpoint.
    # --------------------------------------------------------

    train_result = trainer.train()


except KeyboardInterrupt:

    print()
    print("=" * 80)
    print("RUN 5 TRAINING INTERRUPTED")
    print("=" * 80)
    print()

    print(
        "Any checkpoints already saved remain available."
    )

    raise


except Exception as exc:

    print()
    print("=" * 80)
    print("RUN 5 TRAINING FAILED")
    print("=" * 80)
    print()

    print(
        f"{type(exc).__name__}: {exc}"
    )

    print()

    raise


elapsed_seconds = (
    time.time()
    -
    start_time
)


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print()
print(
    "Saving final QLoRA adapter..."
)


os.makedirs(
    FINAL_ADAPTER_DIR,
    exist_ok=True
)


trainer.save_model(
    FINAL_ADAPTER_DIR
)


tokenizer.save_pretrained(
    FINAL_ADAPTER_DIR
)


print(
    "Final adapter saved."
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print(
    "Running final validation..."
)


validation_metrics = (
    trainer.evaluate()
)


# ============================================================
# GPU MEMORY
# ============================================================

peak_allocated_gb = (

    torch.cuda.max_memory_allocated()
    /
    (1024 ** 3)
)


peak_reserved_gb = (

    torch.cuda.max_memory_reserved()
    /
    (1024 ** 3)
)


# ============================================================
# TRAIN METRICS
# ============================================================

train_metrics = {}


if train_result is not None:

    train_metrics.update(
        train_result.metrics
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "experiment":
        "run5",

    "dataset":
        "llama_extraction_v5_1",

    "dataset_version":
        DATASET_VERSION,

    "schema_type":
        SCHEMA_TYPE,

    "model":
        MODEL_NAME,

    "train_examples":
        len(train_dataset),

    "validation_examples":
        len(validation_dataset),

    "test_examples":
        len(test_dataset),

    "training": {

        "epochs":
            NUM_EPOCHS,

        "micro_batch_size":
            MICRO_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            EFFECTIVE_BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_ratio":
            WARMUP_RATIO,

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "lora_rank":
            LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "optimizer":
            "adamw_8bit",

        "resume":
            False
    },

    "parameters": {

        "total":
            total_parameters,

        "trainable":
            trainable_parameters,

        "trainable_percentage":
            trainable_percentage
    },

    "metrics": {

        "train":
            train_metrics,

        "validation":
            validation_metrics
    },

    "hardware": {

        "gpu":
            gpu_name,

        "total_vram_gb":
            gpu_memory_gb,

        "peak_allocated_vram_gb":
            peak_allocated_gb,

        "peak_reserved_vram_gb":
            peak_reserved_gb
    },

    "runtime": {

        "training_seconds":
            elapsed_seconds,

        "training_hours":
            elapsed_seconds / 3600
    },

    "artifacts": {

        "output_dir":
            OUTPUT_DIR,

        "final_adapter":
            FINAL_ADAPTER_DIR,

        "summary":
            SUMMARY_FILE
    },

    "test_used_during_training":
        False
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2,
        default=str
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("RUN 5 TRAINING COMPLETE")
print("=" * 80)
print()

print(
    f"Train loss          : "
    f"{train_metrics.get('train_loss', 'N/A')}"
)

print(
    f"Validation loss     : "
    f"{validation_metrics.get('eval_loss', 'N/A')}"
)

print(
    f"Peak allocated VRAM : "
    f"{peak_allocated_gb:.2f} GB"
)

print(
    f"Peak reserved VRAM  : "
    f"{peak_reserved_gb:.2f} GB"
)

print(
    f"Training time       : "
    f"{elapsed_seconds / 3600:.2f} hours"
)

print()

print(
    "FINAL ADAPTER:"
)

print(
    FINAL_ADAPTER_DIR
)

print()

print(
    "TRAINING SUMMARY:"
)

print(
    SUMMARY_FILE
)

print()

print("=" * 80)
print(
    "NEXT STEP: EVALUATE RUN 5 ON THE 888-EXAMPLE TEST SET"
)
print("=" * 80)