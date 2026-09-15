import os
import csv
import gc

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)

from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TRAIN_FINAL = os.path.join(
    PROJECT_ROOT,
    "training",
    "train_metadata_final.csv",
)

VAL_FINAL = os.path.join(
    PROJECT_ROOT,
    "training",
    "validation_metadata_final.csv",
)

AUDIO_ROOT = os.path.join(
    PROJECT_ROOT,
    "processed_clean",
)

TRAIN_CSV = os.path.join(
    PROJECT_ROOT,
    "training",
    "train_metadata_xtts.csv",
)

VAL_CSV = os.path.join(
    PROJECT_ROOT,
    "training",
    "validation_metadata_xtts.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "xtts_output",
)

CHECKPOINTS_PATH = os.path.join(
    OUTPUT_PATH,
    "XTTS_v2_original_model_files",
)

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 2

BATCH_SIZE = 2

GRAD_ACCUM_STEPS = 8


# ============================================================
# CHECK DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FINAL DATASET")
print("=" * 70)

print("TRAIN:")
print(TRAIN_FINAL)

print("\nVALIDATION:")
print(VAL_FINAL)

print("\nAUDIO:")
print(AUDIO_ROOT)


for path in [
    TRAIN_FINAL,
    VAL_FINAL,
    AUDIO_ROOT,
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired path not found:\n{path}"
        )


print("\nAll final dataset paths found.")


# ============================================================
# CONVERT METADATA
# ============================================================

def create_xtts_metadata(
    source_csv,
    destination_csv,
):
    """
    Convert:

        audio_file|text|speaker_name

    into:

        audio_file|text

    for Coqui's built-in coqui formatter.
    """

    print("\nConverting metadata:")
    print("SOURCE:", source_csv)
    print("OUTPUT:", destination_csv)

    count = 0

    with open(
        source_csv,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as src:

        reader = csv.DictReader(
            src,
            delimiter="|",
        )

        required = {
            "audio_file",
            "text",
            "speaker_name",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):

            raise ValueError(
                "\nUnexpected metadata columns.\n"
                f"Found: {reader.fieldnames}\n"
                "Expected: "
                "audio_file|text|speaker_name"
            )

        with open(
            destination_csv,
            "w",
            encoding="utf-8",
            newline="",
        ) as dst:

            writer = csv.writer(
                dst,
                delimiter="|",
            )

            writer.writerow(
                [
                    "audio_file",
                    "text",
                ]
            )

            for row in reader:

                audio_file = (
                    row["audio_file"].strip()
                )

                text = row["text"].strip()

                if not audio_file:
                    continue

                if not text:
                    continue

                # Make sure audio exists.
                full_audio = os.path.join(
                    PROJECT_ROOT,
                    audio_file,
                )

                if not os.path.isfile(
                    full_audio
                ):

                    print(
                        "WARNING - missing audio:",
                        full_audio,
                    )

                    continue

                writer.writerow(
                    [
                        audio_file,
                        text,
                    ]
                )

                count += 1

    print(
        f"Converted samples: {count}"
    )

    return count


print("\n" + "=" * 70)
print("PREPARING XTTS METADATA")
print("=" * 70)


train_count = create_xtts_metadata(
    TRAIN_FINAL,
    TRAIN_CSV,
)

val_count = create_xtts_metadata(
    VAL_FINAL,
    VAL_CSV,
)


print("\nTraining samples :", train_count)
print("Validation samples:", val_count)


if train_count == 0:
    raise RuntimeError(
        "No training samples available."
    )

if val_count == 0:
    raise RuntimeError(
        "No validation samples available."
    )


# ============================================================
# XTTS ORIGINAL FILES
# ============================================================

DVAE_CHECKPOINT_LINK = (
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
)

MEL_NORM_LINK = (
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
)

TOKENIZER_FILE_LINK = (
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
)

XTTS_CHECKPOINT_LINK = (
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
)

XTTS_CONFIG_LINK = (
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json"
)


DVAE_CHECKPOINT = os.path.join(
    CHECKPOINTS_PATH,
    "dvae.pth",
)

MEL_NORM_FILE = os.path.join(
    CHECKPOINTS_PATH,
    "mel_stats.pth",
)

TOKENIZER_FILE = os.path.join(
    CHECKPOINTS_PATH,
    "vocab.json",
)

XTTS_CHECKPOINT = os.path.join(
    CHECKPOINTS_PATH,
    "model.pth",
)

XTTS_CONFIG_FILE = os.path.join(
    CHECKPOINTS_PATH,
    "config.json",
)


# ============================================================
# CHECK XTTS FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ORIGINAL XTTS-v2 FILES")
print("=" * 70)


download_files = []


if not os.path.isfile(DVAE_CHECKPOINT):
    download_files.append(
        DVAE_CHECKPOINT_LINK
    )

if not os.path.isfile(MEL_NORM_FILE):
    download_files.append(
        MEL_NORM_LINK
    )

if not os.path.isfile(TOKENIZER_FILE):
    download_files.append(
        TOKENIZER_FILE_LINK
    )

if not os.path.isfile(XTTS_CHECKPOINT):
    download_files.append(
        XTTS_CHECKPOINT_LINK
    )

if not os.path.isfile(XTTS_CONFIG_FILE):
    download_files.append(
        XTTS_CONFIG_LINK
    )


if download_files:

    print(
        "\nDownloading missing XTTS-v2 files..."
    )

    ModelManager._download_model_files(
        download_files,
        CHECKPOINTS_PATH,
        progress_bar=True,
    )

else:

    print(
        "All original XTTS-v2 files already exist."
    )


# ============================================================
# DATASET CONFIG
# ============================================================

dataset_config = BaseDatasetConfig(

    formatter="coqui",

    dataset_name="rachana",

    path=PROJECT_ROOT,

    meta_file_train=TRAIN_CSV,

    meta_file_val=VAL_CSV,

    language="hi",
)


DATASETS_CONFIG_LIST = [
    dataset_config
]


# ============================================================
# GPT ARGUMENTS
# ============================================================

model_args = GPTArgs(

    max_conditioning_length=132300,

    min_conditioning_length=66150,

    debug_loading_failures=False,

    max_wav_length=255995,

    max_text_length=200,

    mel_norm_file=MEL_NORM_FILE,

    dvae_checkpoint=DVAE_CHECKPOINT,

    xtts_checkpoint=XTTS_CHECKPOINT,

    tokenizer_file=TOKENIZER_FILE,

    gpt_num_audio_tokens=1026,

    gpt_start_audio_token=1024,

    gpt_stop_audio_token=1025,

    gpt_use_masking_gt_prompt_approach=True,

    gpt_use_perceiver_resampler=True,
)


# ============================================================
# AUDIO CONFIG
# ============================================================

audio_config = XttsAudioConfig(

    sample_rate=22050,

    dvae_sample_rate=22050,

    output_sample_rate=24000,
)


# ============================================================
# TRAINING CONFIG
# ============================================================

config = GPTTrainerConfig(

    epochs=NUM_EPOCHS,

    output_path=OUTPUT_PATH,

    model_args=model_args,

    run_name="rachana_xtts_clean_2epoch",

    project_name="rachana_xtts",

    run_description=(
        "Rachana multilingual XTTS fine tuning "
        "Hindi Marathi English mixed speech"
    ),

    dashboard_logger="tensorboard",

    audio=audio_config,

    batch_size=BATCH_SIZE,

    eval_batch_size=BATCH_SIZE,

    batch_group_size=48,

    num_loader_workers=0,

    num_eval_loader_workers=0,

    eval_split_max_size=169,

    print_step=10,

    plot_step=100,

    log_model_step=100,

    save_step=500,

    save_n_checkpoints=2,

    save_checkpoints=True,

    print_eval=False,

    optimizer="AdamW",

    optimizer_wd_only_on_weights=True,

    optimizer_params={
        "betas": [0.9, 0.96],
        "eps": 1e-8,
        "weight_decay": 1e-2,
    },

    lr=5e-6,

    lr_scheduler="MultiStepLR",

    lr_scheduler_params={
        "milestones": [
            50000 * 18,
            150000 * 18,
            300000 * 18,
        ],
        "gamma": 0.5,
        "last_epoch": -1,
    },

    test_sentences=[],
)


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING XTTS TRAINER")
print("=" * 70)


model = GPTTrainer.init_from_config(
    config
)


print(
    "Model initialized successfully."
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINAL DATASET")
print("=" * 70)


print(
    "Training metadata:",
    TRAIN_CSV,
)

print(
    "Validation metadata:",
    VAL_CSV,
)

print(
    "Audio:",
    AUDIO_ROOT,
)


train_samples, eval_samples = load_tts_samples(

    DATASETS_CONFIG_LIST,

    eval_split=True,

    eval_split_max_size=(
        config.eval_split_max_size
    ),

    eval_split_size=(
        config.eval_split_size
    ),
)


print("\n" + "-" * 70)

print(
    "TRAIN SAMPLES:",
    len(train_samples),
)

print(
    "VALIDATION SAMPLES:",
    len(eval_samples),
)

print("-" * 70)


if not train_samples:

    raise RuntimeError(
        "No training samples loaded."
    )


if not eval_samples:

    raise RuntimeError(
        "No validation samples loaded."
    )


# ============================================================
# CREATE TRAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINER")
print("=" * 70)


trainer = Trainer(

    TrainerArgs(

        # Fresh training from original XTTS-v2
        restore_path=None,

        skip_train_epoch=False,

        start_with_eval=False,

        grad_accum_steps=GRAD_ACCUM_STEPS,
    ),

    config,

    output_path=OUTPUT_PATH,

    model=model,

    train_samples=train_samples,

    eval_samples=eval_samples,
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING XTTS TRAINING")
print("=" * 70)


print(
    "Epochs              :",
    NUM_EPOCHS,
)

print(
    "Batch size          :",
    BATCH_SIZE,
)

print(
    "Gradient accumulation:",
    GRAD_ACCUM_STEPS,
)

print(
    "Effective batch     :",
    BATCH_SIZE * GRAD_ACCUM_STEPS,
)

print(
    "Training samples    :",
    len(train_samples),
)

print(
    "Validation samples  :",
    len(eval_samples),
)

print(
    "Dataset             : "
    "Hindi + Marathi + English + Mixed"
)

print(
    "\nStarting from original XTTS-v2."
)

print("=" * 70)


trainer.fit()


# ============================================================
# CLEANUP
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)


del model
del trainer
del train_samples
del eval_samples

gc.collect()


print("\nDone.")