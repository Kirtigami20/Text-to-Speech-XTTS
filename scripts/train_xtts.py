import os
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)
from TTS.utils.manage import ModelManager


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAINING_DIR = ROOT / "training"
OUTPUT_DIR = ROOT / "training" / "xtts_run"
MODEL_FILES = ROOT / "training" / "xtts_model_files"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILES.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

DATASET_CONFIG = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="multilingual_female",
    path=str(TRAINING_DIR),
    meta_file_train="train.csv",
    meta_file_val="validation.csv",
)

DATASETS_CONFIG_LIST = [DATASET_CONFIG]


# ============================================================
# XTTS MODEL FILES
# ============================================================

TOKENIZER_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
)

XTTS_CHECKPOINT_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
)

DVAE_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
)

MEL_NORM_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
)


TOKENIZER_FILE = MODEL_FILES / "vocab.json"
XTTS_CHECKPOINT = MODEL_FILES / "model.pth"
DVAE_CHECKPOINT = MODEL_FILES / "dvae.pth"
MEL_NORM_FILE = MODEL_FILES / "mel_stats.pth"


# ============================================================
# DOWNLOAD REQUIRED MODEL FILES
# ============================================================

files_to_download = []

if not TOKENIZER_FILE.exists():
    files_to_download.append(TOKENIZER_URL)

if not XTTS_CHECKPOINT.exists():
    files_to_download.append(XTTS_CHECKPOINT_URL)

if not DVAE_CHECKPOINT.exists():
    files_to_download.append(DVAE_URL)

if not MEL_NORM_FILE.exists():
    files_to_download.append(MEL_NORM_URL)


if files_to_download:
    print("=" * 70)
    print("DOWNLOADING XTTS MODEL FILES")
    print("=" * 70)

    ModelManager._download_model_files(
        files_to_download,
        str(MODEL_FILES),
        progress_bar=True,
    )


# ============================================================
# VERIFY FILES
# ============================================================

required_files = [
    TOKENIZER_FILE,
    XTTS_CHECKPOINT,
    DVAE_CHECKPOINT,
    MEL_NORM_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required XTTS file not found: {file}"
        )


# ============================================================
# TRAINING SETTINGS
# ============================================================

RUN_NAME = "multilingual_female_xtts"
PROJECT_NAME = "multilingual_female_tts"

# RTX 3070 8GB:
# Start conservatively.
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 84

EPOCHS = 10


# ============================================================
# GPT MODEL ARGUMENTS
# ============================================================

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,

    debug_loading_failures=True,

    max_wav_length=255995,
    max_text_length=200,

    mel_norm_file=str(MEL_NORM_FILE),
    dvae_checkpoint=str(DVAE_CHECKPOINT),
    xtts_checkpoint=str(XTTS_CHECKPOINT),
    tokenizer_file=str(TOKENIZER_FILE),

    # XTTS-v2 settings
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
# TRAINER CONFIG
# ============================================================

config = GPTTrainerConfig(
    output_path=str(OUTPUT_DIR),

    model_args=model_args,

    run_name=RUN_NAME,
    project_name=PROJECT_NAME,

    run_description="Multilingual female XTTS fine-tuning",

    dashboard_logger="tensorboard",
    logger_uri=None,

    audio=audio_config,

    # Training
    epochs=EPOCHS,

    batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,

    batch_group_size=48,

    num_loader_workers=0,
    num_eval_loader_workers=0,

    # Evaluation
    eval_split_max_size=174,
    eval_split_size=0.1,

    # Logging
    print_step=25,
    plot_step=100,

    # Checkpoints
    save_step=500,
    save_n_checkpoints=3,
    save_checkpoints=True,

    # Optimizer
    optimizer="AdamW",

    optimizer_wd_only_on_weights=True,

    optimizer_params={
        "betas": [0.9, 0.96],
        "eps": 1e-8,
        "weight_decay": 1e-2,
    },

    lr=5e-6,

    # Mixed precision
    mixed_precision=True,
    precision="fp16",

    # Languages
    languages=["en", "hi", "mr"],

    # Disable unnecessary preprocessing
    use_phonemes=False,

    # Test sentences
    test_sentences=[],
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INITIALIZING XTTS")
print("=" * 70)

model = GPTTrainer.init_from_config(config)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=False,
)


print(f"Training samples : {len(train_samples)}")
print(f"Validation samples: {len(eval_samples)}")


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    ),

    config,

    output_path=str(OUTPUT_DIR),

    model=model,

    train_samples=train_samples,
    eval_samples=eval_samples,
)


# ============================================================
# START TRAINING
# ============================================================

print("=" * 70)
print("STARTING XTTS FINE-TUNING")
print("=" * 70)

trainer.fit()


print("=" * 70)
print("XTTS TRAINING COMPLETE")
print("=" * 70)