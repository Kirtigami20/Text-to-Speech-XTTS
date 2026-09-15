import os
import csv
import gc
import sys
from pathlib import Path

# Force UTF-8 stdout & unbuffered printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

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

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")

TRAIN_FINAL = PROJECT_ROOT / "training" / "train_metadata_final.csv"
VAL_FINAL = PROJECT_ROOT / "training" / "validation_metadata_final.csv"

AUDIO_ROOT = PROJECT_ROOT / "processed_clean"

# E1-specific metadata files (separated by language ID)
TRAIN_HI_CSV = PROJECT_ROOT / "training" / "train_metadata_e1_hi.csv"
TRAIN_EN_CSV = PROJECT_ROOT / "training" / "train_metadata_e1_en.csv"
VAL_HI_CSV = PROJECT_ROOT / "training" / "validation_metadata_e1_hi.csv"
VAL_EN_CSV = PROJECT_ROOT / "training" / "validation_metadata_e1_en.csv"

OUTPUT_PATH = PROJECT_ROOT / "training" / "xtts_output" / "rachana_xtts_e1_english_fix"
CHECKPOINTS_PATH = PROJECT_ROOT / "training" / "xtts_output" / "XTTS_v2_original_model_files"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# TRAINING HYPERPARAMETERS (MUST MATCH BASELINE EXACTLY)
# ============================================================

NUM_EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-6

# ============================================================
# E1 METADATA PREPARATION & PRE-TRAINING CHECKS
# ============================================================

def prepare_e1_metadata(src_csv, dst_hi_csv, dst_en_csv, split_name):
    """
    Reads src_csv (audio_file|text|speaker_name)
    Splits into:
      - dst_hi_csv (for Hindi & Marathi samples, tagged 'hi')
      - dst_en_csv (for English samples, tagged 'en')
    Verifies no text modifications occur.
    """
    log_print(f"\nProcessing {split_name} metadata for E1...")
    
    hi_rows = []
    en_rows = []
    
    hindi_count = 0
    marathi_count = 0
    english_count = 0
    
    with open(src_csv, "r", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src, delimiter="|")
        for row in reader:
            audio_file = row["audio_file"].strip()
            text = row["text"].strip()
            
            if not audio_file or not text:
                continue
            
            # Verify file exists
            full_audio = PROJECT_ROOT / audio_file
            if not full_audio.is_file():
                log_print(f"WARNING - Missing audio file: {full_audio}")
                continue

            audio_lower = audio_file.lower()
            if "english" in audio_lower or "/en_" in audio_lower or "\\en_" in audio_lower:
                english_count += 1
                en_rows.append([audio_file, text])
            elif "marathi" in audio_lower or "/mr_" in audio_lower or "\\mr_" in audio_lower:
                marathi_count += 1
                hi_rows.append([audio_file, text])
            else:
                hindi_count += 1
                hi_rows.append([audio_file, text])
                
    # Write HI/MR CSV
    with open(dst_hi_csv, "w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows(hi_rows)
        
    # Write EN CSV
    with open(dst_en_csv, "w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows(en_rows)
        
    log_print(f"  {split_name} Summary:")
    log_print(f"    Hindi samples   -> language='hi': {hindi_count}")
    log_print(f"    Marathi samples -> language='hi': {marathi_count}")
    log_print(f"    English samples -> language='en': {english_count}")
    log_print(f"    Total {split_name} samples: {hindi_count + marathi_count + english_count}")
    
    return {
        "hindi": hindi_count,
        "marathi": marathi_count,
        "english": english_count,
        "total": hindi_count + marathi_count + english_count
    }

log_print("=" * 70)
log_print("EXPERIMENT E1: ENGLISH LANGUAGE ID FIX")
log_print("=" * 70)

train_stats = prepare_e1_metadata(TRAIN_FINAL, TRAIN_HI_CSV, TRAIN_EN_CSV, "TRAIN")
val_stats = prepare_e1_metadata(VAL_FINAL, VAL_HI_CSV, VAL_EN_CSV, "VALIDATION")

# Pre-training text integrity verification
def verify_text_integrity():
    log_print("\nVerifying transcript integrity against baseline metadata...")
    for src, hi_csv, en_csv in [(TRAIN_FINAL, TRAIN_HI_CSV, TRAIN_EN_CSV), (VAL_FINAL, VAL_HI_CSV, VAL_EN_CSV)]:
        original_map = {}
        with open(src, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter="|"):
                original_map[r["audio_file"].strip()] = r["text"].strip()
                
        e1_map = {}
        for c in [hi_csv, en_csv]:
            with open(c, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="|"):
                    e1_map[r["audio_file"].strip()] = r["text"].strip()
                    
        assert len(original_map) == len(e1_map), f"Mismatch in sample count for {src.name}"
        for k, v in original_map.items():
            assert k in e1_map, f"Missing key {k} in E1 metadata"
            assert e1_map[k] == v, f"Transcript mismatch for {k}"
            
    log_print("Verification Passed: 100% transcript text matching baseline!")

verify_text_integrity()

# ============================================================
# ORIGINAL XTTS FILES CHECK
# ============================================================

DVAE_CHECKPOINT = CHECKPOINTS_PATH / "dvae.pth"
MEL_NORM_FILE = CHECKPOINTS_PATH / "mel_stats.pth"
TOKENIZER_FILE = CHECKPOINTS_PATH / "vocab.json"
XTTS_CHECKPOINT = CHECKPOINTS_PATH / "model.pth"

for path in [DVAE_CHECKPOINT, MEL_NORM_FILE, TOKENIZER_FILE, XTTS_CHECKPOINT]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required XTTS original file: {path}")

# ============================================================
# DATASET CONFIGS FOR E1
# ============================================================

dataset_config_hi = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="rachana_hi_mr",
    path=str(PROJECT_ROOT),
    meta_file_train=str(TRAIN_HI_CSV),
    meta_file_val=str(VAL_HI_CSV),
    language="hi",
)

dataset_config_en = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="rachana_en",
    path=str(PROJECT_ROOT),
    meta_file_train=str(TRAIN_EN_CSV),
    meta_file_val=str(VAL_EN_CSV),
    language="en",
)

DATASETS_CONFIG_LIST = [dataset_config_hi, dataset_config_en]

# ============================================================
# MODEL & TRAINING CONFIGURATION
# ============================================================

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    debug_loading_failures=False,
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file=str(MEL_NORM_FILE),
    dvae_checkpoint=str(DVAE_CHECKPOINT),
    xtts_checkpoint=str(XTTS_CHECKPOINT),
    tokenizer_file=str(TOKENIZER_FILE),
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(
    sample_rate=22050,
    dvae_sample_rate=22050,
    output_sample_rate=24000,
)

config = GPTTrainerConfig(
    epochs=NUM_EPOCHS,
    output_path=str(OUTPUT_PATH.parent),
    model_args=model_args,
    run_name=OUTPUT_PATH.name,
    project_name="rachana_xtts_e1",
    run_description="E1: English Language ID Fix (en for English, hi for Hindi/Marathi)",
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
    lr=LEARNING_RATE,
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={
        "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
        "gamma": 0.5,
        "last_epoch": -1,
    },
    test_sentences=[],
)

# Initialize Model
log_print("\nInitializing GPTTrainer for E1...")
model = GPTTrainer.init_from_config(config)

# Load Dataset Samples
log_print("\nLoading dataset samples...")
train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

log_print("-" * 70)
log_print(f"Total Train Samples Loaded : {len(train_samples)}")
log_print(f"Total Eval Samples Loaded  : {len(eval_samples)}")
log_print("-" * 70)

# Verify Language breakdown in loaded samples
train_lang_counts = {}
for s in train_samples:
    l = s.get("language", "unknown")
    train_lang_counts[l] = train_lang_counts.get(l, 0) + 1

log_print("Loaded Train Samples Language Distribution:")
for l, c in train_lang_counts.items():
    log_print(f"  language='{l}': {c}")

# Create Trainer
trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    ),
    config,
    output_path=str(OUTPUT_PATH.parent),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

log_print("\n" + "=" * 70)
log_print("STARTING E1 TRAINING")
log_print("=" * 70)

trainer.fit()

log_print("\n" + "=" * 70)
log_print("E1 TRAINING COMPLETED SUCCESSFULLY")
log_print("=" * 70)

del model
del trainer
del train_samples
del eval_samples
gc.collect()
