import os
import sys
import re
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")
XTTS_OUTPUT_DIR = PROJECT_ROOT / "training" / "xtts_output"
EVAL_DIR = PROJECT_ROOT / "training" / "xtts_evaluations" / "E1"
REPORT_FILE = EVAL_DIR / "e1_report.txt"
REFERENCE_WAV = PROJECT_ROOT / "processed" / "hindi" / "hi_000621.wav"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

def find_e1_run():
    e1_dirs = [d for d in XTTS_OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("rachana_xtts_e1_english_fix")]
    if not e1_dirs:
        raise FileNotFoundError("No E1 run directory found in training/xtts_output/")
    e1_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return e1_dirs[0]

def find_best_checkpoint(run_dir):
    checkpoints = list(run_dir.glob("best_model*.pth"))
    if not checkpoints:
        checkpoints = list(run_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoint found in {run_dir}")
    checkpoints.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return checkpoints[0]

def parse_losses(run_dir):
    log_file = run_dir / "trainer_0_log.txt"
    train_loss = "Unknown"
    val_loss = "Unknown"
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', l).strip() for l in f]
        
        # Search for evaluation loss
        for l in reversed(lines):
            if "| > avg_loss:" in l:
                val_loss = l.split(":")[-1].strip()
                break
        
        # Search for final training loss
        for l in reversed(lines):
            if "| > loss:" in l and "(" in l:
                train_loss = l.split(":")[-1].strip()
                break

    return train_loss, val_loss

def main():
    print("=" * 70)
    print("EXPERIMENT E1 EVALUATION & REPORT GENERATOR")
    print("=" * 70)

    run_dir = find_e1_run()
    checkpoint_path = find_best_checkpoint(run_dir)
    config_path = run_dir / "config.json"

    print(f"E1 Run Directory : {run_dir}")
    print(f"E1 Checkpoint    : {checkpoint_path}")
    print(f"E1 Config        : {config_path}")
    print(f"Reference Audio  : {REFERENCE_WAV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device     : {device}")

    # Load Config & Model
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(checkpoint_path), eval=True)
    model.to(device)

    # Evaluation sentences
    eval_tests = [
        {
            "id": "01_hindi",
            "category": "Native Hindi",
            "language": "hi",
            "text": "नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में बात करेंगे।"
        },
        {
            "id": "02_english",
            "category": "Native English",
            "language": "en",
            "text": "Hello, today we are going to talk about income tax and finance."
        },
        {
            "id": "03_hinglish",
            "category": "Hinglish",
            "language": "hi",
            "text": "Aaj hum income tax ke baare mein baat karenge aur dekhenge ki tax planning kaise karni hai."
        },
        {
            "id": "04_hindi_english",
            "category": "Hinglish (Mixed)",
            "language": "hi",
            "text": "Aapko apna ITR file karna hai, aur uske baad tax calculation properly check karni hai."
        },
        {
            "id": "05_marathi",
            "category": "Native Marathi",
            "language": "hi",
            "text": "नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल माहिती घेणार आहोत."
        },
        {
            "id": "06_marnglish",
            "category": "Marnglish",
            "language": "hi",
            "text": "Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot."
        },
        {
            "id": "07_marnglish_natural",
            "category": "Marnglish (Natural)",
            "language": "hi",
            "text": "Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara."
        },
        {
            "id": "08_indian_english",
            "category": "Indian English",
            "language": "en",
            "text": "If you are filing your income tax return in India, you should carefully check all your income details."
        }
    ]

    print("\nSynthesizing evaluation sentences...")
    results = []

    for test in eval_tests:
        out_wav = EVAL_DIR / f"{test['id']}.wav"
        print(f"\nSynthesizing [{test['category']}] -> {out_wav.name}")
        print(f"  Language parameter : '{test['language']}'")
        print(f"  Text               : {test['text']}")

        res = model.synthesize(
            text=test["text"],
            config=config,
            speaker_wav=str(REFERENCE_WAV),
            language=test["language"],
            gpt_cond_len=12,
            temperature=0.65,
            enable_text_splitting=True,
        )

        wav = res["wav"]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        wav = np.squeeze(np.asarray(wav)).astype(np.float32)

        sr = 24000
        sf.write(str(out_wav), wav, sr)
        duration = len(wav) / sr
        size_mb = out_wav.stat().st_size / (1024 * 1024)

        print(f"  Duration : {duration:.2f} s | File Size: {size_mb:.2f} MB")

        results.append({
            "test": test,
            "file": out_wav.name,
            "duration": duration,
            "size_mb": size_mb
        })

    # Parse losses
    train_loss, val_loss = parse_losses(run_dir)

    # Build E1 Report
    report = []
    report.append("==========================================================================")
    report.append("                  EXPERIMENT E1 EVALUATION REPORT                         ")
    report.append("==========================================================================")
    report.append("")
    report.append("1. EXPERIMENT SUMMARY")
    report.append("   Objective       : Isolate the effect of correcting English language ID (en).")
    report.append("   Status          : COMPLETED")
    report.append(f"   Run Directory   : {run_dir}")
    report.append(f"   Checkpoint Used : {checkpoint_path}")
    report.append("")
    report.append("2. EXACT TRAINING PARAMETERS")
    report.append("   Learning Rate (lr)      : 5e-6")
    report.append("   Epochs                  : 2")
    report.append("   Batch Size              : 2")
    report.append("   Gradient Accumulation   : 8")
    report.append("   Effective Batch Size    : 16")
    report.append("   Optimizer               : AdamW (betas=[0.9, 0.96], eps=1e-8, weight_decay=1e-2)")
    report.append("   LR Scheduler            : MultiStepLR")
    report.append("")
    report.append("3. LANGUAGE MAPPING LOGIC")
    report.append("   Hindi Samples   (hi_*) -> language = 'hi'")
    report.append("   Marathi Samples (mr_*) -> language = 'hi' (Native XTTS lacks 'mr')")
    report.append("   English Samples (en_*) -> language = 'en' (CORRECTED IN E1)")
    report.append("")
    report.append("4. TRAINING & VALIDATION METRICS")
    report.append(f"   Final Training Loss     : {train_loss}")
    report.append(f"   Final Validation Loss   : {val_loss}")
    report.append("")
    report.append("5. GENERATED EVALUATION AUDIO FILES")
    report.append(f"   Output Directory        : {EVAL_DIR}")
    report.append("")
    report.append(f"{'Filename':<25} {'Category':<22} {'Lang':<6} {'Duration':<10} {'Size MB'}")
    report.append("-" * 75)
    for r in results:
        t = r["test"]
        report.append(f"{r['file']:<25} {t['category']:<22} {t['language']:<6} {r['duration']:<5.2f} sec   {r['size_mb']:.2f} MB")
    report.append("")
    report.append("6. EVALUATION SENTENCES & TEXTS")
    for r in results:
        t = r["test"]
        report.append(f"   [{r['file']}] Category: {t['category']} | Lang Param: '{t['language']}'")
        report.append(f"   Text: {t['text']}")
        report.append("")
    report.append("==========================================================================")

    report_str = "\n".join(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_str)

    print("\n" + report_str)
    print("\n" + "=" * 70)
    print(f"Report saved to: {REPORT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
