import os
import sys
import re
import json
import torch
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")
XTTS_OUTPUT_DIR = PROJECT_ROOT / "training" / "xtts_output"
REFERENCE_WAV = PROJECT_ROOT / "processed" / "hindi" / "hi_000621.wav"
EVALUATION_SENTENCES_FILE = PROJECT_ROOT / "training" / "evaluation" / "evaluation_sentences.txt"

def load_evaluation_sentences():
    sentences = []
    if not EVALUATION_SENTENCES_FILE.exists():
        raise FileNotFoundError(f"Evaluation sentences file not found: {EVALUATION_SENTENCES_FILE}")
    with open(EVALUATION_SENTENCES_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                category = parts[0].strip()
                lang = parts[1].strip()
                text = "|".join(parts[2:]).strip()
                sentences.append({
                    "id": f"eval_{idx+1:02d}",
                    "category": category,
                    "language": lang,
                    "text": text
                })
    return sentences

def find_run_dir(experiment_prefix):
    if experiment_prefix == "B0":
        return XTTS_OUTPUT_DIR / "rachana_xtts_clean_2epoch-August-27-2026_12+14PM-0000000"
        
    dirs = [d for d in XTTS_OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith(experiment_prefix)]
    if not dirs:
        raise FileNotFoundError(f"No run directory found for prefix {experiment_prefix}")
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]

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
        for l in reversed(lines):
            if "| > avg_loss:" in l:
                val_loss = l.split(":")[-1].strip()
                break
        for l in reversed(lines):
            if "| > loss:" in l and "(" in l:
                train_loss = l.split(":")[-1].strip()
                break
    return train_loss, val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment ID (e.g., B0, E1, E2)")
    parser.add_argument("--dir_prefix", type=str, required=True, help="Directory prefix (e.g., rachana_xtts_e1_english_fix)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EVALUATION SCRIPT FOR {args.exp}")
    print("=" * 70)

    run_dir = find_run_dir(args.dir_prefix)
    checkpoint_path = find_best_checkpoint(run_dir)
    config_path = run_dir / "config.json"

    eval_dir = PROJECT_ROOT / "training" / "xtts_evaluations" / args.exp
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_file = eval_dir / "report.txt"

    print(f"Run Directory    : {run_dir}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Output Directory : {eval_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device     : {device}")

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(checkpoint_path), eval=True)
    model.to(device)

    eval_tests = load_evaluation_sentences()

    results = []
    print("\nSynthesizing evaluation sentences...")
    for test in eval_tests:
        out_wav = eval_dir / f"{test['id']}.wav"
        print(f"\nSynthesizing [{test['category']}] -> {out_wav.name}")
        
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

        results.append({
            "test": test,
            "file": out_wav.name,
            "duration": duration,
            "size_mb": size_mb
        })

    train_loss, val_loss = parse_losses(run_dir)

    report = []
    report.append(f"==========================================================================")
    report.append(f"                        REPORT FOR EXPERIMENT {args.exp}                  ")
    report.append(f"==========================================================================")
    report.append("")
    report.append("1. EXPERIMENT SUMMARY")
    report.append(f"   Experiment Name : {args.exp}")
    report.append(f"   Run Directory   : {run_dir}")
    report.append(f"   Checkpoint Used : {checkpoint_path}")
    report.append("")
    report.append("2. TRAINING & VALIDATION METRICS")
    report.append(f"   Final Training Loss     : {train_loss}")
    report.append(f"   Final Validation Loss   : {val_loss}")
    report.append("")
    report.append("3. GENERATED EVALUATION AUDIO FILES")
    report.append(f"{'Filename':<15} {'Category':<35} {'Lang':<6} {'Duration':<10}")
    report.append("-" * 70)
    for r in results:
        t = r["test"]
        report.append(f"{r['file']:<15} {t['category']:<35} {t['language']:<6} {r['duration']:<5.2f} sec")
    report.append("")
    
    if args.exp == "E1":
        report.append("4. SCORING TABLE (B0 vs E1)")
        report.append("| Category | B0 | E1 | Difference |")
        report.append("| --- | --- | --- | --- |")
        report.append("| Hindi | /5 | /5 | |")
        report.append("| Marathi | /5 | /5 | |")
        report.append("| English | /5 | /5 | |")
        report.append("| Hinglish | /5 | /5 | |")
        report.append("| Marnglish | /5 | /5 | |")
        report.append("| English technical terms | /5 | /5 | |")
        report.append("")
        
    report_str = "\n".join(report)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)

    print("\n" + report_str)
    print("=" * 70)
    print(f"Report saved to: {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
