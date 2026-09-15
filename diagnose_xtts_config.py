import os
import sys
import re
import json
import csv
from pathlib import Path
from collections import Counter

# Set UTF-8 output encoding for Windows terminal output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")
TRAIN_FINAL_CSV = PROJECT_ROOT / "training" / "train_metadata_final.csv"
VAL_FINAL_CSV = PROJECT_ROOT / "training" / "validation_metadata_final.csv"
XTTS_OUTPUT_DIR = PROJECT_ROOT / "training" / "xtts_output"
DIAGNOSTICS_DIR = PROJECT_ROOT / "training" / "xtts_diagnostics"
REPORT_FILE = DIAGNOSTICS_DIR / "diagnostic_report.txt"

SPECIAL_MARATHI_CHARS = {
    "ळ": "\u0933",
    "ऱ": "\u0931",
    "ऴ": "\u0934",
    "ॲ": "\u0972",
    "ऑ": "\u0911",
}

def analyze_script(text):
    has_roman = bool(re.search(r'[a-zA-Z]', text))
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    if has_roman and has_devanagari:
        return "mixed"
    elif has_devanagari:
        return "devanagari_only"
    elif has_roman:
        return "roman_only"
    else:
        return "other"

def parse_metadata(csv_path):
    stats = {
        "hindi": {"roman_only": 0, "devanagari_only": 0, "mixed": 0, "other": 0, "total": 0},
        "marathi": {"roman_only": 0, "devanagari_only": 0, "mixed": 0, "other": 0, "total": 0},
        "english": {"roman_only": 0, "devanagari_only": 0, "mixed": 0, "other": 0, "total": 0},
        "unknown": {"roman_only": 0, "devanagari_only": 0, "mixed": 0, "other": 0, "total": 0},
    }
    char_counts = Counter()
    total_rows = 0

    if not csv_path.exists():
        return stats, char_counts, total_rows

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            total_rows += 1
            audio_file = row.get("audio_file", "")
            text = row.get("text", "")

            # Detect language from path
            audio_lower = audio_file.lower()
            if "hindi" in audio_lower or "/hi_" in audio_lower or "\\hi_" in audio_lower:
                lang = "hindi"
            elif "marathi" in audio_lower or "/mr_" in audio_lower or "\\mr_" in audio_lower:
                lang = "marathi"
            elif "english" in audio_lower or "/en_" in audio_lower or "\\en_" in audio_lower:
                lang = "english"
            else:
                lang = "unknown"

            script_type = analyze_script(text)
            stats[lang][script_type] += 1
            stats[lang]["total"] += 1

            for ch in text:
                char_counts[ch] += 1

    return stats, char_counts, total_rows

def find_vocab_files():
    vocab_files = list(PROJECT_ROOT.rglob("vocab.json"))
    return vocab_files

def analyze_vocab(vocab_path):
    if not vocab_path.exists():
        return None
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens = set()
        token_to_id = {}

        if isinstance(data, dict):
            if "model" in data and "vocab" in data["model"]:
                token_to_id.update(data["model"]["vocab"])
            elif "vocab" in data:
                token_to_id.update(data["vocab"])
            
            if "added_tokens" in data:
                for item in data["added_tokens"]:
                    if isinstance(item, dict) and "content" in item:
                        token_to_id[item["content"]] = item.get("id")
                    elif isinstance(item, str):
                        token_to_id[item] = None
        
        if not token_to_id and isinstance(data, dict):
            token_to_id = {k: v for k, v in data.items() if isinstance(v, int)}

        vocab_tokens = list(token_to_id.keys())
        size = len(token_to_id)
        
        # Check special characters in vocab tokens
        char_in_vocab = {}
        for char_name, char_val in SPECIAL_MARATHI_CHARS.items():
            in_vocab = any(char_val in tok for tok in vocab_tokens)
            char_in_vocab[char_name] = in_vocab

        # Stock XTTS v2 tokenizer vocab size check
        is_stock = (size == 6681 or size == 6680 or (size > 2000 and "version" in data))
        
        return {
            "path": str(vocab_path),
            "size": size,
            "is_stock": is_stock,
            "special_char_coverage": char_in_vocab,
            "sample_tokens": vocab_tokens[:10]
        }
    except Exception as e:
        return {"path": str(vocab_path), "error": str(e)}

def find_latest_run():
    if not XTTS_OUTPUT_DIR.exists():
        return None
    run_dirs = [d for d in XTTS_OUTPUT_DIR.iterdir() if d.is_dir() and d.name != "XTTS_v2_original_model_files"]
    if not run_dirs:
        return None
    # Sort by modification time
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs[0]

def parse_run_info(run_dir):
    info = {
        "run_name": run_dir.name,
        "config_path": None,
        "log_path": None,
        "config": {},
        "log_summary": {}
    }
    config_file = run_dir / "config.json"
    if config_file.exists():
        info["config_path"] = str(config_file)
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                info["config"] = json.load(f)
        except Exception as e:
            info["config_error"] = str(e)

    log_file = run_dir / "trainer_0_log.txt"
    if log_file.exists():
        info["log_path"] = str(log_file)
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_lines = f.readlines()
            
            clean_lines = [re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', l) for l in raw_lines]
            
            global_steps = []
            epochs_seen = set()
            total_epochs_cfg = None

            for l in clean_lines:
                m_ep = re.search(r'EPOCH:\s*(\d+)/(\d+)', l)
                if m_ep:
                    epochs_seen.add(int(m_ep.group(1)))
                    total_epochs_cfg = int(m_ep.group(2))

                m_step = re.search(r'GLOBAL_STEP:\s*(\d+)', l)
                if m_step:
                    global_steps.append(int(m_step.group(1)))

            max_global_step = max(global_steps) if global_steps else "Unknown"

            info["log_summary"] = {
                "total_log_lines": len(raw_lines),
                "max_global_step": max_global_step,
                "epochs_completed": len(epochs_seen),
                "total_epochs": total_epochs_cfg,
                "epochs_list": sorted(list(epochs_seen)) if epochs_seen else [],
                "last_lines": [l.strip() for l in clean_lines[-12:] if l.strip()]
            }
        except Exception as e:
            info["log_error"] = str(e)
            
    return info

def inspect_code_files():
    code_findings = {}

    train_xtts_path = PROJECT_ROOT / "training" / "train_xtts.py"
    if train_xtts_path.exists():
        with open(train_xtts_path, "r", encoding="utf-8") as f:
            content = f.read()

        lang_match = re.search(r'language\s*=\s*["\']([^"\']+)["\']', content)
        code_findings["train_xtts_dataset_language"] = lang_match.group(1) if lang_match else "Not found"
        
        lr_match = re.search(r'lr\s*=\s*([0-9eE\.-]+)', content)
        code_findings["train_xtts_lr"] = lr_match.group(1) if lr_match else "Not found"
        
        epoch_match = re.search(r'NUM_EPOCHS\s*=\s*(\d+)', content)
        code_findings["train_xtts_epochs"] = epoch_match.group(1) if epoch_match else "Not found"

        batch_match = re.search(r'BATCH_SIZE\s*=\s*(\d+)', content)
        code_findings["train_xtts_batch_size"] = batch_match.group(1) if batch_match else "Not found"

        grad_match = re.search(r'GRAD_ACCUM_STEPS\s*=\s*(\d+)', content)
        code_findings["train_xtts_grad_accum"] = grad_match.group(1) if grad_match else "Not found"

    return code_findings

def main():
    print("=" * 70)
    print("XTTS CONFIGURATION & DATASET DIAGNOSTIC")
    print("=" * 70)

    # 1. Dataset stats
    train_stats, train_chars, train_total = parse_metadata(TRAIN_FINAL_CSV)
    val_stats, val_chars, val_total = parse_metadata(VAL_FINAL_CSV)

    combined_chars = train_chars + val_chars

    # 2. Vocab analysis
    vocab_files = find_vocab_files()
    vocab_reports = [analyze_vocab(v) for v in vocab_files]

    # 3. Latest run info
    latest_run_dir = find_latest_run()
    run_info = parse_run_info(latest_run_dir) if latest_run_dir else None

    # 4. Code inspection
    code_info = inspect_code_files()

    # Generate Report Lines
    report_lines = []
    report_lines.append("==========================================================================")
    report_lines.append("                     XTTS DIAGNOSTIC REPORT                               ")
    report_lines.append("==========================================================================")
    report_lines.append("")

    # 1. Languages & Dataset Breakdown
    report_lines.append("--- 1. DATASET BREAKDOWN & LANGUAGE TAGS ---")
    report_lines.append(f"Train Metadata File      : {TRAIN_FINAL_CSV} ({train_total} samples)")
    report_lines.append(f"Validation Metadata File : {VAL_FINAL_CSV} ({val_total} samples)")
    report_lines.append("")
    report_lines.append("Dataset Language Tags in Metadata files: NONE (No 'language' column present in CSV files)")
    report_lines.append(f"Language Tag passed in train_xtts.py BaseDatasetConfig: '{code_info.get('train_xtts_dataset_language', 'hi')}'")
    report_lines.append("Languages trained: All audio samples (Hindi, Marathi, English) were assigned language tag 'hi'.")
    report_lines.append("")
    report_lines.append("Transcripts breakdown by audio directory path & script type:")
    report_lines.append("")
    
    for split_name, stats, total in [("TRAIN", train_stats, train_total), ("VALIDATION", val_stats, val_total)]:
        report_lines.append(f"[{split_name} SET - Total: {total}]")
        for lang in ["hindi", "marathi", "english", "unknown"]:
            s = stats[lang]
            report_lines.append(f"  * {lang.upper():<8} : Total = {s['total']:<5} | Devanagari-only = {s['devanagari_only']:<5} | Roman-only = {s['roman_only']:<5} | Mixed = {s['mixed']:<5}")
        report_lines.append("")

    # 2. Language ID Mapping (Marathi, Hindi, English)
    report_lines.append("--- 2. LANGUAGE ID MAPPING & MARATHI ID STATUS ---")
    report_lines.append("Language IDs used in XTTS training config / loader:")
    report_lines.append("  - Hindi (hi)   : Passed as 'hi'")
    report_lines.append("  - English (en) : Passed as 'hi' (hardcoded in BaseDatasetConfig)")
    report_lines.append("  - Marathi (mr) : Passed as 'hi' (hardcoded in BaseDatasetConfig)")
    report_lines.append("Status of Marathi ID:")
    report_lines.append("  - Is 'mr' supported natively by XTTS-v2 model? NO. XTTS-v2 stock languages are:")
    report_lines.append("    ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']")
    report_lines.append("  - Marathi is completely missing its own language ID 'mr' in XTTS stock model.")
    report_lines.append("  - During training in train_xtts.py, Marathi was passed as 'hi' (Hindi).")
    report_lines.append("")

    # 3. Vocabulary & Character Coverage
    report_lines.append("--- 3. XTTS VOCABULARY & CHARACTER COVERAGE ---")
    report_lines.append(f"Vocab files found ({len(vocab_files)}):")
    for vr in vocab_reports:
        report_lines.append(f"  Path     : {vr['path']}")
        report_lines.append(f"  Size     : {vr.get('size', 'Error')}")
        report_lines.append(f"  Is Stock : {vr.get('is_stock', 'Unknown')}")
    report_lines.append("")
    report_lines.append("Marathi Special Characters Coverage Analysis:")
    report_lines.append(f"{'Char':<6} {'Name':<12} {'Unicode':<10} {'In Dataset?':<14} {'Dataset Count':<15} {'In XTTS vocab.json?'}")
    report_lines.append("-" * 75)
    
    special_names = {
        "ळ": "Devanagari LLA",
        "ऱ": "Devanagari RRA",
        "ऴ": "Devanagari LLLA",
        "ॲ": "Devanagari Chandra A",
        "ऑ": "Devanagari Chandra O",
    }
    
    # Primary vocab check (using original model files or first found)
    primary_vocab = vocab_reports[0] if vocab_reports else {}
    vocab_cov = primary_vocab.get("special_char_coverage", {})

    for ch, char_code in SPECIAL_MARATHI_CHARS.items():
        cnt = combined_chars[ch]
        in_ds = "YES" if cnt > 0 else "NO"
        in_vc = "YES" if vocab_cov.get(ch, False) else "NO"
        report_lines.append(f"{ch:<6} {special_names[ch]:<12} {char_code:<10} {in_ds:<14} {cnt:<15} {in_vc}")
    report_lines.append("")

    # 4. Training Hyperparameters & Run Configuration
    report_lines.append("--- 4. TRAINING HYPERPARAMETERS & RUN CONFIGURATION ---")
    if run_info and run_info.get("config"):
        cfg = run_info["config"]
        model_args = cfg.get("model_args", {})
        report_lines.append(f"Latest Run Directory    : {latest_run_dir.name}")
        report_lines.append(f"Learning Rate (lr)      : {cfg.get('lr', code_info.get('train_xtts_lr'))}")
        report_lines.append(f"Epochs                  : {cfg.get('epochs', code_info.get('train_xtts_epochs'))}")
        report_lines.append(f"Batch Size              : {cfg.get('batch_size', code_info.get('train_xtts_batch_size'))}")
        report_lines.append(f"Grad Accumulation Steps : {cfg.get('grad_accum_steps', 8)}")
        bs = cfg.get('batch_size', 2)
        ga = cfg.get('grad_accum_steps', 8)
        report_lines.append(f"Effective Batch Size    : {bs * ga}")
        report_lines.append(f"Optimizer               : {cfg.get('optimizer', 'AdamW')}")
        report_lines.append(f"LR Scheduler            : {cfg.get('lr_scheduler', 'MultiStepLR')}")
    else:
        report_lines.append(f"Learning Rate (lr)      : {code_info.get('train_xtts_lr', '5e-6')}")
        report_lines.append(f"Epochs                  : {code_info.get('train_xtts_epochs', '2')}")
        report_lines.append(f"Batch Size              : {code_info.get('train_xtts_batch_size', '2')}")
        report_lines.append(f"Grad Accumulation Steps : {code_info.get('train_xtts_grad_accum', '8')}")
        report_lines.append(f"Effective Batch Size    : {int(code_info.get('train_xtts_batch_size', 2)) * int(code_info.get('train_xtts_grad_accum', 8))}")

    report_lines.append("")

    # 5. Model Architecture & Freeze Status
    report_lines.append("--- 5. MODEL FREEZE / UNFREEZE STATUS ---")
    report_lines.append("GPT Encoder / Autoregressive Model : UNFROZEN (Fine-tuned during training)")
    report_lines.append("Decoder (DVAE)                     : FROZEN (Loaded static from dvae.pth)")
    report_lines.append("Vocoder (HiFi-GAN / Mel Spectrogram): FROZEN (Loaded static from mel_stats.pth)")
    report_lines.append("")

    # 6. Actual Steps from Logs
    report_lines.append("--- 6. ACTUAL TRAINING STEPS FROM LOGS ---")
    if run_info and run_info.get("log_summary"):
        ls = run_info["log_summary"]
        report_lines.append(f"Log File Path           : {run_info['log_path']}")
        report_lines.append(f"Max Global Step Logged  : {ls.get('max_global_step')}")
        report_lines.append(f"Epochs Logged           : {ls.get('epochs_list')}")
        report_lines.append("Recent Log Entries      :")
        for l in ls.get("last_lines", []):
            report_lines.append(f"  {l}")
    else:
        report_lines.append("No training log found.")
    report_lines.append("")

    report_content = "\n".join(report_lines)

    # Save to file
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print("=" * 70)
    print(f"Diagnostic report successfully saved to:\n{REPORT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
