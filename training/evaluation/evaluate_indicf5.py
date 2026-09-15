"""
TRACK B — IndicF5 Zero-Shot Baseline Evaluation
================================================
Direct inference using f5_tts and ai4bharat/IndicF5 weights.
Zero-shot voice cloning with a clean Marathi reference clip.
Same 7 evaluation sentences as B0/E1 for direct comparison.
"""

import sys
import os
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import safetensors.torch
from vocos import Vocos

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import infer_process

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")
EVAL_DIR = PROJECT_ROOT / "training" / "xtts_evaluations" / "TrackB"
EVAL_SENTENCES = PROJECT_ROOT / "training" / "evaluation" / "evaluation_sentences.txt"
REPORT_FILE = EVAL_DIR / "report.txt"

# Reference: clean ~12.6s Marathi clip from the dataset
REF_AUDIO = PROJECT_ROOT / "processed_clean" / "marathi" / "mr_001682.wav"
REF_TEXT = "अजुने सांगते जन्नी क्रिप्टो मदे वगर घेत लेसेल, तरहां VDS मन्टाद, Virtual Digital Asset क्रिप्टो मदे जर्ग उन्तर नुकी लेसेल, तरही सुद्धा ITR टूट्स भराल"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRACK B — IndicF5 ZERO-SHOT EVALUATION")
print("=" * 70)

assert EVAL_SENTENCES.exists(), f"Missing: {EVAL_SENTENCES}"
assert REF_AUDIO.exists(), f"Missing: {REF_AUDIO}"

# Load evaluation sentences
sentences = []
with open(EVAL_SENTENCES, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            sentences.append({
                "id": f"eval_{idx+1:02d}",
                "category": parts[0].strip(),
                "language": parts[1].strip(),
                "text": "|".join(parts[2:]).strip()
            })

print(f"\nLoaded {len(sentences)} evaluation sentences.")
print(f"Reference audio : {REF_AUDIO}")
print(f"Reference text  : {REF_TEXT[:80]}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device    : {device}")

print("\nDownloading IndicF5 checkpoints...")
vocab_file = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="checkpoints/vocab.txt")
ckpt_file = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="model.safetensors")

print("Initializing tokenizer & architecture...")
vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
print(f"Vocab size: {vocab_size}")

model = CFM(
    transformer=DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        text_num_embeds=vocab_size,
        mel_dim=100,
    ),
    mel_spec_kwargs=dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    ),
    odeint_kwargs=dict(method="euler"),
    vocab_char_map=vocab_char_map,
).to(device)

print("Loading IndicF5 weights from safetensors...")
state_dict = safetensors.torch.load_file(ckpt_file, device=device)
new_sd = {}
for k, v in state_dict.items():
    if k.startswith("ema_model._orig_mod."):
        new_k = k.replace("ema_model._orig_mod.", "")
        new_sd[new_k] = v
    elif k.startswith("ema_model."):
        new_k = k.replace("ema_model.", "")
        new_sd[new_k] = v

model.load_state_dict(new_sd, strict=False)
model.eval()

print("Loading Vocos vocoder...")
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
vocoder.eval()
print("All models ready for synthesis!")

# Synthesize
results = []
for test in sentences:
    out_wav = EVAL_DIR / f"{test['id']}.wav"
    print(f"\nSynthesizing [{test['category']}] -> {out_wav.name}")
    print(f"  Text: {test['text'][:100]}...")
    sys.stdout.flush()

    try:
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio=str(REF_AUDIO),
            ref_text=REF_TEXT,
            gen_text=test["text"],
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            nfe_step=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            speed=1.0,
            device=device,
        )

        sf.write(str(out_wav), final_wave, final_sample_rate)
        duration = len(final_wave) / final_sample_rate
        size_mb = out_wav.stat().st_size / (1024 * 1024)

        print(f"  Duration: {duration:.2f}s | Size: {size_mb:.2f} MB")
        results.append({
            "test": test,
            "filename": out_wav.name,
            "duration": duration,
            "size_mb": size_mb,
            "status": "SUCCESS"
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({
            "test": test,
            "filename": out_wav.name,
            "duration": 0,
            "size_mb": 0,
            "status": f"FAILED: {e}"
        })

# Write report
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("                   TRACK B (IndicF5) EVALUATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write("1. EXPERIMENT SUMMARY\n")
    f.write("   Model              : ai4bharat/IndicF5 (zero-shot, no fine-tuning)\n")
    f.write(f"   Reference Audio    : {REF_AUDIO}\n")
    f.write(f"   Reference Text     : {REF_TEXT}\n\n")
    f.write("2. GENERATED AUDIO FILES\n")
    f.write(f"{'Filename':<15} {'Category':<45} {'Duration':<10} {'Status'}\n")
    f.write("-" * 80 + "\n")
    for r in results:
        f.write(f"{r['filename']:<15} {r['test']['category']:<45} {r['duration']:.2f}s     {r['status']}\n")
    f.write("\n" + "=" * 70 + "\n")

print(f"\nReport saved to: {REPORT_FILE}")
