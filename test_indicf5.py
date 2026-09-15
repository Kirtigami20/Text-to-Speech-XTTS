import sys
import io
import os

# Force UTF-8 output so Hindi/Marathi chars don't crash Windows cmd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
from f5_tts.model import DiT


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# IndicF5 cached model files (downloaded by previous AutoModel attempt)
SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ai4bharat--IndicF5"
    / "snapshots"
    / "ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
)

CKPT_PATH  = str(SNAPSHOT / "model.safetensors")
VOCAB_FILE = str(SNAPSHOT / "checkpoints" / "vocab.txt")

# Same reference audio used in the XTTS test
REFERENCE_AUDIO = ROOT / "processed" / "hindi" / "hi_000621.wav"
REFERENCE_TEXT  = "यही आपको validate करना है तो आप skip these questions को select करें"

OUTPUT_DIR = ROOT / "indicf5_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STARTUP INFO
# ============================================================

print("=" * 70)
print("INDICF5 VOICE TEST - RACHANA (Zero-Shot Voice Cloning)")
print("=" * 70)
print()
print("CKPT       :", CKPT_PATH)
print("VOCAB      :", VOCAB_FILE)
print("REFERENCE  :", REFERENCE_AUDIO)
print("REF TEXT   :", REFERENCE_TEXT)
print("OUTPUT DIR :", OUTPUT_DIR)
print()

for p in [CKPT_PATH, VOCAB_FILE, str(REFERENCE_AUDIO)]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing: {p}")

print("All files found.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE     :", device)
if device == "cuda":
    print("GPU        :", torch.cuda.get_device_name(0))


# ============================================================
# LOAD VOCODER
# ============================================================

print()
print("=" * 70)
print("LOADING VOCOS VOCODER")
print("=" * 70)

vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
print("Vocoder loaded.")


# ============================================================
# LOAD INDICF5 MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING INDICF5 MODEL")
print("=" * 70)

from f5_tts.infer.utils_infer import get_tokenizer, CFM
from safetensors.torch import load_file

vocab_char_map, vocab_size = get_tokenizer(VOCAB_FILE, "custom")

model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

model = CFM(
    transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
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

raw_ckpt = load_file(CKPT_PATH, device=device)
clean_ckpt = {
    k.replace("ema_model.", "").replace("_orig_mod.", ""): v
    for k, v in raw_ckpt.items()
    if not k.startswith("vocoder.") and k not in ["initted", "step"]
}

model.load_state_dict(clean_ckpt, strict=False)

print("IndicF5 model loaded successfully.")



# ============================================================
# PREPROCESS REFERENCE AUDIO
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING REFERENCE AUDIO")
print("=" * 70)

ref_audio, ref_text = preprocess_ref_audio_text(
    str(REFERENCE_AUDIO),
    REFERENCE_TEXT,
)

print("Reference audio preprocessed.")
print("Ref text used:", ref_text)


# ============================================================
# TEST SENTENCES (same as XTTS for direct comparison)
# ============================================================

tests = [

    {
        "name": "01_hindi",
        "text": "नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में बात करेंगे।",
    },

    {
        "name": "02_english",
        "text": "Hello, today we are going to talk about income tax and finance.",
    },

    {
        "name": "03_hinglish",
        "text": "Aaj hum income tax ke baare mein baat karenge aur dekhenge ki tax planning kaise karni hai.",
    },

    {
        "name": "04_hindi_english",
        "text": "Aapko apna ITR file karna hai, aur uske baad tax calculation properly check karni hai.",
    },

    {
        "name": "05_marathi",
        "text": "नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल माहिती घेणार आहोत.",
    },

    {
        "name": "06_marnglish",
        "text": "Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot.",
    },

    {
        "name": "07_marnglish_natural",
        "text": "Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara.",
    },

    {
        "name": "08_indian_english",
        "text": "If you are filing your income tax return in India, you should carefully check all your income details.",
    },

]


# ============================================================
# GENERATION
# ============================================================

print()
print("=" * 70)
print("STARTING INDICF5 VOICE TESTS")
print("=" * 70)

successful = 0

for i, test in enumerate(tests, 1):

    name = test["name"]
    text = test["text"]
    output_file = OUTPUT_DIR / f"{name}.wav"

    print()
    print("-" * 70)
    print(f"TEST {i}/{len(tests)}")
    print("NAME :", name)
    print("TEXT :", text)
    print("-" * 70)

    try:

        audio_wave, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            device=device,
        )

        if audio_wave is None:
            print("FAILED - no audio returned")
            continue

        sf.write(str(output_file), audio_wave, sample_rate)

        duration = len(audio_wave) / sample_rate

        print("SUCCESS")
        print(f"SAVED    : {output_file}")
        print(f"DURATION : {duration:.2f} sec")

        successful += 1

    except Exception as e:
        print("FAILED")
        print("ERROR:", repr(e))
        import traceback
        traceback.print_exc()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("INDICF5 TEST COMPLETE")
print("=" * 70)
print(f"Successful: {successful}/{len(tests)}")
print()
print("Generated files:")

for wav in sorted(OUTPUT_DIR.glob("*.wav")):
    size_mb = wav.stat().st_size / (1024 * 1024)
    print(f"  {wav.name:<35} {size_mb:.2f} MB")

print()
print("OUTPUT FOLDER:")
print(OUTPUT_DIR)
print()
print("Done.")
