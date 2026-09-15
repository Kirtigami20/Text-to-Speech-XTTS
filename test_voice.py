import sys
import io

# Force UTF-8 output so Hindi/Marathi chars don't crash Windows cmd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

RUN_DIR = (
    ROOT
    / "training"
    / "xtts_output"
    / "rachana_xtts_clean_2epoch-August-27-2026_12+14PM-0000000"
)

MODEL = RUN_DIR / "best_model_1522.pth"
CONFIG = RUN_DIR / "config.json"

REFERENCE = (
    ROOT
    / "processed"
    / "hindi"
    / "hi_000621.wav"
)

OUTPUT = ROOT / "test_outputs"
OUTPUT.mkdir(exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("XTTS VOICE TEST - RACHANA")
print("=" * 70)

print("MODEL     :", MODEL)
print("CONFIG    :", CONFIG)
print("REFERENCE :", REFERENCE)

for path in [MODEL, CONFIG, REFERENCE]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

print("\nAll required files found.")


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nDEVICE:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CLEAN OLD TEST OUTPUTS
# ============================================================

print("\nCleaning old test files...")

for wav in OUTPUT.glob("*.wav"):
    try:
        wav.unlink()
    except Exception:
        pass

print("Output folder ready.")


# ============================================================
# LOAD CONFIG
# ============================================================

print("\n" + "=" * 70)
print("LOADING CONFIG")
print("=" * 70)

config = XttsConfig()
config.load_json(str(CONFIG))

print("Config loaded.")

print("\nXTTS supported languages:")
print(config.languages)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = Xtts.init_from_config(config)

print("Model initialized.")

model.load_checkpoint(
    config,
    checkpoint_path=str(MODEL),
    eval=True,
)

model.to(device)

print("Model loaded successfully.")


# ============================================================
# TEST SENTENCES
# ============================================================

tests = [

    {
        "name": "01_hindi",
        "language": "hi",
        "text": (
            "नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में "
            "बात करेंगे।"
        ),
    },

    {
        "name": "02_english",
        "language": "en",
        "text": (
            "Hello, today we are going to talk about "
            "income tax and finance."
        ),
    },

    {
        "name": "03_hinglish",
        "language": "hi",
        "text": (
            "Aaj hum income tax ke baare mein baat karenge "
            "aur dekhenge ki tax planning kaise karni hai."
        ),
    },

    {
        "name": "04_hindi_english",
        "language": "hi",
        "text": (
            "Aapko apna ITR file karna hai, aur uske baad "
            "tax calculation properly check karni hai."
        ),
    },

    {
        "name": "05_marathi",
        "language": "hi",
        "text": (
            "नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल "
            "माहिती घेणार आहोत."
        ),
    },

    {
        "name": "06_marnglish",
        "language": "hi",
        "text": (
            "Aaj apan income tax cha calculation kasa karaycha "
            "te samjun ghenar aahot."
        ),
    },

    {
        "name": "07_marnglish_natural",
        "language": "hi",
        "text": (
            "Tumhala ITR file karaycha asel tar first tumhi "
            "your income details properly check kara."
        ),
    },

    {
        "name": "08_indian_english",
        "language": "en",
        "text": (
            "If you are filing your income tax return in India, "
            "you should carefully check all your income details."
        ),
    },
]


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING VOICE TESTS")
print("=" * 70)

successful = 0

for i, test in enumerate(tests, 1):

    name = test["name"]
    language = test["language"]
    text = test["text"]

    output_file = OUTPUT / f"{name}.wav"

    print("\n" + "-" * 70)
    print(f"TEST {i}/{len(tests)}")
    print("NAME     :", name)
    print("LANGUAGE :", language)
    print("TEXT     :", text)
    print("-" * 70)

    try:

        result = model.synthesize(
            text=text,
            config=config,
            speaker_wav=str(REFERENCE),
            language=language,

            # Speaker conditioning
            gpt_cond_len=12,

            # Sampling
            temperature=0.65,

            # Keep generation stable
            enable_text_splitting=True,
        )

        wav = result["wav"]

        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()

        wav = np.asarray(wav)
        wav = np.squeeze(wav)
        wav = wav.astype(np.float32)

        # XTTS output sample rate
        sample_rate = 24000

        sf.write(
            str(output_file),
            wav,
            sample_rate,
        )

        duration = len(wav) / sample_rate

        print("SUCCESS")
        print("SAVED    :", output_file)
        print(f"DURATION : {duration:.2f} sec")

        successful += 1

    except Exception as e:

        print("FAILED")
        print("ERROR:", repr(e))


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VOICE TEST COMPLETE")
print("=" * 70)

print(
    f"Successful: {successful}/{len(tests)}"
)

print("\nGenerated files:")

for wav in sorted(OUTPUT.glob("*.wav")):

    size_mb = wav.stat().st_size / (1024 * 1024)

    print(
        f"{wav.name:<35} "
        f"{size_mb:.2f} MB"
    )

print("\nOUTPUT FOLDER:")
print(OUTPUT)

print("\nDone.")