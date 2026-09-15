import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

MODEL = ROOT / "training" / "xtts_output" / \
    "rachana_xtts_clean_2epoch-August-27-2026_12+14PM-0000000" / \
    "best_model_1522.pth"

CONFIG = MODEL.parent / "config.json"

SPEAKER = ROOT / "processed" / "hindi" / "hi_000621.wav"

OUT_DIR = ROOT / "training" / "voice_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Import XTTS
# ---------------------------------------------------------

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# ---------------------------------------------------------
# Import normalizer
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(ROOT / "training" / "text_normalizer")
)

from roman_normalizer import normalize_text


# ---------------------------------------------------------
# Text
# ---------------------------------------------------------

roman_text = (
    "Aaj apan income tax cha calculation kasa karaycha "
    "te samjun ghenar aahot."
)

normalized_text = normalize_text(
    roman_text,
    language_hint="mr"
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("=" * 70)
print("XTTS ROMAN vs NORMALIZED COMPARISON")
print("=" * 70)

print("\nROMAN INPUT:")
print(roman_text)

print("\nNORMALIZED INPUT:")
print(normalized_text)


# ---------------------------------------------------------
# Load XTTS
# ---------------------------------------------------------

print("\nLoading XTTS...")

config = XttsConfig()
config.load_json(str(CONFIG))

model = Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_path=str(MODEL),
    eval=True
)

print("Model loaded.")


# ---------------------------------------------------------
# Generate helper
# ---------------------------------------------------------

def generate(text, filename):

    print(f"\nGenerating: {filename}")

    result = model.synthesize(
        text,
        config,
        speaker_wav=str(SPEAKER),
        language="hi",
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
    )

    wav = np.asarray(
        result["wav"],
        dtype=np.float32
    )

    output = OUT_DIR / filename

    sf.write(
        str(output),
        wav,
        24000
    )

    print("Saved:")
    print(output)


# ---------------------------------------------------------
# Generate BOTH versions
# ---------------------------------------------------------

generate(
    roman_text,
    "01_direct_roman_marathi.wav"
)

generate(
    normalized_text,
    "02_normalized_marathi.wav"
)


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nCompare these two files:")

print(
    OUT_DIR / "01_direct_roman_marathi.wav"
)

print(
    OUT_DIR / "02_normalized_marathi.wav"
)