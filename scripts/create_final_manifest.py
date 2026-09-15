from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "training" / "metadata.csv"
OUTPUT_JSONL = ROOT / "training" / "manifest.jsonl"
OUTPUT_CSV = ROOT / "training" / "final_manifest.csv"
SUMMARY_JSON = ROOT / "training" / "manifest_summary.json"

print("=" * 70)
print("MULTILINGUAL FEMALE TTS — FINAL MANIFEST CREATION")
print("=" * 70)

# ---------------------------------------------------------
# Load prepared training metadata
# ---------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

print(f"\nInput samples : {len(df)}")
print(f"Columns       : {df.columns.tolist()}")

# ---------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------
df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["language"] = (
    df["language"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# ---------------------------------------------------------
# Validate samples
# ---------------------------------------------------------
valid_rows = []
missing_audio = 0
empty_text = 0
invalid_language = 0

for _, row in df.iterrows():

    audio_path = ROOT / str(row["audio_path"])
    text = row["text"]
    language = row["language"]

    if not text:
        empty_text += 1
        continue

    if not audio_path.exists():
        missing_audio += 1
        continue

    if language not in {"en", "hi", "mr"}:
        invalid_language += 1
        continue

    valid_rows.append({
        "audio_path": str(
            audio_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "text": text,
        "language": language,
    })

manifest = pd.DataFrame(valid_rows)

# ---------------------------------------------------------
# Save CSV manifest
# ---------------------------------------------------------
manifest.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Save JSONL manifest
# ---------------------------------------------------------
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:

    for row in valid_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )

# ---------------------------------------------------------
# Language distribution
# ---------------------------------------------------------
language_counts = manifest["language"].value_counts().to_dict()

summary = {
    "input_samples": len(df),
    "final_samples": len(manifest),
    "removed": len(df) - len(manifest),
    "empty_text": empty_text,
    "missing_audio": missing_audio,
    "invalid_language": invalid_language,
    "languages": {
        "en": int(language_counts.get("en", 0)),
        "hi": int(language_counts.get("hi", 0)),
        "mr": int(language_counts.get("mr", 0)),
    }
}

# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------
with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL MANIFEST READY")
print("=" * 70)

print(f"\nInput samples : {summary['input_samples']}")
print(f"Final samples : {summary['final_samples']}")
print(f"Removed       : {summary['removed']}")

print("\nValidation:")
print(f"  Empty text       : {summary['empty_text']}")
print(f"  Missing audio    : {summary['missing_audio']}")
print(f"  Invalid language : {summary['invalid_language']}")

print("\nLanguage distribution:")

for lang in ["en", "hi", "mr"]:
    print(
        f"  {lang.upper():<5} "
        f"{summary['languages'][lang]}"
    )

print("\nOutputs:")
print(f"  CSV     : {OUTPUT_CSV}")
print(f"  JSONL   : {OUTPUT_JSONL}")
print(f"  Summary : {SUMMARY_JSON}")

print("\nFINAL MANIFEST CREATION COMPLETE")