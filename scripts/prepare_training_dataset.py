from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

METADATA = REPORTS / "transcription_metadata_v2.csv"
QC_REPORT = REPORTS / "dataset_qc.csv"

OUTPUT_DIR = ROOT / "training"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"
SUMMARY_JSON = OUTPUT_DIR / "training_summary.json"

print("=" * 70)
print("MULTILINGUAL FEMALE TTS — TRAINING DATA PREPARATION")
print("=" * 70)

# ---------------------------------------------------------
# Check required files
# ---------------------------------------------------------

if not METADATA.exists():
    raise FileNotFoundError(
        f"Missing metadata file:\n{METADATA}"
    )

if not QC_REPORT.exists():
    raise FileNotFoundError(
        f"Missing QC report:\n{QC_REPORT}"
    )

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

df = pd.read_csv(METADATA)
qc = pd.read_csv(QC_REPORT)

print(f"\nMetadata samples : {len(df)}")
print(f"QC samples       : {len(qc)}")

# ---------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------

df["audio_path"] = df["audio_path"].fillna("").astype(str).str.strip()
df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()
df["language"] = df["language"].fillna("").astype(str).str.strip()

qc["audio_path"] = qc["audio_path"].fillna("").astype(str).str.strip()
qc["quality"] = qc["quality"].fillna("").astype(str).str.strip()

# ---------------------------------------------------------
# Merge QC information
# ---------------------------------------------------------

df = df.merge(
    qc[["audio_path", "quality", "flags"]],
    on="audio_path",
    how="left"
)

# ---------------------------------------------------------
# Keep ONLY QC-approved samples
# ---------------------------------------------------------

before = len(df)

df = df[df["quality"] == "KEEP"].copy()

print("\nQC filtering:")
print(f"  Before : {before}")
print(f"  KEEP   : {len(df)}")
print(f"  Removed: {before - len(df)}")

# ---------------------------------------------------------
# Remove empty transcripts
# ---------------------------------------------------------

before_empty = len(df)

df = df[df["transcript"].str.len() > 0].copy()

print("\nEmpty transcript filtering:")
print(f"  Before : {before_empty}")
print(f"  After  : {len(df)}")

# ---------------------------------------------------------
# Verify audio files
# ---------------------------------------------------------

def audio_exists(path):
    return (ROOT / path).exists()


df["audio_exists"] = df["audio_path"].apply(audio_exists)

missing_audio = int((~df["audio_exists"]).sum())

print("\nAudio verification:")
print(f"  Missing audio : {missing_audio}")

if missing_audio > 0:
    print("\nMissing files:")
    for path in df.loc[~df["audio_exists"], "audio_path"]:
        print(f"  {path}")

df = df[df["audio_exists"]].copy()

# ---------------------------------------------------------
# Remove duplicate audio paths
# ---------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["audio_path"],
    keep="first"
)

print("\nDuplicate audio filtering:")
print(f"  Before : {before_duplicates}")
print(f"  After  : {len(df)}")

# ---------------------------------------------------------
# Create final training metadata
# ---------------------------------------------------------

training_df = pd.DataFrame({
    "audio_path": df["audio_path"],
    "text": df["transcript"],
    "language": df["language"]
})

# Normalize Windows paths
training_df["audio_path"] = (
    training_df["audio_path"]
    .str.replace("\\", "/", regex=False)
)

# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------
# Save training metadata
# ---------------------------------------------------------

training_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Create summary
# ---------------------------------------------------------

languages = {}

for language in sorted(training_df["language"].unique()):
    languages[language] = int(
        (training_df["language"] == language).sum()
    )

summary = {
    "source_samples": int(before),
    "training_samples": int(len(training_df)),
    "missing_audio": missing_audio,
    "languages": languages
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING DATA READY")
print("=" * 70)

print(f"\nTotal training samples : {len(training_df)}")

print("\nLanguage distribution:")

for language, count in languages.items():
    print(f"  {language.upper():<5} {count}")

print("\nOutputs:")
print(f"  Metadata : {OUTPUT_CSV}")
print(f"  Summary  : {SUMMARY_JSON}")

print("\nTRAINING DATA PREPARATION COMPLETE")
