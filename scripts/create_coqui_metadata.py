from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"

TRAIN_CSV = TRAINING / "train.csv"
VAL_CSV = TRAINING / "validation.csv"

TRAIN_META = TRAINING / "train_metadata.csv"
VAL_META = TRAINING / "validation_metadata.csv"


def create_metadata(input_csv, output_file):
    # Explicitly read the source as UTF-8
    df = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    required = {"audio_path", "text", "language"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    metadata = pd.DataFrame({
        "audio_file": df["audio_path"].str.strip(),
        "text": df["text"].str.strip(),
        "speaker_name": "rachana",
    })

    # Write TRUE UTF-8, no BOM
    metadata.to_csv(
        output_file,
        sep="|",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return len(metadata)


print("=" * 70)
print("CREATING COQUI METADATA")
print("=" * 70)

train_count = create_metadata(TRAIN_CSV, TRAIN_META)
val_count = create_metadata(VAL_CSV, VAL_META)

print(f"train_metadata.csv: {train_count} samples")
print(f"validation_metadata.csv: {val_count} samples")

print("=" * 70)
print("DONE")
print("=" * 70)