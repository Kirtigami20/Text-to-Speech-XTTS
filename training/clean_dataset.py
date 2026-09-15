import csv
import shutil
import subprocess
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_AUDIO = ROOT / "processed"
OUTPUT_AUDIO = ROOT / "processed_clean"

TRAIN_CSV = ROOT / "training" / "train_metadata.csv"
VAL_CSV = ROOT / "training" / "validation_metadata.csv"

OUTPUT_TRAIN_CSV = ROOT / "training" / "train_metadata_clean.csv"
OUTPUT_VAL_CSV = ROOT / "training" / "validation_metadata_clean.csv"

TARGET_SR = 24000

# Conservative cleaning
HIGH_PASS = 60
LOW_PASS = 11000

# ============================================================
# CHECK FFMPEG
# ============================================================

print("=" * 70)
print("XTTS DATASET CLEANING")
print("=" * 70)

print("\nChecking FFmpeg...")

if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "\nFFmpeg was not found in PATH.\n"
        "Install FFmpeg and restart PowerShell before continuing."
    )

print("FFmpeg: OK")

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_AUDIO.mkdir(parents=True, exist_ok=True)

print("\nInput audio :", INPUT_AUDIO)
print("Output audio:", OUTPUT_AUDIO)

# ============================================================
# CLEAN ONE AUDIO FILE
# ============================================================

def clean_audio(src, dst):

    dst.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = (
        f"highpass=f={HIGH_PASS},"
        f"lowpass=f={LOW_PASS},"
        "dynaudnorm=f=75:g=3"
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",

        "-i",
        str(src),

        # Mono
        "-ac",
        "1",

        # XTTS-friendly sample rate
        "-ar",
        str(TARGET_SR),

        # Conservative cleanup
        "-af",
        filter_chain,

        # WAV PCM
        "-c:a",
        "pcm_s16le",

        str(dst),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


# ============================================================
# PROCESS METADATA
# ============================================================

def process_metadata(input_csv, output_csv):

    print("\n" + "=" * 70)
    print("PROCESSING")
    print(input_csv.name)
    print("=" * 70)

    with open(
        input_csv,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as infile:

        reader = csv.DictReader(
            infile,
            delimiter="|"
        )

        rows = list(reader)

    print("Metadata rows:", len(rows))

    cleaned_rows = []

    success = 0
    failed = 0

    for index, row in enumerate(rows, 1):

        relative_audio = Path(row["audio_file"])

        # Original:
        # processed/hindi/file.wav
        #
        # New:
        # processed_clean/hindi/file.wav

        src = ROOT / relative_audio

        try:

            if not src.exists():
                print(
                    f"[{index}/{len(rows)}] "
                    f"MISSING: {src}"
                )

                failed += 1
                continue

            # Replace processed -> processed_clean
            parts = relative_audio.parts

            if parts[0].lower() == "processed":
                clean_relative = Path(
                    "processed_clean",
                    *parts[1:]
                )
            else:
                clean_relative = Path(
                    "processed_clean",
                    *parts
                )

            dst = ROOT / clean_relative

            # Skip if already cleaned
            if not dst.exists():
                clean_audio(src, dst)

            new_row = dict(row)

            # Metadata points to cleaned file
            new_row["audio_file"] = (
                clean_relative.as_posix()
            )

            cleaned_rows.append(new_row)

            success += 1

            if index % 25 == 0:
                print(
                    f"Processed {index}/{len(rows)} "
                    f"| Success: {success} "
                    f"| Failed: {failed}"
                )

        except Exception as e:

            print(
                f"[{index}/{len(rows)}] "
                f"FAILED: {src}"
            )

            print("   ERROR:", e)

            failed += 1

    # ========================================================
    # WRITE CLEAN METADATA
    # ========================================================

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "audio_file",
                "text",
                "speaker_name"
            ],
            delimiter="|"
        )

        writer.writeheader()
        writer.writerows(cleaned_rows)

    print("\nCompleted:", input_csv.name)

    print("Successful:", success)
    print("Failed    :", failed)

    print("Output CSV:", output_csv)

    return success, failed


# ============================================================
# PROCESS TRAINING DATA
# ============================================================

train_success, train_failed = process_metadata(
    TRAIN_CSV,
    OUTPUT_TRAIN_CSV
)

# ============================================================
# PROCESS VALIDATION DATA
# ============================================================

val_success, val_failed = process_metadata(
    VAL_CSV,
    OUTPUT_VAL_CSV
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(
    f"\nTRAIN:"
    f"\n  Success: {train_success}"
    f"\n  Failed : {train_failed}"
)

print(
    f"\nVALIDATION:"
    f"\n  Success: {val_success}"
    f"\n  Failed : {val_failed}"
)

print("\nClean audio:")
print(OUTPUT_AUDIO)

print("\nClean training metadata:")
print(OUTPUT_TRAIN_CSV)

print("\nClean validation metadata:")
print(OUTPUT_VAL_CSV)

print("\nOriginal dataset was NOT modified.")

print("\nDone.")