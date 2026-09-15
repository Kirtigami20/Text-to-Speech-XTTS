import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_INPUT = ROOT / "training" / "train_metadata_clean.csv"
VAL_INPUT = ROOT / "training" / "validation_metadata_clean.csv"

TRAIN_OUTPUT = ROOT / "training" / "train_metadata_final.csv"
VAL_OUTPUT = ROOT / "training" / "validation_metadata_final.csv"


# ============================================================
# CHECK CORRUPTED UNICODE
# ============================================================

def has_corrupted_unicode(text):

    if not text:
        return True

    # Replacement character
    if "\ufffd" in text:
        return True

    for ch in text:

        code = ord(ch)

        # NULL
        if code == 0:
            return True

        # Control characters
        if code < 32 and ch not in "\t\n\r":
            return True

        # Private-use Unicode
        if 0xE000 <= code <= 0xF8FF:
            return True

    return False


# ============================================================
# AUDIO CHECK
# ============================================================

def audio_exists(audio_file):

    if not audio_file:
        return False

    return (ROOT / audio_file).exists()


# ============================================================
# PROCESS DATASET
# ============================================================

def process_metadata(input_file, output_file, label):

    print()
    print("=" * 70)
    print("PROCESSING", label)
    print("=" * 70)

    print("INPUT :", input_file)
    print("OUTPUT:", output_file)

    with open(
        input_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="|"
        )

        rows = list(reader)

    print()
    print("Original rows:", len(rows))

    kept = []
    removed = []

    stats = {
        "CORRUPTED_UNICODE": 0,
        "SHORT_TEXT": 0,
        "MISSING_AUDIO": 0,
    }

    for row in rows:

        audio_file = row.get(
            "audio_file",
            ""
        ).strip()

        text = row.get(
            "text",
            ""
        ).strip()

        reason = None

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if not audio_exists(audio_file):

            reason = "MISSING_AUDIO"

        # ----------------------------------------------------
        # EMPTY / VERY SHORT TEXT
        # ----------------------------------------------------

        elif len(text) < 5:

            reason = "SHORT_TEXT"

        # ----------------------------------------------------
        # CORRUPTED UNICODE
        # ----------------------------------------------------

        elif has_corrupted_unicode(text):

            reason = "CORRUPTED_UNICODE"

        # ----------------------------------------------------
        # KEEP
        # ----------------------------------------------------

        if reason is None:

            kept.append(row)

        else:

            stats[reason] += 1

            removed.append({
                "audio_file": audio_file,
                "text": text,
                "reason": reason
            })

    # ========================================================
    # WRITE FINAL CSV
    # ========================================================

    with open(
        output_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        fieldnames = list(
            rows[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="|"
        )

        writer.writeheader()
        writer.writerows(kept)

    # ========================================================
    # WRITE REMOVED REPORT
    # ========================================================

    removed_file = output_file.with_name(
        output_file.stem +
        "_removed.csv"
    )

    with open(
        removed_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_file",
                "text",
                "reason"
            ],
            delimiter="|"
        )

        writer.writeheader()
        writer.writerows(removed)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print("Original :", len(rows))
    print("Kept     :", len(kept))
    print("Removed  :", len(removed))

    print()
    print("Removal reasons:")

    for key, value in stats.items():

        print(
            f"{key:<25} {value}"
        )

    print()
    print("Final metadata:")
    print(output_file)

    print()
    print("Removed report:")
    print(removed_file)

    return len(rows), len(kept), len(removed)


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("XTTS CONSERVATIVE DATASET CLEANING")
print("=" * 70)

print()
print("Rules:")
print("  KEEP long samples")
print("  KEEP repeated domain terminology")
print("  KEEP Hindi")
print("  KEEP Hinglish")
print("  KEEP Marathi")
print("  KEEP Manglish")
print("  KEEP English")
print("  REMOVE corrupted Unicode")
print("  REMOVE missing audio")
print("  REMOVE empty/very short text")

train_result = process_metadata(
    TRAIN_INPUT,
    TRAIN_OUTPUT,
    "TRAIN"
)

val_result = process_metadata(
    VAL_INPUT,
    VAL_OUTPUT,
    "VALIDATION"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print()
print("TRAIN")
print("Original :", train_result[0])
print("Kept     :", train_result[1])
print("Removed  :", train_result[2])

print()
print("VALIDATION")
print("Original :", val_result[0])
print("Kept     :", val_result[1])
print("Removed  :", val_result[2])

print()
print("TRAIN METADATA:")
print(TRAIN_OUTPUT)

print()
print("VALIDATION METADATA:")
print(VAL_OUTPUT)

print()
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70) 