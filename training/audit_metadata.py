import csv
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

INPUT_FILES = [
    Path("training/train_metadata_clean.csv"),
    Path("training/validation_metadata_clean.csv"),
]

REPORT_DIR = Path("training")
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# AUDIT FUNCTION
# ============================================================

def audit_file(input_file):

    output_file = REPORT_DIR / (
        input_file.stem.replace("_clean", "") + "_audit.csv"
    )

    print("\n" + "=" * 70)
    print("AUDITING")
    print("=" * 70)

    print("Input :", input_file)
    print("Report:", output_file)

    if not input_file.exists():
        print("ERROR: File not found!")
        return

    suspicious = []
    problem_counts = Counter()

    with input_file.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="|"
        )

        for row_number, row in enumerate(reader, 1):

            text = row.get("text", "").strip()

            problems = []

            # ------------------------------------------------
            # Unicode replacement character
            # ------------------------------------------------

            if "\ufffd" in text:
                problems.append("CORRUPTED_UNICODE")

            # ------------------------------------------------
            # Very short text
            # ------------------------------------------------

            if len(text) < 10:
                problems.append("VERY_SHORT")

            # ------------------------------------------------
            # Long text
            #
            # NOT automatically considered bad.
            # We only flag it for review.
            # ------------------------------------------------

            if len(text) > 250:
                problems.append("LONG_TEXT")

            # ------------------------------------------------
            # Excessive spaces
            # ------------------------------------------------

            if "  " in text:
                problems.append("EXTRA_SPACES")

            # ------------------------------------------------
            # Excessive commas
            # ------------------------------------------------

            if text.count(",") > 15:
                problems.append("EXCESSIVE_COMMAS")

            # ------------------------------------------------
            # Excessive periods
            # ------------------------------------------------

            if text.count(".") > 15:
                problems.append("EXCESSIVE_PERIODS")

            # ------------------------------------------------
            # Repeated 3-word phrase
            #
            # Only flag for review.
            # Do NOT automatically delete.
            # ------------------------------------------------

            words = text.split()

            repeated = False

            if len(words) >= 10:

                phrases = []

                for i in range(len(words) - 2):

                    phrase = tuple(words[i:i + 3])

                    if phrase in phrases:
                        repeated = True
                        break

                    phrases.append(phrase)

            if repeated:
                problems.append("REPEATED_PHRASE")

            # ------------------------------------------------
            # Save suspicious sample
            # ------------------------------------------------

            if problems:

                unique_problems = sorted(set(problems))

                for problem in unique_problems:
                    problem_counts[problem] += 1

                suspicious.append({
                    "row": row_number,
                    "audio_file": row.get("audio_file", ""),
                    "text_length": len(text),
                    "problems": ",".join(unique_problems),
                    "text": text,
                })


    # ========================================================
    # WRITE REPORT
    # ========================================================

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row",
                "audio_file",
                "text_length",
                "problems",
                "text",
            ],
            delimiter="|",
        )

        writer.writeheader()
        writer.writerows(suspicious)


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print("Total rows checked :", row_number)
    print("Suspicious samples :", len(suspicious))

    print("\nProblem counts:")

    if problem_counts:

        for problem, count in problem_counts.most_common():
            print(f"{problem:<25} {count}")

    else:
        print("No suspicious samples found.")

    print("\nReport saved:")
    print(output_file)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("XTTS DATASET METADATA AUDIT")
print("=" * 70)

for input_file in INPUT_FILES:
    audit_file(input_file)

print("\n" + "=" * 70)
print("ALL AUDITS COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("This script ONLY audits metadata.")
print("It does NOT delete or modify any training samples.")
print("It does NOT modify your audio files.")

print("\nDone.")