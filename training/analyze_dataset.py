import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ROOT / "training" / "train_metadata.csv",
    ROOT / "training" / "validation_metadata.csv",
]

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")

def classify(text):
    d = bool(DEVANAGARI.search(text))
    l = bool(LATIN.search(text))

    if d and l:
        return "MIXED_DEVANAGARI_LATIN"
    elif d:
        return "DEVANAGARI"
    elif l:
        return "LATIN"
    return "OTHER"


for file in FILES:

    print("\n" + "=" * 80)
    print(file.name)
    print("=" * 80)

    rows = []

    with open(file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")

        for row in reader:
            text = row["text"].strip()
            audio = row["audio_file"].strip()

            rows.append({
                "audio": audio,
                "text": text,
                "type": classify(text),
                "length": len(text),
            })

    print("TOTAL:", len(rows))

    print("\nSCRIPT DISTRIBUTION")
    counts = Counter(x["type"] for x in rows)

    for k, v in counts.items():
        print(f"{k:30} {v}")

    print("\nLONG TEXT (>150 CHARACTERS)")

    long_rows = [
        x for x in rows
        if x["length"] > 150
    ]

    print("COUNT:", len(long_rows))

    for x in long_rows[:20]:
        print("\nAUDIO:", x["audio"])
        print("LENGTH:", x["length"])
        print("TEXT:", x["text"])

    print("\nSUSPICIOUS TEXT")

    suspicious = []

    for x in rows:

        text = x["text"]

        # replacement character
        if "�" in text:
            suspicious.append((x, "replacement_character"))

        # repeated spaces
        elif re.search(r"\s{3,}", text):
            suspicious.append((x, "many_spaces"))

        # strange repeated characters
        elif re.search(r"(.)\1{5,}", text):
            suspicious.append((x, "repeated_character"))

        # extremely long text
        elif len(text) > 250:
            suspicious.append((x, "very_long"))

    print("COUNT:", len(suspicious))

    for x, reason in suspicious[:30]:
        print("\nREASON:", reason)
        print("AUDIO:", x["audio"])
        print("TEXT:", x["text"])


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)