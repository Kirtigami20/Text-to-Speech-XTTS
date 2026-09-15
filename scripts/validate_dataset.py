from pathlib import Path
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "reports" / "transcription_metadata_v2.csv"
REPORTS = ROOT / "reports"

df = pd.read_csv(METADATA)

# Normalize transcript column
df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()

results = []

for idx, row in df.iterrows():
    text = row["transcript"]
    language = row["language"]
    audio_path = ROOT / row["audio_path"]

    flags = []
    score = 0

    # ---------------------------------------------------------
    # 1. Empty transcript
    # ---------------------------------------------------------
    if not text:
        flags.append("EMPTY")
        score += 10

    # ---------------------------------------------------------
    # 2. Very short transcript
    # ---------------------------------------------------------
    if 0 < len(text) < 20:
        flags.append("VERY_SHORT")
        score += 2

    # ---------------------------------------------------------
    # 3. Very long transcript
    # ---------------------------------------------------------
    if len(text) > 400:
        flags.append("VERY_LONG")
        score += 2

    # ---------------------------------------------------------
    # 4. Repeated words / hallucination-like repetition
    # ---------------------------------------------------------
    words = text.lower().split()

    if len(words) >= 8:
        repeated_pairs = sum(
            words[i] == words[i + 1]
            for i in range(len(words) - 1)
        )

        if repeated_pairs >= 3:
            flags.append("REPETITION")
            score += 4

    # ---------------------------------------------------------
    # 5. Excessive punctuation
    # ---------------------------------------------------------
    if len(text) > 20:
        punctuation = len(re.findall(r"[!?.,;:]", text))
        ratio = punctuation / len(text)

        if ratio > 0.15:
            flags.append("HIGH_PUNCTUATION")
            score += 2

    # ---------------------------------------------------------
    # 6. Audio file existence
    # ---------------------------------------------------------
    if not audio_path.exists():
        flags.append("AUDIO_MISSING")
        score += 10

    # ---------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------
    if score >= 8:
        quality = "REJECT"
    elif score >= 2:
        quality = "REVIEW"
    else:
        quality = "KEEP"

    results.append({
        "index": idx,
        "audio_path": row["audio_path"],
        "language": language,
        "transcript": text,
        "text_length": len(text),
        "flags": "|".join(flags),
        "quality": quality,
    })


qc = pd.DataFrame(results)

# -------------------------------------------------------------
# Duplicate transcript information
# -------------------------------------------------------------
duplicate_mask = qc["transcript"].duplicated(keep=False)

qc.loc[duplicate_mask & (qc["quality"] == "KEEP"), "quality"] = "REVIEW"

qc.loc[
    duplicate_mask,
    "flags"
] = qc.loc[
    duplicate_mask,
    "flags"
].apply(
    lambda x: f"{x}|DUPLICATE_TRANSCRIPT" if x else "DUPLICATE_TRANSCRIPT"
)

# -------------------------------------------------------------
# Save complete QC report
# -------------------------------------------------------------
qc_path = REPORTS / "dataset_qc.csv"
qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

# -------------------------------------------------------------
# Save review-only report
# -------------------------------------------------------------
review = qc[qc["quality"] != "KEEP"]

review_path = REPORTS / "qc_review.csv"
review.to_csv(review_path, index=False, encoding="utf-8-sig")

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
summary = {
    "total_samples": len(qc),
    "keep": int((qc["quality"] == "KEEP").sum()),
    "review": int((qc["quality"] == "REVIEW").sum()),
    "reject": int((qc["quality"] == "REJECT").sum()),
    "languages": {
        lang: int((qc["language"] == lang).sum())
        for lang in sorted(qc["language"].dropna().unique())
    },
    "duplicate_transcripts": int(duplicate_mask.sum()),
}

summary_path = REPORTS / "qc_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# -------------------------------------------------------------
# Console output
# -------------------------------------------------------------
print("=" * 70)
print("MULTILINGUAL FEMALE TTS — DATASET QUALITY CONTROL")
print("=" * 70)

print(f"\nTotal samples : {summary['total_samples']}")
print(f"KEEP          : {summary['keep']}")
print(f"REVIEW        : {summary['review']}")
print(f"REJECT        : {summary['reject']}")
print(f"Duplicate     : {summary['duplicate_transcripts']}")

print("\nLanguage distribution:")
for lang, count in summary["languages"].items():
    print(f"  {lang.upper():<5} {count}")

print("\nReports:")
print(f"QC report     : {qc_path}")
print(f"Review report : {review_path}")
print(f"Summary       : {summary_path}")

print("\nQC COMPLETE")