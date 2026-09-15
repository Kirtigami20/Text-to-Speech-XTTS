import pandas as pd
from pathlib import Path


MASTER_FILE = Path("reports/transcription_metadata.csv")
RETRANSCRIPTION_FILE = Path("reports/retranscription_review.csv")
OUTPUT_FILE = Path("reports/transcription_metadata_v2.csv")


def clean_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def main():
    print("=" * 70)
    print("APPLY SECOND-PASS TRANSCRIPTIONS")
    print("=" * 70)

    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing: {MASTER_FILE}")

    if not RETRANSCRIPTION_FILE.exists():
        raise FileNotFoundError(f"Missing: {RETRANSCRIPTION_FILE}")

    master = pd.read_csv(MASTER_FILE)
    review = pd.read_csv(RETRANSCRIPTION_FILE)

    print(f"Master samples       : {len(master)}")
    print(f"Second-pass samples  : {len(review)}")

    # Preserve the original transcript
    if "original_transcript" not in master.columns:
        master["original_transcript"] = master["transcript"]

    # Build lookup using audio_path
    review_lookup = {}

    for _, row in review.iterrows():
        audio_path = str(row["audio_path"]).strip()
        second = clean_text(row["second_transcript"])

        if audio_path and second:
            review_lookup[audio_path] = second

    updated = 0

    for idx, row in master.iterrows():
        audio_path = str(row["audio_path"]).strip()

        if audio_path in review_lookup:
            new_text = review_lookup[audio_path]
            old_text = clean_text(row["transcript"])

            if new_text != old_text:
                master.at[idx, "transcript"] = new_text
                updated += 1

    # Recalculate text length if the column exists
    if "text_length" in master.columns:
        master["text_length"] = master["transcript"].fillna("").astype(str).str.len()

    master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("UPDATE COMPLETE")
    print("=" * 70)
    print(f"Updated transcripts : {updated}")
    print(f"Unchanged           : {len(master) - updated}")
    print(f"Total samples       : {len(master)}")
    print()
    print(f"Output:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()