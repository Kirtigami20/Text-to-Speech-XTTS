from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
QC_FILE = ROOT / "reports" / "qc_review.csv"
OUTPUT = ROOT / "reports" / "retranscription_review.csv"

# Only ASR-repetition samples
df = pd.read_csv(QC_FILE)
df = df[df["flags"].astype(str).str.contains("REPETITION", na=False)].copy()

print("=" * 70)
print("SECOND-PASS WHISPER TRANSCRIPTION")
print("=" * 70)
print(f"Samples: {len(df)}")

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
)

results = []

for i, row in enumerate(df.itertuples(index=False), 1):
    audio_path = ROOT / row.audio_path

    print(f"\n[{i}/{len(df)}] {row.audio_path}")
    print(f"Original : {row.transcript}")

    segments, info = model.transcribe(
        str(audio_path),
        language=row.language,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
        },
    )

    new_text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    print(f"Second  : {new_text}")

    results.append({
        "audio_path": row.audio_path,
        "language": row.language,
        "original_transcript": row.transcript,
        "second_transcript": new_text,
    })

out = pd.DataFrame(results)
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("SECOND PASS COMPLETE")
print("=" * 70)
print(f"Output: {OUTPUT}")