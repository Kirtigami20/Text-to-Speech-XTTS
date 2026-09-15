from pathlib import Path
import csv
import time

from faster_whisper import WhisperModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_CSV = REPORT_DIR / "transcription_metadata.csv"

LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
}

MODEL_NAME = "large-v3-turbo"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Beam size: higher = potentially better accuracy, slower.
BEAM_SIZE = 5


# ============================================================
# HELPERS
# ============================================================

def load_existing_results():

    completed = {}

    if not OUTPUT_CSV.exists():
        return completed

    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            path = row["audio_path"]

            completed[path] = row

    return completed


def write_result(row):

    file_exists = OUTPUT_CSV.exists()

    fieldnames = [
        "segment_id",
        "audio_path",
        "language",
        "transcript",
        "duration_seconds",
        "source_file",
        "status",
    ]

    with open(
        OUTPUT_CSV,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def get_segment_id(audio_path):

    return audio_path.stem


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("MULTILINGUAL TTS — FULL TRANSCRIPTION")
    print("=" * 70)

    print()
    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {DEVICE}")
    print(f"Compute     : {COMPUTE_TYPE}")
    print(f"Beam size   : {BEAM_SIZE}")
    print()

    print("Loading Whisper Large-v3 Turbo...")

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Whisper loaded successfully.")

    existing = load_existing_results()

    print(
        f"\nAlready completed: "
        f"{len(existing)}"
    )

    total_files = 0

    for folder in LANGUAGES:

        language_folder = (
            PROCESSED_DIR / folder
        )

        files = sorted(
            language_folder.glob("*.wav")
        )

        total_files += len(files)

    print(
        f"Total audio files: "
        f"{total_files}"
    )

    completed_now = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    # ========================================================
    # PROCESS LANGUAGES
    # ========================================================

    for folder, language in LANGUAGES.items():

        language_folder = (
            PROCESSED_DIR / folder
        )

        files = sorted(
            language_folder.glob("*.wav")
        )

        print()
        print("=" * 70)
        print(
            f"{folder.upper()} ({language})"
        )
        print(
            f"Files: {len(files)}"
        )
        print("=" * 70)

        for index, audio_path in enumerate(
            files,
            start=1
        ):

            relative_path = str(
                audio_path.relative_to(
                    PROJECT_ROOT
                )
            ).replace("\\", "/")

            # ------------------------------------------------
            # RESUME SUPPORT
            # ------------------------------------------------

            if relative_path in existing:

                skipped += 1

                print(
                    f"[{index}/{len(files)}] "
                    f"SKIP: "
                    f"{audio_path.name}"
                )

                continue

            print()
            print(
                f"[{index}/{len(files)}] "
                f"Transcribing:"
            )

            print(
                f"  {audio_path.name}"
            )

            file_start = time.time()

            try:

                segments, info = model.transcribe(
                    str(audio_path),

                    language=language,

                    beam_size=BEAM_SIZE,

                    vad_filter=False,

                    condition_on_previous_text=False,

                    word_timestamps=False,
                )

                transcript_parts = []

                duration = 0.0

                for segment in segments:

                    text = segment.text.strip()

                    if text:
                        transcript_parts.append(
                            text
                        )

                    duration = max(
                        duration,
                        segment.end
                    )

                transcript = " ".join(
                    transcript_parts
                ).strip()

                elapsed = (
                    time.time() -
                    file_start
                )

                row = {
                    "segment_id":
                        get_segment_id(
                            audio_path
                        ),

                    "audio_path":
                        relative_path,

                    "language":
                        language,

                    "transcript":
                        transcript,

                    "duration_seconds":
                        round(
                            duration,
                            3
                        ),

                    "source_file":
                        "",

                    "status":
                        "success",
                }

                write_result(row)

                existing[
                    relative_path
                ] = row

                completed_now += 1

                print(
                    f"  ✓ {transcript}"
                )

                print(
                    f"  Time: "
                    f"{elapsed:.2f}s"
                )

            except Exception as e:

                failed += 1

                print(
                    f"  ✗ ERROR: {e}"
                )

                row = {
                    "segment_id":
                        get_segment_id(
                            audio_path
                        ),

                    "audio_path":
                        relative_path,

                    "language":
                        language,

                    "transcript":
                        "",

                    "duration_seconds":
                        "",

                    "source_file":
                        "",

                    "status":
                        f"error: {e}",
                }

                write_result(row)

    total_time = (
        time.time() -
        start_time
    )

    print()
    print("=" * 70)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 70)

    print(
        f"Newly transcribed : "
        f"{completed_now}"
    )

    print(
        f"Skipped           : "
        f"{skipped}"
    )

    print(
        f"Failed            : "
        f"{failed}"
    )

    print(
        f"Total processed   : "
        f"{completed_now + skipped}"
    )

    print(
        f"Time this run     : "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        f"Metadata CSV:"
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()