from pathlib import Path
import csv
import json
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source"
REPORT_DIR = PROJECT_ROOT / "reports"

LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
}


def analyze_file(audio_path: Path, language: str):
    try:
        info = sf.info(audio_path)

        duration = info.duration
        sample_rate = info.samplerate
        channels = info.channels
        subtype = info.subtype

        audio, sr = sf.read(audio_path, always_2d=False)

        # Convert stereo to mono only for analysis.
        # Original file is NOT modified.
        if audio.ndim > 1:
            
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        mono = mono.astype(np.float64)

        if len(mono) == 0:
            return {
                "filename": audio_path.name,
                "language": language,
                "duration_seconds": 0,
                "sample_rate": sample_rate,
                "channels": channels,
                "subtype": subtype,
                "rms": 0,
                "peak": 0,
                "clipping_percentage": 0,
                "silence_percentage": 100,
                "file_size_mb": audio_path.stat().st_size / (1024 ** 2),
                "status": "EMPTY",
            }

        # RMS
        rms = float(np.sqrt(np.mean(mono ** 2)))

        # Peak
        peak = float(np.max(np.abs(mono)))

        # Clipping
        clipping_samples = np.sum(np.abs(mono) >= 0.999)
        clipping_percentage = float(
            clipping_samples / len(mono) * 100
        )

        # Simple silence estimation.
        # This is ONLY a rough analysis.
        silence_threshold = 0.01
        silence_samples = np.sum(np.abs(mono) < silence_threshold)

        silence_percentage = float(
            silence_samples / len(mono) * 100
        )

        warnings = []

        if rms < 0.01:
            warnings.append("VERY_LOW_VOLUME")

        if clipping_percentage > 1:
            warnings.append("CLIPPING")

        if silence_percentage > 60:
            warnings.append("HIGH_SILENCE")

        if duration < 10:
            warnings.append("VERY_SHORT")

        status = "OK" if not warnings else ";".join(warnings)

        return {
            "filename": audio_path.name,
            "language": language,
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "subtype": subtype,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "clipping_percentage": round(clipping_percentage, 4),
            "silence_percentage": round(silence_percentage, 2),
            "file_size_mb": round(
                audio_path.stat().st_size / (1024 ** 2), 3
            ),
            "status": status,
        }

    except Exception as e:
        return {
            "filename": audio_path.name,
            "language": language,
            "duration_seconds": 0,
            "sample_rate": None,
            "channels": None,
            "subtype": None,
            "rms": None,
            "peak": None,
            "clipping_percentage": None,
            "silence_percentage": None,
            "file_size_mb": None,
            "status": f"ERROR: {e}",
        }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 70)
    print("MULTILINGUAL FEMALE TTS — SOURCE AUDIO ANALYSIS")
    print("=" * 70)

    for folder_name, language_code in LANGUAGES.items():

        language_dir = SOURCE_DIR / folder_name

        if not language_dir.exists():
            print(f"\nWARNING: Missing folder: {language_dir}")
            continue

        files = sorted(language_dir.glob("*.wav"))

        print(f"\n{folder_name.upper()} ({language_code})")
        print(f"Files found: {len(files)}")

        for audio_file in files:
            print(f"  Analyzing: {audio_file.name}")

            result = analyze_file(
                audio_file,
                language_code
            )

            results.append(result)

    if not results:
        print("\nNo WAV files found.")
        print(f"Expected files inside: {SOURCE_DIR}")
        return

    # CSV
    csv_path = REPORT_DIR / "source_audio_report.csv"

    fieldnames = [
        "filename",
        "language",
        "duration_seconds",
        "sample_rate",
        "channels",
        "subtype",
        "rms",
        "peak",
        "clipping_percentage",
        "silence_percentage",
        "file_size_mb",
        "status",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # JSON
    json_path = REPORT_DIR / "source_audio_report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    total_duration = 0

    for language_code in ["en", "hi", "mr"]:

        language_results = [
            r for r in results
            if r["language"] == language_code
            and r["duration_seconds"] is not None
        ]

        duration = sum(
            r["duration_seconds"]
            for r in language_results
        )

        total_duration += duration

        print(
            f"{language_code.upper():<10} "
            f"{len(language_results):>3} files   "
            f"{duration / 3600:.2f} hours"
        )

    print("-" * 70)

    print(
        f"TOTAL      {len(results):>3} files   "
        f"{total_duration / 3600:.2f} hours"
    )

    print("\nReports:")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()