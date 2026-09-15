from pathlib import Path
import csv

import torch
import soundfile as sf
import numpy as np

from tqdm import tqdm
from silero_vad import load_silero_vad, get_speech_timestamps


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "source"
OUTPUT_DIR = PROJECT_ROOT / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
}

TARGET_SR = 24000
VAD_SR = 16000

MIN_DURATION = 2.0
MAX_DURATION = 15.0

PADDING = 0.15


def load_audio(path):
    """
    Load WAV using SoundFile.
    This completely avoids TorchCodec/TorchAudio decoding.
    """

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False
    )

    # Stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def resample_audio(audio, original_sr, target_sr):
    """
    Resample using scipy.
    """

    if original_sr == target_sr:
        return audio

    from scipy.signal import resample_poly

    gcd = np.gcd(original_sr, target_sr)

    up = target_sr // gcd
    down = original_sr // gcd

    resampled = resample_poly(
        audio,
        up,
        down
    )

    return resampled.astype(np.float32)


def save_segment(audio, path):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    sf.write(
        str(path),
        audio,
        TARGET_SR,
        subtype="PCM_16"
    )


def process_file(audio_path, language, vad_model):

    audio, sample_rate = load_audio(audio_path)

    print(
        f"\n    Source: "
        f"{audio_path.name}"
    )

    print(
        f"    Sample rate: "
        f"{sample_rate}"
    )

    print(
        f"    Duration: "
        f"{len(audio) / sample_rate:.2f}s"
    )

    # Resample to 16 kHz for Silero VAD.
    vad_audio = resample_audio(
        audio,
        sample_rate,
        VAD_SR
    )

    # Silero expects torch tensor.
    vad_tensor = torch.from_numpy(
        vad_audio
    )

    speech_timestamps = get_speech_timestamps(
        vad_tensor,
        vad_model,
        sampling_rate=VAD_SR,
        threshold=0.5,
        min_speech_duration_ms=300,
        min_silence_duration_ms=400,
        speech_pad_ms=150,
        return_seconds=False,
    )

    # Resample original audio to target TTS rate.
    target_audio = resample_audio(
        audio,
        sample_rate,
        TARGET_SR
    )

    segments = []

    for timestamp in speech_timestamps:

        start_sec = (
            timestamp["start"] /
            VAD_SR
        )

        end_sec = (
            timestamp["end"] /
            VAD_SR
        )

        # Add padding.
        start_sec = max(
            0,
            start_sec - PADDING
        )

        end_sec = min(
            len(target_audio) / TARGET_SR,
            end_sec + PADDING
        )

        duration = end_sec - start_sec

        if duration < MIN_DURATION:
            continue

        # Split long speech regions.
        current = start_sec

        while current < end_sec:

            segment_end = min(
                current + MAX_DURATION,
                end_sec
            )

            segment_duration = (
                segment_end - current
            )

            if segment_duration >= MIN_DURATION:

                start_sample = int(
                    current * TARGET_SR
                )

                end_sample = int(
                    segment_end * TARGET_SR
                )

                segment_audio = target_audio[
                    start_sample:end_sample
                ]

                segments.append({
                    "audio": segment_audio,
                    "language": language,
                    "source_file": audio_path.name,
                    "start": round(
                        current,
                        3
                    ),
                    "end": round(
                        segment_end,
                        3
                    ),
                    "duration": round(
                        segment_duration,
                        3
                    ),
                })

            current = segment_end

    return segments


def main():

    print("=" * 70)
    print("MULTILINGUAL TTS — VAD SEGMENTATION")
    print("=" * 70)

    print("\nDevice information:")

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nLoading Silero VAD...")

    vad_model = load_silero_vad()

    print("Silero VAD loaded.")

    all_records = []

    global_segment_id = 0

    total_duration = 0.0

    for folder, language in LANGUAGES.items():

        source_folder = (
            SOURCE_DIR / folder
        )

        output_folder = (
            OUTPUT_DIR / folder
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        files = sorted(
            source_folder.glob("*.wav")
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"{folder.upper()} "
            f"({language})"
        )

        print(
            f"Files: {len(files)}"
        )

        print(
            f"{'=' * 60}"
        )

        language_segments = 0
        language_duration = 0.0

        for audio_file in tqdm(
            files,
            desc=folder
        ):

            segments = process_file(
                audio_file,
                language,
                vad_model
            )

            for segment in segments:

                global_segment_id += 1

                segment_id = (
                    f"{language}_"
                    f"{global_segment_id:06d}"
                )

                output_path = (
                    output_folder /
                    f"{segment_id}.wav"
                )

                save_segment(
                    segment["audio"],
                    output_path
                )

                record = {
                    "segment_id": segment_id,
                    "audio_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ).replace("\\", "/"),
                    "language": language,
                    "source_file": segment[
                        "source_file"
                    ],
                    "start_seconds": segment[
                        "start"
                    ],
                    "end_seconds": segment[
                        "end"
                    ],
                    "duration_seconds": segment[
                        "duration"
                    ],
                }

                all_records.append(record)

                language_segments += 1
                language_duration += (
                    segment["duration"]
                )

                total_duration += (
                    segment["duration"]
                )

        print(
            f"\n{folder}:"
        )

        print(
            f"  Segments: "
            f"{language_segments}"
        )

        print(
            f"  Speech duration: "
            f"{language_duration / 3600:.2f} hours"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORT_DIR /
        "segmentation_report.csv"
    )

    fieldnames = [
        "segment_id",
        "audio_path",
        "language",
        "source_file",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
    ]

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\n")
    print("=" * 70)
    print("SEGMENTATION COMPLETE")
    print("=" * 70)

    print(
        f"Total segments : "
        f"{global_segment_id}"
    )

    print(
        f"Total duration : "
        f"{total_duration / 3600:.2f} hours"
    )

    print(
        f"Report         : "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()