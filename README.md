# 🎙️ Multilingual TTS & Voice Cloning

> Fine-tuned **XTTS v2** and zero-shot **IndicF5** voice cloning for Hindi, English, Marathi, Hinglish, and Marnglish — with an interactive Gradio web UI.

---

## 📌 Overview

This project builds a production-ready **Multilingual Text-to-Speech (TTS) pipeline** around the voice of **Rachana**, a female speaker covering:

| Language | Script | Example |
|:---|:---|:---|
| Hindi | Devanagari | नमस्ते, आज हम इनकम टैक्स के बारे में बात करेंगे। |
| English | Latin | Hello, today we are going to talk about income tax. |
| Hinglish | Latin (Roman Hindi) | Aaj hum income tax ke baare mein baat karenge. |
| Marathi | Devanagari | नमस्कार, आज आपण इनकम टॅक्स बद्दल माहिती घेणार आहोत. |
| Marnglish | Latin (Roman Marathi) | Tumhala ITR file karaycha asel tar check kara. |
| Mixed / Code-switched | Devanagari + Latin | Aapko apna ITR file karna hai, tax calculation check karni hai. |

Two models are implemented and compared:

- **🔵 Fine-Tuned XTTS v2** — Autoregressive GPT + Perceiver Resampler fine-tuned on Rachana's cleaned audio dataset.
- **🟢 IndicF5 (AI4Bharat)** — Zero-shot Diffusion Transformer (DiT) with Vocos vocoder; no training required.

---

## 🖥️ Interactive Web UI

Launch the Gradio-based web studio to generate speech interactively:

```bash
python app.py
```

Then open **`http://127.0.0.1:7860`** in your browser.

**Features:**
- Switch between XTTS v2 and IndicF5 with one click
- 8 pre-loaded multilingual test prompts
- Custom text input in any supported script
- Adjust Temperature and GPT Conditioning Length
- Upload custom reference voice clips
- Instant in-browser audio playback & download

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Kirtigami20/Text-to-Speech-XTTS.git
cd Text-to-Speech-XTTS
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install coqui-tts f5-tts gradio soundfile torch torchaudio safetensors
```

> ⚠️ Requires **CUDA-capable GPU** (tested on NVIDIA RTX 3070 8GB). CPU inference is supported but significantly slower.

### 4. Place your model files

Put your fine-tuned XTTS v2 checkpoint under:
```
training/xtts_output/<run-name>/best_model_XXXX.pth
training/xtts_output/<run-name>/config.json
```

IndicF5 weights are auto-downloaded from HuggingFace on first run (`ai4bharat/IndicF5`).

### 5. Place a reference voice clip

Default reference audio: `processed/hindi/hi_000621.wav`
Any 6–12 second clean WAV file of the target speaker will work.

---

## 🗂️ Project Structure

```
rachana-tts/
│
├── app.py                        # Gradio interactive web UI
├── test_voice.py                 # XTTS v2 batch inference (8 test cases)
├── test_indicf5.py               # IndicF5 batch inference (8 test cases)
├── diagnose_xtts_config.py       # Dataset & vocabulary diagnostic tool
├── export_audios.ps1             # Package test_outputs/ as ZIP to Downloads
├── play_outputs.ps1              # Auto-play test_outputs/ sequentially
├── PROJECT_DOCUMENTATION.md     # Full technical codebase knowledge base
│
├── scripts/                      # Data pipeline & training scripts
│   ├── segment_audio.py          # VAD-based silence splitting
│   ├── transcribe_dataset.py     # Whisper ASR auto-transcription
│   ├── analyze_audio.py          # SNR, RMS, clipping quality checks
│   ├── retranscribe_suspicious.py# Re-transcribe low-confidence segments
│   ├── apply_retranscription.py  # Patch corrected transcripts into CSV
│   ├── prepare_training_dataset.py # QC filtering + train/val split
│   ├── create_coqui_metadata.py  # Format metadata to LJSpeech CSV
│   ├── create_final_manifest.py  # Build JSONL training manifests
│   ├── train_xtts.py             # XTTS v2 fine-tuning (Coqui Trainer)
│   └── xtts_dataset.py           # PyTorch Dataset for XTTS loader
│
├── data/source/                  # Raw long-form source audio
├── processed/                    # Segmented audio split by language
│   ├── hindi/
│   ├── english/
│   └── marathi/
├── test_outputs/                 # XTTS v2 generated samples
└── indicf5_outputs/              # IndicF5 generated samples
```

---

## 🤖 Models

### Fine-Tuned XTTS v2

| Parameter | Value |
|:---|:---|
| Base Model | Coqui XTTS v2 |
| Architecture | Autoregressive GPT + Perceiver Resampler + DVAE |
| Vocoder | HiFi-GAN / Mel Spectrogram |
| Training Hardware | NVIDIA RTX 3070 8GB |
| Optimizer | AdamW (lr=5e-6, β₁=0.9, β₂=0.96) |
| Batch Size | 1 × 84 grad accum steps (eff. 84) |
| Precision | fp16 mixed precision |
| Languages | `hi`, `en`, `mr` (Marathi trained under `hi`) |
| Output Sample Rate | 24 kHz |

### IndicF5 (Zero-Shot)

| Parameter | Value |
|:---|:---|
| Source | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) |
| Architecture | Diffusion Transformer (DiT) — Flow Matching |
| Vocoder | Vocos (24 kHz) |
| Inference | Zero-shot (reference audio + transcript) |
| Model Dim | 1024, 22 layers, 16 heads |

---

## 🧪 Benchmark Test Prompts

Both models use the same 8 test cases for direct comparison:

| # | Name | Language | Prompt |
|:--|:--|:--|:--|
| 01 | hindi | `hi` | नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में बात करेंगे। |
| 02 | english | `en` | Hello, today we are going to talk about income tax and finance. |
| 03 | hinglish | `hi` | Aaj hum income tax ke baare mein baat karenge aur dekhenge ki tax planning kaise karni hai. |
| 04 | hindi_english | `hi` | Aapko apna ITR file karna hai, aur uske baad tax calculation properly check karni hai. |
| 05 | marathi | `hi` | नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल माहिती घेणार आहोत. |
| 06 | marnglish | `hi` | Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot. |
| 07 | marnglish_natural | `hi` | Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara. |
| 08 | indian_english | `en` | If you are filing your income tax return in India, you should carefully check all your income details. |

---

## 🛠️ Data Pipeline

The full pipeline from raw audio to trained model:

```
Raw Audio (data/source/)
        ↓
[segment_audio.py]         — VAD silence-based splitting → segments/
        ↓
[transcribe_dataset.py]    — Whisper ASR auto-transcription → reports/
        ↓
[analyze_audio.py]         — SNR / RMS / clipping QC → reports/dataset_qc.csv
        ↓
[retranscribe_suspicious.py] — Fix low-confidence segments
        ↓
[prepare_training_dataset.py] — QC filter + dedup → training/train.csv, validation.csv
        ↓
[scripts/train_xtts.py]    — XTTS v2 fine-tuning → training/xtts_output/
```

---

## 🔊 Utilities

### Play all test outputs sequentially
```powershell
powershell -ExecutionPolicy Bypass -File .\play_outputs.ps1
```

### Export test outputs as ZIP to Downloads
```powershell
powershell -ExecutionPolicy Bypass -File .\export_audios.ps1
```

### Run full dataset & config diagnostics
```bash
python diagnose_xtts_config.py
```

---

## ⚠️ Known Limitations

- **Marathi** is not a native language in XTTS v2 — it is trained and inferred under the `hi` language tag. For better Marathi phoneme accuracy, use **IndicF5**.
- **IndicF5 requires** both a reference audio clip and its **exact transcript** for optimal zero-shot voice cloning quality.
- Training was done on a consumer RTX 3070 8GB GPU with gradient accumulation to compensate for VRAM limits.

---

## 📄 License

This project uses:
- [Coqui TTS](https://github.com/coqui-ai/TTS) — Mozilla Public License 2.0
- [F5-TTS / IndicF5](https://github.com/SWivid/F5-TTS) — MIT License
- Fine-tuned weights and audio data are proprietary to this project.

---

## 🙏 Acknowledgements

- [Coqui AI](https://github.com/coqui-ai/TTS) for the XTTS v2 architecture and Trainer framework.
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicF5 Indic language TTS model.
- [charactr/vocos](https://github.com/hubert-siuzdak/vocos) for the Vocos neural vocoder.
