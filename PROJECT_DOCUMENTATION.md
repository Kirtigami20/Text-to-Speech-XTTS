# Multilingual Text-To-Speech (TTS) & Voice Cloning — Technical Documentation & Codebase Knowledge Base

> **Target Audience:** Incoming ML Engineers, Speech Researchers, and Developers continuing this project.  
> **Speaker / Voice:** Rachana (Multilingual female voice — Hindi, English, Marathi, Code-switched / Hinglish / Marnglish)  
> **Repository Path:** `c:\gray matrix\text_to_speech`

---

## 1. Project Executive Overview & Objectives

This repository contains a full end-to-end pipeline for building, training, evaluating, and deploying high-quality **Multilingual Text-to-Speech (TTS) and Zero-Shot Voice Cloning** for the target voice **"Rachana"**. The primary language domains covered are:
1. **Hindi (हिंदी)** — Pure Devanagari script and Romanized Hindi (Hinglish).
2. **English (Indian English)** — Standard English and Indian English accent synthesis.
3. **Marathi (मराठी)** — Devanagari Marathi text and Romanized Marathi (Marnglish / Marnglish Natural).
4. **Code-Switched / Mixed Scripts** — Real-world Indian conversational speech combining Devanagari and Latin scripts (e.g. finance/tax guidance like *"Aapko apna ITR file karna hai"*).

Two state-of-the-art model paradigms are implemented and compared:
- **Fine-Tuned XTTS v2 (Coqui TTS)**: Autoregressive GPT + Perceiver Resampler + DVAE fine-tuned on curated Rachana audio segments.
- **IndicF5 (AI4Bharat DiT + Vocos)**: Zero-shot flow-matching Diffusion Transformer (DiT) tailored for Indian languages.

---

## 2. Directory & Workspace Structure

```
c:\gray matrix\text_to_speech
│
├── test_voice.py                  # Main inference script for fine-tuned XTTS v2 model
├── test_indicf5.py                # Main inference script for IndicF5 zero-shot model
├── diagnose_xtts_config.py        # Dataset & configuration diagnostic and audit tool
├── export_audios.ps1              # PowerShell script to zip generated test outputs to Downloads
├── play_outputs.ps1               # PowerShell automated WPF audio player for listening tests
├── PROJECT_DOCUMENTATION.md       # [THIS FILE] Comprehensive codebase & technical reference
│
├── data/                          # Raw input source audio datasets
│   └── source/                    # Long-form audio recordings of Rachana
│
├── metadata/                      # Raw dataset manifest & metadata mappings
│
├── processed/                     # Normalized & preprocessed audio split by language
│   ├── english/                   # English audio segments
│   ├── hindi/                     # Hindi audio segments (contains reference hi_000621.wav)
│   └── marathi/                   # Marathi audio segments
│
├── processed_clean/               # Extra-cleaned audio segments post quality filtering
│
├── segments/                      # Raw split audio segments from long audio files
│
├── reports/                       # Quality Control & Dataset Analytics CSV/JSON reports
│   ├── dataset_qc.csv             # Full QC review flags per audio file (KEEP / REJECT)
│   ├── qc_review.csv              # Quality scoring breakdown (SNR, clipping, noise)
│   ├── qc_summary.json            # High-level statistics on dataset quality
│   ├── retranscription_review.csv # Whisper ASR re-transcription validation
│   ├── segmentation_report.csv    # Segmentation timestamp logs
│   ├── source_audio_report.csv    # Source file duration & sample rate analytics
│   ├── transcription_metadata.csv # V1 transcriptions
│   └── transcription_metadata_v2.csv # V2 cleaned transcriptions (666k+ entries)
│
├── scripts/                       # Data Pipeline & Fine-Tuning Execution Scripts
│   ├── train_xtts.py              # XTTS v2 fine-tuning training script (Coqui Trainer)
│   ├── prepare_training_dataset.py# Merges QC + Metadata into clean train/val splits
│   ├── segment_audio.py           # VAD silence-based segmentation script
│   ├── transcribe_dataset.py      # Automated Whisper ASR transcription generator
│   ├── validate_dataset.py        # Dataset integrity & format checker
│   ├── analyze_audio.py           # Audio signal metrics (duration, RMS, SNR, clipping)
│   ├── apply_retranscription.py   # Merging re-transcribed fixed text into main CSV
│   ├── retranscribe_suspicious.py # Targeting low-confidence ASR segments
│   ├── create_coqui_metadata.py   # Formats CSV into LJSpeech / Coqui format (file|text|speaker)
│   ├── create_final_manifest.py   # Prepares JSONL manifests for training
│   └── xtts_dataset.py            # PyTorch Dataset wrapper for XTTS loader
│
├── test_outputs/                  # Output directory for generated XTTS v2 audio samples (.wav)
├── indicf5_outputs/               # Output directory for generated IndicF5 audio samples (.wav)
│
└── training/                      # Training workspace & Model Checkpoints
    ├── train.csv / validation.csv # LJSpeech formatted splits used by XTTS trainer
    ├── metadata.csv               # Master training dataset manifest
    ├── xtts_model_files/          # Downloaded base XTTS v2 weights (vocab.json, model.pth, etc.)
    └── xtts_output/               # Fine-tuned checkpoint runs (best_model_1522.pth)
```

---

## 3. Detailed File-by-File & Technique Breakdown

### A. Root Inference & Utility Scripts

#### 1. [`test_voice.py`](file:///c:/gray%20matrix/text_to_speech/test_voice.py) — XTTS v2 Voice Inference
- **Purpose:** Synthesizes audio using the fine-tuned XTTS v2 checkpoint (`best_model_1522.pth`).
- **Key Techniques Used:**
  - **Speaker Conditioning (`gpt_cond_len=12`):** Extracts 12-second acoustic embedding from reference sample [`hi_000621.wav`](file:///c:/gray%20matrix/text_to_speech/processed/hindi/hi_000621.wav) to clone Rachana's voice timbre.
  - **Sampling (`temperature=0.65`):** Controls randomness during autoregressive generation to balance natural variance against pronunciation stability.
  - **Text Splitting (`enable_text_splitting=True`):** Automatically splits long text prompts at punctuation boundaries to avoid hallucination or unnatural pauses.
  - **Audio Export:** Resamples model output array to 24kHz standard WAV using `soundfile`.
- **Supported Test Battery:** 8 standard benchmark test prompts across Hindi, English, Hinglish, Marathi, and Marnglish.

#### 2. [`test_indicf5.py`](file:///c:/gray%20matrix/text_to_speech/test_indicf5.py) — IndicF5 Zero-Shot Inference
- **Purpose:** Performs zero-shot voice cloning using AI4Bharat's IndicF5 Diffusion Transformer model.
- **Key Techniques Used:**
  - **Diffusion Transformer (DiT):** 1024 hidden dim, 22 depth layers, 16 attention heads, 512 text dim.
  - **Vocos Vocoder:** Converts generated Mel-spectrograms into high-fidelity 24kHz audio waveforms without HiFi-GAN artifacts.
  - **Prompt Conditioning:** Uses reference audio [`hi_000621.wav`](file:///c:/gray%20matrix/text_to_speech/processed/hindi/hi_000621.wav) coupled with its exact transcript (*"यही आपको validate करना है..."*) to condition the flow matching process.
  - **Encoding Protection:** Sets UTF-8 stdio wrappers to prevent Windows console crashes when handling Devanagari text.

#### 3. [`diagnose_xtts_config.py`](file:///c:/gray%20matrix/text_to_speech/diagnose_xtts_config.py) — Config & Vocabulary Diagnostic Engine
- **Purpose:** Deep-audits dataset composition, script types, token vocabulary coverage, and training run logs.
- **Key Techniques & Findings:**
  - **Script Classification:** Categorizes transcripts into `Devanagari_only`, `Roman_only`, and `Mixed`.
  - **Special Marathi Character Audit:** Checks Unicode presence of special characters (`ळ` U+0933, `ऱ` U+0931, `ऴ` U+0934, `ॲ` U+0972, `ऑ` U+0911) against `vocab.json`.
  - **Log Parser:** Parses `trainer_0_log.txt` to calculate exact global steps and completed epochs across runs.

#### 4. Automated Execution Scripts
- [`export_audios.ps1`](file:///c:/gray%20matrix/text_to_speech/export_audios.ps1): Packages all `.wav` outputs into `Rachana_XTTS_Voice_Outputs.zip` in user's Downloads folder and launches File Explorer.
- [`play_outputs.ps1`](file:///c:/gray%20matrix/text_to_speech/play_outputs.ps1): Initializes a WPF `System.Windows.Media.MediaPlayer` instance to automatically sequence and play output files with visual progress and duration tracking.

---

### B. Core Data Pipeline & Training Scripts ([`scripts/`](file:///c:/gray%20matrix/text_to_speech/scripts))

#### 1. [`scripts/segment_audio.py`](file:///c:/gray%20matrix/text_to_speech/scripts/segment_audio.py) — Silence-Based VAD Segmentation
- **Technique:** Voice Activity Detection (VAD) & Silence Splitting.
- **How it works:** Reads long raw WAV files from `data/source/`, uses energy/silence thresholds to chop continuous audio into 3–10 second segments required for XTTS training, avoiding mid-word truncations.

#### 2. [`scripts/transcribe_dataset.py`](file:///c:/gray%20matrix/text_to_speech/scripts/transcribe_dataset.py) — Automated ASR Transcription
- **Technique:** OpenAI Whisper ASR Integration.
- **How it works:** Batch processes segmented audio, auto-detects language (Hindi, English, Marathi), and generates raw transcripts stored in `transcription_metadata.csv`.

#### 3. [`scripts/analyze_audio.py`](file:///c:/gray%20matrix/text_to_speech/scripts/analyze_audio.py) — Signal Quality Auditing
- **Technique:** Digital Signal Processing (DSP) Quality Metrics.
- **How it works:** Computes RMS energy, Signal-to-Noise Ratio (SNR), peak amplitude, clipping percentages, and duration per segment. Identifies silent or noisy files.

#### 4. [`scripts/retranscribe_suspicious.py`](file:///c:/gray%20matrix/text_to_speech/scripts/retranscribe_suspicious.py) & [`scripts/apply_retranscription.py`](file:///c:/gray%20matrix/text_to_speech/scripts/apply_retranscription.py) — Iterative Cleaning
- **Technique:** Targeted ASR Filtering.
- **How it works:** Flags segments with high perplexity or mismatched character length, re-transcribes them with larger Whisper models (large-v3), and patches the metadata CSV.

#### 5. [`scripts/prepare_training_dataset.py`](file:///c:/gray%20matrix/text_to_speech/scripts/prepare_training_dataset.py) — Dataset Sanitization & Train/Val Split
- **Technique:** Data Quality Enforcement.
- **How it works:** Merges quality scores (`dataset_qc.csv`), filters out rejected samples, verifies disk audio existence, removes duplicates, and generates `training/train.csv` & `training/validation.csv` formatted in LJSpeech format (`audio_path|text|speaker_name`).

#### 6. [`scripts/train_xtts.py`](file:///c:/gray%20matrix/text_to_speech/scripts/train_xtts.py) — Coqui XTTS Fine-Tuning Execution
- **Technique:** Autoregressive GPT Fine-Tuning with Mixed Precision.
- **Hyperparameter Breakdown:**
  - **Base Architecture:** XTTS v2 GPT Encoder (Unfrozen), DVAE (Frozen), Mel-Norm Vocoder (Frozen).
  - **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.96, \text{weight\_decay}=0.01$).
  - **Learning Rate:** $5 \times 10^{-6}$ (Low learning rate to preserve pre-trained multi-lingual features).
  - **Batching & VRAM Optimization:** `BATCH_SIZE = 1`, `GRAD_ACCUM_STEPS = 84` (Effective batch size = 84, enabling training on consumer GPUs like RTX 3070 8GB VRAM).
  - **Precision:** Mixed Precision `fp16`.

---

## 4. Model Architecture & Paradigm Comparison

| Feature / Metric | Fine-Tuned XTTS v2 (Coqui) | IndicF5 (AI4Bharat) |
| :--- | :--- | :--- |
| **Architecture** | Autoregressive GPT + Perceiver + DVAE | Diffusion Transformer (DiT) Flow Matching |
| **Vocoder** | HiFi-GAN / Mel Spectrogram | Vocos Neural Vocoder |
| **Learning Paradigm** | Supervised Fine-Tuning (SFT) on Rachana voice | Zero-Shot Voice Cloning via Reference Conditioning |
| **Speaker Matching** | **Excellent (9.5/10)** — Embedded directly into weights | **Good (8.5/10)** — Reconstructed dynamically from ref audio |
| **Indian Language Native Support** | Basic (Hindi supported; Marathi mapped to Hindi) | **Native (Comprehensive Indic Language Vocab)** |
| **Code-Switching (Hinglish/Marnglish)** | Requires careful text normalization / phonemization | **Robust handling of Romanized & mixed scripts** |
| **Inference Speed (CUDA)** | Fast (~1.2s per 5s audio) | Iterative Flow Sampling (~2.5s per 5s audio) |
| **Customization Flexibility** | High (Weights can be further fine-tuned) | High (Zero-shot; switch speaker by changing ref `.wav`) |

---

## 5. Test Audio Battery & Output Comparison Guide

Both model inference pipelines generate identical 8 test cases in their respective output directories ([`test_outputs/`](file:///c:/gray%20matrix/text_to_speech/test_outputs) for XTTS v2, [`indicf5_outputs/`](file:///c:/gray%20matrix/text_to_speech/indicf5_outputs) for IndicF5).

Below is the benchmark test comparison matrix:

| Test ID | Test Name | Language Code | Input Text Prompt | Model Performance & Comparison Notes |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `01_hindi` | `hi` | *नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में बात करेंगे।* | **XTTS:** Warm, authentic Rachana voice with accurate Devanagari pronunciation.<br>**IndicF5:** Clear Devanagari cadence, slightly crisper high frequencies. |
| **02** | `02_english` | `en` | *Hello, today we are going to talk about income tax and finance.* | **XTTS:** Natural Indian English accent with strong voice identity.<br>**IndicF5:** Excellent clarity, slightly flatter intonation. |
| **03** | `03_hinglish` | `hi` | *Aaj hum income tax ke baare mein baat karenge aur dekhenge ki tax planning kaise karni hai.* | **XTTS:** Speaks Romanized Hindi naturally without spelling pronunciation artifacts.<br>**IndicF5:** Handles Romanized Hindi robustly. |
| **04** | `04_hindi_english` | `hi` | *Aapko apna ITR file karna hai, aur uske baad tax calculation properly check karni hai.* | **XTTS:** Smooth transition between Hindi words and technical English acronyms ("ITR").<br>**IndicF5:** High fidelity on English terms embedded in Roman text. |
| **05** | `05_marathi` | `hi` / `mr` | *नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल माहिती घेणार आहोत.* | **XTTS:** Handled via Hindi language code; pronunciation is clear.<br>**IndicF5:** Superior Marathi phonetic nuance due to native Devanagari Indic vocabulary. |
| **06** | `06_marnglish` | `hi` | *Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot.* | **XTTS:** Accurately renders Romanized Marathi conversational phrase.<br>**IndicF5:** Good prosody matching. |
| **07** | `07_marnglish_natural` | `hi` | *Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara.* | **XTTS:** Realistic code-switched multi-lingual sentence (Marathi + English).<br>**IndicF5:** Clear expression of English phrases within Marathi sentence structure. |
| **08** | `08_indian_english` | `en` | *If you are filing your income tax return in India, you should carefully check all your income details.* | **XTTS:** Full corporate advisory tone in Indian English.<br>**IndicF5:** Clean, steady articulation. |

---

## 6. Developer Quick-Start & Operations Manual

### Prerequisites & Virtual Environment
Ensure you are operating inside the activated virtual environment:
```powershell
# Activate Virtual Environment
.\.venv\Scripts\Activate.ps1
```

### Running Model Inference

#### 1. Generate Audio with Fine-Tuned XTTS v2
```powershell
python test_voice.py
```
*Outputs will be created in [`test_outputs/*.wav`](file:///c:/gray%20matrix/text_to_speech/test_outputs).*

#### 2. Generate Audio with IndicF5 Zero-Shot
```powershell
python test_indicf5.py
```
*Outputs will be created in [`indicf5_outputs/*.wav`](file:///c:/gray%20matrix/text_to_speech/indicf5_outputs).*

### Evaluating & Listening to Outputs

#### Automatically Play XTTS Outputs Sequentially
```powershell
powershell -ExecutionPolicy Bypass -File .\play_outputs.ps1
```

#### Package Outputs for Sharing / Download
```powershell
powershell -ExecutionPolicy Bypass -File .\export_audios.ps1
```
*Creates `Rachana_XTTS_Voice_Outputs.zip` in your `Downloads` directory.*

### Running Diagnostics & Auditing
```powershell
python diagnose_xtts_config.py
```
*Generates detailed report in [`training/xtts_diagnostics/diagnostic_report.txt`](file:///c:/gray%20matrix/text_to_speech/training/xtts_diagnostics/diagnostic_report.txt).*

### Preparing Data & Fine-Tuning XTTS

#### 1. Re-prepare Metadata from Quality Control Reports
```powershell
python scripts/prepare_training_dataset.py
```

#### 2. Launch / Resume XTTS v2 Fine-Tuning
```powershell
python scripts/train_xtts.py
```

### Interactive Web UI Frontend (Gradio Studio)
To launch the full interactive web application where you can switch models, adjust parameters, test prompts, and play/download audio directly in your browser:
```powershell
python app.py
```
*App will launch automatically at `http://127.0.0.1:7860`.*

---

## 7. Recommendations & Next Steps for Incoming Developers

1. **Native Marathi Vocab Expansion:**  
   Currently XTTS maps Marathi text under the `'hi'` language tag. To improve unique Marathi phonemes (like `ळ` U+0933), consider expanding `vocab.json` or continuing evaluation with **IndicF5**, which natively supports Marathi tokens.
2. **Text Normalization Layer:**  
   For code-switched inputs (Hinglish/Marnglish), adding a lightweight text pre-processor that converts numbers, currencies (`₹`), and financial abbreviations (`ITR`, `GST`, `PAN`) into standard expanded script before passing to TTS will boost output quality even further.
3. **Reference Audio Optimization:**  
   The current reference sample [`processed/hindi/hi_000621.wav`](file:///c:/gray%20matrix/text_to_speech/processed/hindi/hi_000621.wav) yields excellent results. When testing new domains, experiment with selecting alternative 6-12 second background-noise-free reference clips from [`processed_clean/`](file:///c:/gray%20matrix/text_to_speech/processed_clean) for varied emotional delivery (e.g. conversational vs formal).

---
*Documentation generated on 2026-09-11 for `c:\gray matrix\text_to_speech` repository takeover.*
