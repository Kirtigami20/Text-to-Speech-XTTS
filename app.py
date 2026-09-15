import sys
import io
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import gradio as gr

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

XTTS_RUN_DIR = (
    ROOT
    / "training"
    / "xtts_output"
    / "rachana_xtts_clean_2epoch-August-27-2026_12+14PM-0000000"
)
XTTS_MODEL_PATH = XTTS_RUN_DIR / "best_model_1522.pth"
XTTS_CONFIG_PATH = XTTS_RUN_DIR / "config.json"

INDICF5_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ai4bharat--IndicF5"
    / "snapshots"
    / "ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
)
INDICF5_CKPT = str(INDICF5_SNAPSHOT / "model.safetensors")
INDICF5_VOCAB = str(INDICF5_SNAPSHOT / "checkpoints" / "vocab.txt")

DEFAULT_REF_AUDIO = ROOT / "processed" / "hindi" / "hi_000621.wav"
DEFAULT_REF_TEXT = "यही आपको validate करना है तो आप skip these questions को select करें"

OUTPUT_DIR = ROOT / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# GLOBAL MODEL CONTAINERS (Lazy Loaded)
# ============================================================

xtts_model = None
xtts_config = None

indicf5_model = None
indicf5_vocoder = None

# ============================================================
# MODEL LOADERS
# ============================================================

def load_xtts():
    global xtts_model, xtts_config
    if xtts_model is not None:
        return xtts_model, xtts_config

    print("Loading XTTS v2 Fine-Tuned Model...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(XTTS_CONFIG_PATH))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(XTTS_MODEL_PATH),
        eval=True,
    )
    model.to(DEVICE)

    xtts_model = model
    xtts_config = config
    print("XTTS v2 Loaded Successfully.")
    return xtts_model, xtts_config


def load_indicf5():
    global indicf5_model, indicf5_vocoder
    if indicf5_model is not None:
        return indicf5_model, indicf5_vocoder

    print("Loading IndicF5 Model & Vocoder...")
    from f5_tts.infer.utils_infer import load_vocoder, get_tokenizer, CFM
    from f5_tts.model import DiT
    from safetensors.torch import load_file

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=DEVICE)

    vocab_char_map, vocab_size = get_tokenizer(str(INDICF5_VOCAB), "custom")

    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )

    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    ).to(DEVICE)

    raw_ckpt = load_file(INDICF5_CKPT, device=DEVICE)
    clean_ckpt = {
        k.replace("ema_model.", "").replace("_orig_mod.", ""): v
        for k, v in raw_ckpt.items()
        if not k.startswith("vocoder.") and k not in ["initted", "step"]
    }

    model.load_state_dict(clean_ckpt, strict=False)

    indicf5_model = model
    indicf5_vocoder = vocoder
    print("IndicF5 Loaded Successfully.")
    return indicf5_model, indicf5_vocoder



# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    "01 - Hindi (हिंदी)": {
        "text": "नमस्ते, आज हम इनकम टैक्स और फाइनेंस के बारे में बात करेंगे।",
        "lang": "hi"
    },
    "02 - English": {
        "text": "Hello, today we are going to talk about income tax and finance.",
        "lang": "en"
    },
    "03 - Hinglish (Roman Hindi)": {
        "text": "Aaj hum income tax ke baare mein baat karenge aur dekhenge ki tax planning kaise karni hai.",
        "lang": "hi"
    },
    "04 - Mixed Hindi + English": {
        "text": "Aapko apna ITR file karna hai, aur uske baad tax calculation properly check karni hai.",
        "lang": "hi"
    },
    "05 - Marathi (मराठी)": {
        "text": "नमस्कार, आज आपण इनकम टॅक्स आणि फायनान्स बद्दल माहिती घेणार आहोत.",
        "lang": "hi"
    },
    "06 - Marnglish (Roman Marathi)": {
        "text": "Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot.",
        "lang": "hi"
    },
    "07 - Marnglish Natural": {
        "text": "Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara.",
        "lang": "hi"
    },
    "08 - Indian English": {
        "text": "If you are filing your income tax return in India, you should carefully check all your income details.",
        "lang": "en"
    },
    "Custom Text": {
        "text": "",
        "lang": "hi"
    }
}


# ============================================================
# GENERATION HANDLER
# ============================================================

def generate_speech(
    model_choice,
    text,
    language_code,
    ref_audio_path,
    ref_transcript,
    temperature,
    gpt_cond_len,
):
    if not text or not text.strip():
        return None, "Error: Text input cannot be empty."

    ref_audio = ref_audio_path if ref_audio_path else str(DEFAULT_REF_AUDIO)

    try:
        output_wav_path = OUTPUT_DIR / "interactive_generated_voice.wav"

        if model_choice == "Fine-Tuned XTTS v2 (Rachana)":
            model, config = load_xtts()
            
            result = model.synthesize(
                text=text.strip(),
                config=config,
                speaker_wav=ref_audio,
                language=language_code,
                gpt_cond_len=int(gpt_cond_len),
                temperature=float(temperature),
                enable_text_splitting=True,
            )

            wav = result["wav"]
            if torch.is_tensor(wav):
                wav = wav.detach().cpu().numpy()

            wav = np.asarray(wav)
            wav = np.squeeze(wav).astype(np.float32)
            sample_rate = 24000

            sf.write(str(output_wav_path), wav, sample_rate)
            duration = len(wav) / sample_rate
            status = f"✅ XTTS v2 Generated Successfully! Duration: {duration:.2f}s | Device: {DEVICE.upper()}"

        elif model_choice == "IndicF5 (AI4Bharat Zero-Shot)":
            model, vocoder = load_indicf5()
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

            ref_text_to_use = ref_transcript.strip() if (ref_transcript and ref_transcript.strip()) else DEFAULT_REF_TEXT
            
            ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
                str(ref_audio),
                ref_text_to_use,
            )

            audio_wave, sample_rate, _ = infer_process(
                ref_audio=ref_audio_proc,
                ref_text=ref_text_proc,
                gen_text=text.strip(),
                model_obj=model,
                vocoder=vocoder,
                mel_spec_type="vocos",
                device=DEVICE,
            )

            if audio_wave is None:
                return None, "❌ IndicF5 synthesis failed: No audio returned."

            if torch.is_tensor(audio_wave):
                audio_wave = audio_wave.detach().cpu().numpy()

            audio_wave = np.asarray(audio_wave)
            audio_wave = np.squeeze(audio_wave).astype(np.float32)

            sf.write(str(output_wav_path), audio_wave, sample_rate)
            duration = len(audio_wave) / sample_rate
            status = f"✅ IndicF5 Zero-Shot Generated Successfully! Duration: {duration:.2f}s | Device: {DEVICE.upper()}"


        else:
            return None, "Error: Unknown model selected."

        return str(output_wav_path), status

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        return None, f"❌ Error: {str(e)}"


def apply_preset(preset_name):
    if preset_name in PRESETS:
        p = PRESETS[preset_name]
        return p["text"], p["lang"]
    return "", "hi"


# ============================================================
# GRADIO UI INTERFACE
# ============================================================

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

with gr.Blocks(title="Rachana Multilingual TTS & Voice Cloning Studio") as demo:

    gr.Markdown(
        """
        # 🎙️ Rachana Multilingual TTS & Voice Cloning Studio
        Adjust settings, switch models, test preset benchmarks, or input custom text to generate high-fidelity speech.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):

            model_choice = gr.Radio(
                choices=["Fine-Tuned XTTS v2 (Rachana)", "IndicF5 (AI4Bharat Zero-Shot)"],
                value="Fine-Tuned XTTS v2 (Rachana)",
                label="🤖 Select Model",
                info="Choose between Fine-tuned XTTS v2 or Zero-shot IndicF5 Flow Matching."
            )

            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="01 - Hindi (हिंदी)",
                label="📝 Test Presets",
                info="Select pre-configured Hindi, English, Marathi, or code-switched benchmark prompts."
            )

            text_input = gr.Textbox(
                label="💬 Text Prompt to Synthesize",
                value=PRESETS["01 - Hindi (हिंदी)"]["text"],
                lines=4,
                placeholder="Enter text in Hindi (Devanagari/Roman), English, or Marathi..."
            )

            with gr.Row():
                language_code = gr.Dropdown(
                    choices=["hi", "en", "mr"],
                    value="hi",
                    label="🌐 Language Tag",
                    info="Language code passed to synthesizer."
                )

            with gr.Accordion("⚙️ Advanced Parameters & Reference Audio", open=False):

                ref_audio_input = gr.Audio(
                    value=str(DEFAULT_REF_AUDIO),
                    type="filepath",
                    label="🎙️ Reference Voice Clip (Conditioning Audio — Default is Rachana hi_000621.wav)"
                )

                ref_transcript_input = gr.Textbox(
                    value=DEFAULT_REF_TEXT,
                    label="📜 Reference Audio Transcript (Required for IndicF5)",
                    lines=2
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.2,
                    value=0.65,
                    step=0.05,
                    label="🔥 Temperature (Sampling Variance)",
                    info="Lower values = steady/monotone; Higher values = expressive/varied."
                )

                gpt_cond_len = gr.Slider(
                    minimum=3,
                    maximum=24,
                    value=12,
                    step=1,
                    label="⏱️ GPT Conditioning Length (Seconds)",
                    info="XTTS speaker conditioning duration."
                )

            generate_btn = gr.Button("✨ Synthesize Voice", variant="primary", size="lg")

        with gr.Column(scale=2):

            gr.Markdown("### 🔊 Output Audio Player")
            audio_output = gr.Audio(
                label="Synthesized Audio Result",
                type="filepath",
                autoplay=True
            )
            status_output = gr.Textbox(
                label="Status & Execution Diagnostics",
                interactive=False,
                lines=3
            )

    # Event handlers
    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[text_input, language_code]
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[
            model_choice,
            text_input,
            language_code,
            ref_audio_input,
            ref_transcript_input,
            temperature,
            gpt_cond_len,
        ],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        theme=theme,
        share=False,
        inbrowser=True,
    )



