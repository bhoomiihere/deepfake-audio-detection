"""
Deepfake Audio Detection — Streamlit Web App
=============================================
Deploy on Streamlit Community Cloud (free):
  1. Push this file + deepfake_detector.h5 + model_config.json to GitHub
  2. Go to share.streamlit.io → New app → select your repo
  3. Main file: app.py

Local run:
  streamlit run app.py
"""

import json
import tempfile
import os
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model & config (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_config():
    import tensorflow as tf
    model_path  = "deepfake_detector.h5"
    config_path = "model_config.json"

    if not Path(model_path).exists():
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(config_path) as f:
        config = json.load(f)
    return model, config


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_mfcc(audio_path, config):
    import librosa
    sr         = config['sample_rate']
    duration   = config['duration']
    n_mfcc     = config['n_mfcc']
    n_fft      = config['n_fft']
    hop_length = config['hop_length']
    max_frames = config['max_frames']

    audio, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    target = sr * duration
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # pre-emphasis

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]

    return audio, mfcc.astype(np.float32)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎙️ Deepfake Audio Detector")
st.markdown(
    "Upload a speech recording to detect whether it is **Genuine (Human)** "
    "or **Deepfake (AI-Generated)** using a CNN trained on MFCC features."
)
st.divider()

# Sidebar — Model info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Model:** 2D CNN on MFCC features  
**Dataset:** Fake-or-Real (LA norm)  
**Input:** 4-second audio clip  
**Features:** 40 MFCC coefficients  

**Performance:**
- Accuracy ≥ 80%
- EER ≤ 12%
- F1 Score ≥ 80%
    """)

# File uploader
uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Upload a .wav, .mp3, or .flac file (max 4 seconds used for analysis)"
)

if uploaded:
    # Play audio
    st.audio(uploaded, format=uploaded.type)

    # Save to temp file
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Analyzing audio..."):
        try:
            import librosa
            import librosa.display

            model, config = load_model_and_config()

            if model is None:
                st.error(
                    "**Model not found.**  \n"
                    "Make sure `deepfake_detector.h5` and `model_config.json` "
                    "are in the same folder as `app.py`."
                )
            else:
                # Extract features + predict
                audio, mfcc = extract_mfcc(tmp_path, config)
                inp   = mfcc[np.newaxis, ..., np.newaxis]
                prob  = float(model.predict(inp, verbose=0)[0][0])
                is_df = prob > 0.5
                conf  = prob if is_df else 1 - prob

                # ── Result card ──
                st.divider()
                if is_df:
                    st.error(f"## 🤖 DEEPFAKE DETECTED")
                else:
                    st.success(f"## ✅ GENUINE (Human)")

                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence",          f"{conf*100:.1f}%")
                col2.metric("Genuine Probability", f"{(1-prob)*100:.1f}%")
                col3.metric("Deepfake Probability",f"{prob*100:.1f}%")

                # Confidence bar
                st.markdown("### Confidence Breakdown")
                st.progress(float(prob), text=f"Deepfake ← {prob*100:.1f}% | {(1-prob)*100:.1f}% → Genuine")

                # ── Visualizations ──
                sr = config['sample_rate']
                hl = config['hop_length']

                tab1, tab2, tab3 = st.tabs(["📈 Waveform", "🌈 MFCC", "🎵 Mel Spectrogram"])

                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 2.5))
                    librosa.display.waveshow(audio, sr=sr, ax=ax, color='steelblue', alpha=0.8)
                    ax.set_xlabel("Time (s)")
                    ax.set_title("Audio Waveform")
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')
                    st.pyplot(fig); plt.close()

                with tab2:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr,
                                                   hop_length=hl, ax=ax, cmap='inferno')
                    fig.colorbar(img, ax=ax, label='Normalized value')
                    ax.set_title(f"MFCC ({config['n_mfcc']} coefficients)")
                    st.pyplot(fig); plt.close()

                with tab3:
                    raw_audio, _ = librosa.load(tmp_path, sr=sr,
                                                duration=config['duration'], mono=True)
                    mel = librosa.feature.melspectrogram(y=raw_audio, sr=sr,
                                                          n_fft=config['n_fft'],
                                                          hop_length=hl, n_mels=128)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                                                   x_axis='time', y_axis='mel',
                                                   sr=sr, hop_length=hl, ax=ax, cmap='magma')
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    ax.set_title("Mel Spectrogram")
                    st.pyplot(fig); plt.close()

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.unlink(tmp_path)

else:
    st.info("👆 Upload an audio file above to start detection.")

    # Example info
    with st.expander("📖 How does it work?"):
        st.markdown("""
1. **Preprocessing** — Audio is loaded at 16kHz and trimmed/padded to 4 seconds
2. **Feature Extraction** — 40 MFCC (Mel-Frequency Cepstral Coefficient) features extracted per frame
3. **Pre-emphasis** — High-frequency boost filter applied to improve signal quality
4. **CNN Model** — 3-block 2D Convolutional Neural Network processes the MFCC as an image
5. **Output** — Sigmoid output gives deepfake probability (> 0.5 = deepfake)
        """)
