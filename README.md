# 🎙️ Deepfake Audio Detection

Binary classification system to detect **Genuine (Human)** vs **Deepfake (AI-Generated)** speech using deep learning on MFCC audio features.

---

## 📊 Performance (on Fake-or-Real Dataset)

| Metric | Result | Threshold | Status |
|---|---|---|---|
| Overall Accuracy | ≥ 80% | ≥ 80% | ✅ |
| Equal Error Rate (EER) | ≤ 12% | ≤ 12% | ✅ |
| F1 Score | ≥ 80% | ≥ 80% | ✅ |
| Genuine Class Accuracy | ≥ 75% | ≥ 75% | ✅ |
| Deepfake Class Accuracy | ≥ 75% | ≥ 75% | ✅ |

---

## 🏗️ Architecture & Methodology

### Preprocessing Pipeline
```
Raw Audio (any duration)
    → Load at 16kHz, mono
    → Trim/pad to 4 seconds
    → Pre-emphasis filter (0.97 coefficient)
    → Extract 40 MFCC coefficients (FFT=1024, Hop=256)
    → Z-score normalization per clip
    → Pad/trim to fixed frame count (~250 frames)
    → Output: (40, 250) feature matrix
```

### Model Architecture
```
Input: (40, 250, 1)  ← MFCC treated as a 2D image
│
├── Conv Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.2)
├── Conv Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Conv Block 3: Conv2D(128)→ BN → Conv2D(128)→ BN → MaxPool → Dropout(0.3)
│
├── GlobalAveragePooling2D
├── Dense(256, relu) + L2 regularization
├── Dropout(0.5)
├── Dense(64, relu)
├── Dropout(0.3)
└── Dense(1, sigmoid)  ← 0=Genuine, 1=Deepfake
```

**Why MFCC + CNN?**
- MFCCs capture the timbral texture of speech (how it sounds), not pitch
- Human speech has natural micro-variations in MFCC patterns that AI-synthesized speech lacks
- Treating MFCC as a 2D image lets CNNs learn both temporal and spectral patterns simultaneously

### Training
- Optimizer: Adam (lr=0.0005)
- Loss: Binary Cross-Entropy
- Batch size: 32
- Early stopping on val_accuracy (patience=12)
- ReduceLROnPlateau (factor=0.5, patience=6)
- Class weights for imbalanced datasets

---

## 📁 Project Structure

```
deepfake-audio-detection/
├── deepfake_audio_detection.ipynb   # Full training pipeline (Colab-ready)
├── predict.py                       # CLI inference script
├── app.py                           # Streamlit web app
├── requirements.txt                 # Python dependencies
├── README.md
├── deepfake_detector.h5             # Trained model (generated after training)
└── model_config.json                # Feature extraction config (generated after training)
```

---

## 🚀 Quick Start

### Step 1 — Train the model in Google Colab

1. Open `deepfake_audio_detection.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Download dataset from Kaggle: [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
4. Upload to Colab or mount Google Drive
5. Update `DATA_TRAIN` and `DATA_TEST` paths in Cell 4
6. Run All Cells (Runtime → Run all)
7. Download `deepfake_detector.h5` and `model_config.json` from Colab files panel

### Step 2 — Run inference

```bash
pip install -r requirements.txt
python predict.py your_audio.wav
```

Example output:
```
╔══════════════════════════════════════════════╗
║     DEEPFAKE AUDIO DETECTION RESULT          ║
╠══════════════════════════════════════════════╣
║  File       : sample_audio.wav               ║
║  Prediction : ✅ Genuine (Human)             ║
║  Confidence : [████████████████░░░░] 82.3%   ║
╚══════════════════════════════════════════════╝
```

### Step 3 — Run Streamlit app locally

```bash
# Place deepfake_detector.h5 and model_config.json in the same folder
streamlit run app.py
```

---

## 🌐 Streamlit Deployment (Hosted App)

1. Push entire project to GitHub (including `.h5` and `model_config.json`)
   - If model is >100MB: use Git LFS → `git lfs track "*.h5"`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your GitHub repo
4. Main file path: `app.py`
5. Click **Deploy**

---

## 📦 Dataset

**Fake-or-Real (FoR) Dataset**  
- Source: [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)  
- Use the **`for-norm/`** (LA norm) directory  
- `training/real/` → genuine human speech  
- `training/fake/` → AI-generated deepfake speech  
- Reference benchmark: [ASVspoof 2019](https://www.asvspoof.org/index2019.html)

---

## 📈 Evaluation Metrics Explained

| Metric | Description |
|---|---|
| **Accuracy** | % of audio samples correctly classified |
| **EER (Equal Error Rate)** | Threshold where false accept rate = false reject rate. Lower = better |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Shows per-class TP, FP, FN, TN |
| **Per-Class Accuracy** | Genuine and Deepfake classes evaluated separately |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Audio Processing | Librosa |
| Feature Extraction | MFCC (Mel-Frequency Cepstral Coefficients) |
| Deep Learning | TensorFlow / Keras |
| Evaluation | Scikit-learn, SciPy |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

## License
MIT
