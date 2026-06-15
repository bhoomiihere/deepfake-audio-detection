#!/usr/bin/env python3
"""
Deepfake Audio Detection — CLI Inference Script
================================================
Usage:
    python predict.py audio.wav
    python predict.py audio.wav --model deepfake_detector.h5

Output:
    Genuine (Human) or Deepfake (AI-Generated) + confidence score
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

def load_deps():
    try:
        import librosa
        import tensorflow as tf
        return librosa, tf
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install librosa tensorflow")
        sys.exit(1)

def extract_mfcc(file_path, config):
    librosa, _ = load_deps()
    sr         = config['sample_rate']
    duration   = config['duration']
    n_mfcc     = config['n_mfcc']
    n_fft      = config['n_fft']
    hop_length = config['hop_length']
    max_frames = config['max_frames']

    audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)

    # Fix length
    target = sr * duration
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.astype(np.float32)


def predict(audio_path, model_path='deepfake_detector.h5', config_path='model_config.json'):
    _, tf = load_deps()

    # Load config
    if not Path(config_path).exists():
        print(f"ERROR: {config_path} not found. Train the model first.")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    # Load model
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found. Train the model first.")
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)

    # Extract features and predict
    mfcc   = extract_mfcc(audio_path, config)
    inp    = mfcc[np.newaxis, ..., np.newaxis]           # (1, n_mfcc, T, 1)
    prob   = float(model.predict(inp, verbose=0)[0][0])   # deepfake probability
    label  = 'Deepfake (AI-Generated)' if prob > 0.5 else 'Genuine (Human)'
    conf   = prob if prob > 0.5 else 1 - prob

    return {
        'file': str(audio_path),
        'prediction': label,
        'confidence': conf,
        'deepfake_probability': prob,
        'genuine_probability': 1 - prob,
    }


def main():
    parser = argparse.ArgumentParser(description='Detect deepfake audio')
    parser.add_argument('audio', help='Path to audio file (.wav, .mp3, .flac)')
    parser.add_argument('--model',  default='deepfake_detector.h5',  help='Path to .h5 model')
    parser.add_argument('--config', default='model_config.json',      help='Path to model config')
    parser.add_argument('--json',   action='store_true',              help='Output as JSON')
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    result = predict(str(audio_path), args.model, args.config)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        bar  = '█' * int(result['confidence'] * 20) + '░' * (20 - int(result['confidence'] * 20))
        icon = '🤖' if 'Deepfake' in result['prediction'] else '✅'
        print()
        print("╔══════════════════════════════════════════════╗")
        print("║     DEEPFAKE AUDIO DETECTION RESULT          ║")
        print("╠══════════════════════════════════════════════╣")
        print(f"║  File       : {Path(result['file']).name[:30]:<30} ║")
        print(f"║  Prediction : {icon} {result['prediction']:<27} ║")
        print(f"║  Confidence : [{bar}] {result['confidence']*100:.1f}%  ║")
        print(f"║  Genuine    : {result['genuine_probability']*100:.1f}%                              ║")
        print(f"║  Deepfake   : {result['deepfake_probability']*100:.1f}%                              ║")
        print("╚══════════════════════════════════════════════╝")
        print()


if __name__ == '__main__':
    main()
