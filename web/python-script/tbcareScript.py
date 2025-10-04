import os
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import tensorflow_hub as hub
import librosa
import json
from copy import deepcopy
import speechproc
from scipy.signal import lfilter

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")
MODEL_PATH = os.path.join('python-script', 'models_tbcare', 'model2.keras')
SCALER_PATH = os.path.join('python-script', 'models_tbcare', 'scaler2.pkl')
YAMNET_MAP_PATH = os.path.join(script_dir, 'yamnet_class_map.csv')

print(MODEL_PATH)
try:
    with open(f"{MODEL_PATH}/config.json") as f:
        config = json.load(f)
    model = tf.keras.models.model_from_json(json.dumps(config),
                                            custom_objects={'TFSMLayer': hub.KerasLayer})
    model.load_weights(f"{MODEL_PATH}/model.weights.h5")
    
    scaler = joblib.load(SCALER_PATH)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = list(pd.read_csv(YAMNET_MAP_PATH)['display_name'])
except Exception as e:
    print(json.dumps({"status": "error", "message": f"Error loading models or dependencies: {e}"}))
    sys.exit(1)

def getVad(data, fs):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1
    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)
    pv01 = np.zeros(nfr10)
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)
    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)
    noise_samp, _, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )
    for j in range(n_noise_samp):
        fdata[round(noise_samp[j, 0]): round(noise_samp[j, 1]) + 1] = 0
    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )
    return vad_seg

def segment_audio(file_path):
    X, sample_rate = librosa.load(file_path, sr=22050, mono=True)
    fvad = getVad(X, sample_rate)
    list_X, temp = [], []
    for i in range(1, len(fvad)):
        if fvad[i - 1] == 1:
            len_sector = math.floor(len(X) / len(fvad))
            start = (i - 1) * len_sector
            for j in range(start, start + len_sector):
                if j < len(X):
                    temp.append(X[j])
        if fvad[i - 1] == 1 and fvad[i] == 0:
            list_X.append(temp)
            temp = []
    return np.array(list_X, dtype=object), sample_rate

def validate_cough(waveform, sr=16000):
    waveform = waveform.astype(np.float32)
    scores, _, _ = yamnet_model(waveform)
    scores_np = scores.numpy()
    cough_index = class_names.index('Cough')
    cough_score = float(np.max(scores_np[:, cough_index]))
    return 1 if cough_score > 0.5 else -1

def extract_features(segments, sr):
    X_all = []
    for seg in segments:
        if len(seg) == 0: continue
        seg = np.array(seg).astype(np.float32)
        stft = np.abs(librosa.stft(seg))
        mfcc = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=seg, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(seg), sr=sr), axis=1)
        centroid = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=275), axis=1)
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr, n_fft=275), axis=1)
        flatness = np.mean(librosa.feature.spectral_flatness(y=seg, n_fft=275), axis=1)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, n_fft=275), axis=1)
        feature_vector = np.concatenate([mfcc, chroma, mel, contrast, tonnetz, centroid, bandwidth, flatness, rolloff])
        X_all.append(feature_vector)
    return np.array(X_all)

def process_and_predict(file_path):
    output = {}
    y, sr_orig = librosa.load(file_path, sr=None)
    output["waveform"] = y[::10].tolist()
    mfccs_visual = librosa.feature.mfcc(y=y, sr=sr_orig, n_mfcc=13)
    output["mfcc"] = mfccs_visual.tolist()
    if not file_path.lower().endswith(('.wav', '.mp3', '.ogg')):
        return {"status": "error", "message": "Format file tidak didukung."}
    segments, sr = segment_audio(file_path)
    if len(segments) == 0:
        return {"status": "error", "message": "Tidak ada segmen audio yang ditemukan."}
    cough_scores = []
    for seg in segments:
        if len(seg) > 0:
            is_cough = validate_cough(np.array(seg))
            cough_scores.append(is_cough)
        else:
            cough_scores.append(-1)
    if not any(score == 1 for score in cough_scores):
        return {"status": "error", "message": "Tidak ada segmen batuk yang valid."}
    valid_segments = [seg for seg, score in zip(segments, cough_scores) if score == 1]
    if not valid_segments:
        return {"status": "error", "message": "Tidak ada segmen batuk setelah validasi."}
    features = extract_features(valid_segments, sr)
    predictions = []
    for feat in features:
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        feat_reshaped = feat_scaled.reshape((1, 1, feat_scaled.shape[1]))
        pred = model.predict(feat_reshaped, verbose=0)[0][0]
        predictions.append(pred)
    tb_count = sum(1 for p in predictions if p > 0.5)
    output["status"] = "success"
    output["prediction"] = "TB" if tb_count > 0 else "NON-TB"
    output["detail"] = {
        "tb_segments": tb_count,
        "non_tb_segments": len(predictions) - tb_count,
        "total_segments": len(predictions)
    }
    return output

if len(sys.argv) > 1:
    audio_file_path = sys.argv[1]
    try:
        if not os.path.isabs(audio_file_path):
            audio_file_path = os.path.abspath(audio_file_path)
        
        result = process_and_predict(audio_file_path)
        print(json.dumps(result))
    except Exception as e:
        print({"status": "error", "message": str(e)})
else:
    print(json.dumps({"status": "error", "message": "No audio file path provided."}))