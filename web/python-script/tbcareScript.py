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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Script directory: {script_dir}")
MODEL_PATH = os.path.join('python-script', 'models_tbcare', 'model2.keras')
SCALER_PATH = os.path.join('python-script', 'models_tbcare', 'scaler2.pkl')
YAMNET_MAP_PATH = os.path.join(script_dir, 'yamnet_class_map.csv')

# print(MODEL_PATH)
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
    try:
        X, sample_rate = librosa.load(file_path, sr=22050, mono=True)
        if len(X) == 0:
            return np.array([], dtype=object), 0
    except Exception as e:
        return np.array([], dtype=object), 0

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
        
        # --- PERBAIKAN UTAMA DI SINI ---
        # Ekstrak mean dan std untuk setiap fitur untuk mendapatkan 394 fitur

        stft = np.abs(librosa.stft(seg))
        
        mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=seg, sr=sr)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(seg), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)

        centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=275)
        centroid_mean = np.mean(centroid, axis=1)
        centroid_std = np.std(centroid, axis=1)

        bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr, n_fft=275)
        bandwidth_mean = np.mean(bandwidth, axis=1)
        bandwidth_std = np.std(bandwidth, axis=1)

        flatness = librosa.feature.spectral_flatness(y=seg, n_fft=275)
        flatness_mean = np.mean(flatness, axis=1)
        flatness_std = np.std(flatness, axis=1)
        
        rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, n_fft=275)
        rolloff_mean = np.mean(rolloff, axis=1)
        rolloff_std = np.std(rolloff, axis=1)

        feature_vector = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std,
            tonnetz_mean, tonnetz_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            flatness_mean, flatness_std,
            rolloff_mean, rolloff_std
        ])
        
        X_all.append(feature_vector)
    return np.array(X_all)

def process_and_predict(file_path):
    output = {}
    try:
        y, sr_orig = librosa.load(file_path, sr=None)
        # Downsample untuk visualisasi agar tidak terlalu berat
        output["waveform"] = y[::10].tolist() 
        mfccs_visual = librosa.feature.mfcc(y=y, sr=sr_orig, n_mfcc=13)
        output["mfcc"] = mfccs_visual.tolist()
    except Exception as e:
        return {"status": "error", "message": f"Gagal memuat file audio: {str(e)}"}

    if not file_path.lower().endswith(('.wav', '.mp3', '.ogg')):
        return {"status": "error", "message": "Format file tidak didukung."}

    segments, sr = segment_audio(file_path)
    if sr == 0 or len(segments) == 0:
        return {"status": "error", "message": "Tidak dapat memproses file audio atau tidak ada segmen audio yang ditemukan."}
    
    cough_scores = []
    for seg in segments:
        if len(seg) > 0:
            is_cough = validate_cough(np.array(seg))
            cough_scores.append(is_cough)
        else:
            cough_scores.append(-1)
            
    if not any(score == 1 for score in cough_scores):
        return {"status": "error", "message": "Tidak ada segmen batuk yang valid terdeteksi di dalam rekaman."}

    valid_segments = [seg for seg, score in zip(segments, cough_scores) if score == 1]
    
    if not valid_segments:
        return {"status": "error", "message": "Tidak ada segmen batuk yang valid setelah proses validasi."}
    
    features = extract_features(valid_segments, sr)
    
    if features.shape[0] == 0:
        return {"status": "error", "message": "Ekstraksi fitur tidak menghasilkan data untuk diprediksi."}

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        try:
            if not os.path.isabs(audio_file_path):
                audio_file_path = os.path.abspath(audio_file_path)
            
            result = process_and_predict(audio_file_path)
            print(json.dumps(result))

        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
    else:
        print(json.dumps({"status": "error", "message": "No audio file path provided."}))