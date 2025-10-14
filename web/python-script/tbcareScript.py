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
import traceback
from copy import deepcopy
import speechproc
from scipy.signal import lfilter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models_tbcare')

MODEL1_DIR = os.path.join(models_dir, 'model1_mfcc.keras')
YAMNET_DIR = os.path.join(models_dir, 'yamnet_saved_model')
YAMNET_PB = os.path.join(models_dir, 'yamnet_saved_model.pb')
CLASSIFIER_DIR = os.path.join(models_dir, 'yamnet_classifier', 'model.keras')

# helper
def exists_readable(p):
    return os.path.exists(p) and os.access(p, os.R_OK)

try:
    # === LOAD SEGMENT-LEVEL MFCC MODEL (model1) ===
    if not exists_readable(MODEL1_DIR):
        raise FileNotFoundError(f"Model dir not found: {MODEL1_DIR}")
    model = None
    config_path = os.path.join(MODEL1_DIR, 'config.json')
    weights_path = os.path.join(MODEL1_DIR, 'model.weights.h5')
    if os.path.isdir(MODEL1_DIR) and exists_readable(config_path) and exists_readable(weights_path):
        with open(config_path, 'r') as f:
            cfg_txt = f.read()
        model = tf.keras.models.model_from_json(cfg_txt)
        model.load_weights(weights_path)
    else:
        model = tf.keras.models.load_model(MODEL1_DIR, compile=False)

    # === LOAD YAMNET EMBEDDING MODEL ===
    yamnet_embed_model = None
    if os.path.isdir(YAMNET_DIR) and exists_readable(os.path.join(YAMNET_DIR, "saved_model.pb")):
        try:
            yamnet_embed_model = tf.saved_model.load(YAMNET_DIR)
            print(json.dumps({"status": "info", "message": "Loaded YAMNet model from directory."}))
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Failed to load YAMNet directory: {e}"}))
    elif exists_readable(YAMNET_PB):
        try:
            yamnet_embed_model = hub.load(YAMNET_PB)
            print(json.dumps({"status": "info", "message": "Loaded YAMNet model from .pb file via TF Hub."}))
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Failed to load YAMNet .pb file: {e}"}))
    else:
        print(json.dumps({"status": "warning", "message": "No YAMNet model found in either directory or .pb file."}))

    # === LOAD CLASSIFIER MODEL ===
    classifier_model = None
    classifier_config = os.path.join(CLASSIFIER_DIR, 'config.json')
    classifier_weights = os.path.join(CLASSIFIER_DIR, 'model.weights.h5')
    if os.path.isdir(CLASSIFIER_DIR) and exists_readable(classifier_config) and exists_readable(classifier_weights):
        with open(classifier_config, 'r') as f:
            cfg_txt = f.read()
        classifier_model = tf.keras.models.model_from_json(cfg_txt)
        classifier_model.load_weights(classifier_weights)
    else:
        try:
            if exists_readable(CLASSIFIER_DIR):
                classifier_model = tf.keras.models.load_model(CLASSIFIER_DIR, compile=False)
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Failed to load classifier: {e}"}))
        if classifier_model is None:
            print(json.dumps({"status": "warning", "message": "yamnet classifier not found; skipping classifier load."}))

except Exception as e:
    err = {"status": "error", "message": str(e), "trace": traceback.format_exc()}
    print(json.dumps(err))
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

def validate_cough(waveform, sr=22050):
    """
    Validate if a segment is cough.
    Preferred: yamnet_embed + classifier_model.
    Fallback: use segment-level MFCC model (model) to classify segment.
    Return: 1 if cough, -1 otherwise.
    """
    try:
        # Preferred pipeline: yamnet embedding + classifier (if available)
        if yamnet_embed_model is not None and classifier_model is not None:
            emb = extract_yamnet_embedding(waveform, target_sr=sr)
            if emb is None:
                return -1
            emb_reshaped = emb.reshape(1, -1)
            pred = classifier_model.predict(emb_reshaped, verbose=0)
            score = float(pred[0][0]) if pred is not None else 0.0
            return 1 if score > 0.5 else -1

        # Fallback: use MFCC + segment-level model (model)
        if 'model' in globals() and model is not None:
            feats = extract_features([waveform], sr)  # returns (1, n_mfcc) if OK
            if feats is None or feats.shape[0] == 0:
                return -1
            # try a few reshape heuristics depending on model input shape
            try:
                # common Keras conv1D expects (batch, timesteps, features)
                x = feats.reshape((1, feats.shape[1], 1))
                pred = model.predict(x, verbose=0)
            except Exception:
                try:
                    # some models expect (batch, features)
                    x = feats
                    pred = model.predict(x, verbose=0)
                except Exception:
                    # last attempt: (1, features)
                    x = feats.reshape((1, feats.shape[1]))
                    pred = model.predict(x, verbose=0)
            score = float(pred[0][0])
            return 1 if score > 0.5 else -1

        # If nothing available, conservative: mark as not cough
        return -1

    except Exception as e:
        # don't raise — mark invalid segment
        return -1

def extract_yamnet_embedding(waveform, target_sr=22050, model_sr=16000, max_len=4.0):
    if yamnet_embed_model is None:
        return None
    """
    Menghitung mean embedding YAMNet (1024D) dari waveform.
    """
    if len(waveform) == 0:
        return None
        
    # YAMNet distandarkan pada 16000 Hz, jadi perlu downsampling jika sr asli bukan 16000
    if target_sr != model_sr:
        waveform = librosa.resample(waveform, orig_sr=target_sr, target_sr=model_sr)
        target_sr = model_sr
        
    # Pad atau potong ke panjang 4 detik (sesuai kode referensi Anda)
    if len(waveform) > target_sr * max_len:
        y = waveform[:int(target_sr * max_len)]
    else:
        y = np.pad(waveform, (0, int(target_sr * max_len) - len(waveform)))
        
    waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # yamnet_embed_model adalah model YAMNet base yang sudah dimuat
    scores, embeddings, spectrogram = yamnet_embed_model(waveform_tf)
    
    emb_mean = tf.reduce_mean(embeddings, axis=0)
    return emb_mean.numpy()

def extract_features(segments, sr):
    X_all = []
    
    # === Konfigurasi Parameter yang Relevan ===
    SR = 22050 
    N_FFT = 512
    HOP_LENGTH = 256
    N_MFCC = 40
    
    for seg in segments:
        if len(seg) == 0: continue
        seg = np.array(seg).astype(np.float32)
        
        # --- Ekstraksi Mean MFCC ---
        # 1. Hitung MFCC (40 koefisien)
        mfccs = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH) 
        
        # 2. Hitung Mean dari MFCCs sepanjang waktu (axis=1)
        feature_vector = np.mean(mfccs, axis=1)
        
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
            is_cough = validate_cough(np.array(seg), sr=sr) 
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
        feat_reshaped = feat.reshape((feat.shape[0], 1, feat.shape[1]))
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