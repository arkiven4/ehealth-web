"""
TB Care Prediction Script - Enhanced Pipeline
==============================================

Pipeline:
1. Segmentasi audio menggunakan VAD (Voice Activity Detection)
2. Validasi batuk dengan YAMNet fine-tuned classifier (model-1s.keras)
   - Ekstraksi YAMNet embedding (1024 features)
   - Prediksi apakah segmen adalah batuk (pos=1, neg=0)
3. Ekstraksi fitur MFCC (1 detik, 40 coefficients)
   - SR: 22050 Hz
   - N_FFT: 100 ms window (2205 samples)
   - HOP_LENGTH: 5 ms (110 samples)
   - Duration: 1 second (~199 frames)
4. Prediksi TB dengan LSTM MFCC model (model1_mfcc.keras)
5. Soft Voting: mean probability threshold
   - Threshold 0.5: lebih sensitif untuk deteksi TB (recommended)
   - Threshold 0.7: lebih spesifik untuk non-TB testing

Models:
- YAMNet base: yamnet_saved_model (embedding extraction)
- YAMNet classifier: model-1s.keras (cough validation)
- LSTM classifier: model1_mfcc.keras (TB classification)
"""

import os
import sys

# Set environment variables BEFORE importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (avoid CUDA errors if GPU not available)

# CRITICAL: Disable numba caching completely to avoid Docker read-only filesystem issues
# This is necessary because numba tries to cache compiled functions in site-packages
# which is read-only in Docker containers
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
# Set to 1 to disable caching, but keep JIT compilation
os.environ['NUMBA_DISABLE_JIT'] = '1'  # DISABLE JIT completely as last resort

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from scipy.signal import lfilter
from copy import deepcopy
import speechproc

# ===========================================================================
# PERUBAHAN UTAMA (Updated):
# ===========================================================================
# 1. Validasi Batuk YAMNet: 
#    - Ekstraksi YAMNet embedding (1024 features)
#    - Klasifikasi dengan model-1s.keras (pos=1, neg=0)
#
# 2. Ekstraksi Fitur MFCC (1 detik):
#    - SR: 22050 Hz
#    - N_FFT: 2205 samples (100 ms window)
#    - HOP_LENGTH: 110 samples (5 ms hop)
#    - N_MFCC: 40 coefficients
#    - Duration: 1 second (~200 frames)
#    - TIDAK pakai normalisasi mean (untuk konsistensi dengan training)
#
# 3. Prediksi TB:
#    - Model LSTM: model1_mfcc_.keras
#    - Input shape: (samples, timesteps, features) = (1, 200, 40)
#
# 4. Soft Voting (Evaluasi Per Audio):
#    - Mean probability dari semua segmen batuk yang valid
#    - Threshold 0.5: Lebih sensitif untuk deteksi TB (recommended)
#    - Threshold 0.7: Lebih spesifik untuk testing non-TB
# ===========================================================================

BASE_PATH = os.path.join('python-script', 'models_tbcare')
MODEL_YAMNET_EMBEDDING = os.path.join(BASE_PATH, 'yamnet_saved_model')

# YAMNet classifier (model-1s.keras adalah folder, bukan file)
YAMNET_CLASSIFIER_PATH = os.path.join(BASE_PATH, 'model-1s.keras')

# LSTM MFCC model untuk prediksi TB (juga folder)
MODEL_MFCC_LSTM = os.path.join(BASE_PATH, 'model1_mfcc_.keras')

# === MFCC Parameters (1 second window) ===
SR = 22050
N_FFT = int(0.1 * SR)  # 100 ms window = 2205 samples
HOP_LENGTH = int(0.005 * SR)  # 5 ms hop = 110 samples
N_MFCC = 40
SEQ_DURATION = 1.0  # 1 second
TARGET_FRAMES = int((SR * SEQ_DURATION) / HOP_LENGTH)  # ~200 frames


# === LOAD MODEL UTILITY ===
def load_keras_model(model_path):
    """
    Load Keras model with multiple fallback methods:
    1. Try direct load (for .keras files or SavedModel format)
    2. Try with compile=False for compatibility issues
    3. Try loading from folder structure (config.json + model.weights.h5) with KerasLayer support
    """
    # Method 1: Direct load
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e1:
        pass
    
    # Method 2: Try with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e2:
        pass
    
    # Method 3: Load from folder structure with custom objects (KerasLayer)
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, 'config.json')
        weights_path = os.path.join(model_path, 'model.weights.h5')
        
        if os.path.exists(config_path) and os.path.exists(weights_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                # Load model dengan custom objects untuk KerasLayer
                model = tf.keras.models.model_from_json(
                    json.dumps(config), 
                    custom_objects={'KerasLayer': hub.KerasLayer}
                )
                model.load_weights(weights_path)
                return model
            except Exception as e3:
                print(json.dumps({"status": "debug", "message": f"Folder load failed: {str(e3)[:100]}"}))
    
    raise FileNotFoundError(f"Could not load model from {model_path} using any method")


# === LOAD MODELS ===
try:
    print(json.dumps({"status": "info", "message": "Loading models..."}))
    
    # Load YAMNet for embedding extraction
    yamnet_model = tf.saved_model.load(MODEL_YAMNET_EMBEDDING)
    print(json.dumps({"status": "info", "message": "Loaded YAMNet embedding model"}))

    # Load YAMNet classifier (model-1s.keras)
    if not os.path.exists(YAMNET_CLASSIFIER_PATH):
        raise FileNotFoundError(f"YAMNet classifier not found at: {YAMNET_CLASSIFIER_PATH}")
    
    yamnet_classifier = load_keras_model(YAMNET_CLASSIFIER_PATH)
    print(json.dumps({"status": "info", "message": f"Loaded YAMNet classifier (model-1s.keras)"}))

    # Load LSTM MFCC model (model1_mfcc_.keras)
    if not os.path.exists(MODEL_MFCC_LSTM):
        raise FileNotFoundError(f"LSTM MFCC model not found at: {MODEL_MFCC_LSTM}")
    
    model_lstm = load_keras_model(MODEL_MFCC_LSTM)
    print(json.dumps({"status": "info", "message": "Loaded LSTM MFCC model (model1_mfcc_.keras)"}))

except Exception as e:
    print(json.dumps({"status": "error", "message": f"Error loading models: {str(e)}"}))
    sys.exit(1)


# === SEGMENTATION ===
def getVad(data, fs):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres, vadThres = 0.5, 0.4
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
    except Exception:
        return np.array([], dtype=object), 0

    fvad = getVad(X, sample_rate)
    list_X, temp = [], []
    for i in range(1, len(fvad)):
        if fvad[i - 1] == 1:
            len_sector = int(len(X) / len(fvad))
            start = (i - 1) * len_sector
            for j in range(start, start + len_sector):
                if j < len(X):
                    temp.append(X[j])
        if fvad[i - 1] == 1 and fvad[i] == 0:
            list_X.append(temp)
            temp = []
    return np.array(list_X, dtype=object), sample_rate


# === YAMNET VALIDATION (Step 1: Validate Cough) ===
def extract_yamnet_embedding(audio_segment, target_sr=16000, max_len=1.0):
    """
    Extract YAMNet embedding from audio segment
    Args:
        audio_segment: numpy array of audio data
        target_sr: target sample rate for YAMNet (16000 Hz)
        max_len: maximum length in seconds (1.0)
    Returns:
        YAMNet embedding (1024 features) or None if error
    """
    try:
        # Resample to 16kHz for YAMNet
        if len(audio_segment) == 0:
            return None
        
        y_resampled = librosa.resample(audio_segment, orig_sr=SR, target_sr=target_sr)
        target_len = int(target_sr * max_len)
        
        # Pad or truncate to 1 second
        if len(y_resampled) > target_len:
            y_resampled = y_resampled[:target_len]
        else:
            y_resampled = np.pad(y_resampled, (0, target_len - len(y_resampled)))
        
        # Extract YAMNet embedding
        waveform = tf.convert_to_tensor(y_resampled, dtype=tf.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        emb_mean = tf.reduce_mean(embeddings, axis=0)  # shape (1024,)
        return emb_mean.numpy()
    except Exception as e:
        print(json.dumps({"status": "warning", "message": f"Error extracting YAMNet embedding: {str(e)}"}))
        return None


def validate_cough_with_yamnet(audio_segment, threshold=0.5):
    """
    Validate if segment is a cough using YAMNet classifier (model-1s.keras)
    Args:
        audio_segment: numpy array of audio data
        threshold: classification threshold (default 0.5)
    Returns:
        is_cough (bool): True if cough detected
        cough_score (float): probability of being a cough
    """
    emb = extract_yamnet_embedding(audio_segment)
    if emb is None:
        return False, 0.0
    
    # Predict with YAMNet classifier
    emb = emb.reshape(1, -1)  # (1, 1024)
    pred = yamnet_classifier.predict(emb, verbose=0)[0][0]
    cough_score = float(pred)
    is_cough = cough_score > threshold
    
    return is_cough, cough_score


# === FEATURE EXTRACTION (Step 2: Extract MFCC 1s) ===
def extract_mfcc_1s(audio_segment, sr=SR):
    """
    Extract MFCC features from 1 second audio segment
    Args:
        audio_segment: numpy array of audio data
        sr: sample rate (22050 Hz)
    Returns:
        MFCC features (40, ~200 frames) or None if error
    """
    try:
        y = np.array(audio_segment, dtype=np.float32)
        
        # Pad or truncate to 1 second
        target_len = int(sr * SEQ_DURATION)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        
        # Extract MFCC with parameters matching training
        # SR: 22050 Hz
        # N_FFT: 2205 (100 ms window)
        # HOP_LENGTH: 110 (5 ms hop)
        # N_MFCC: 40
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Pad or truncate to target frames (~200)
        cur_len = mfcc.shape[1]
        if cur_len < TARGET_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_FRAMES - cur_len)), mode='constant')
        else:
            mfcc = mfcc[:, :TARGET_FRAMES]
        
        return mfcc
    except Exception as e:
        print(json.dumps({"status": "warning", "message": f"Error extracting MFCC: {str(e)}"}))
        return None


# === MAIN PREDICTION ===
def process_and_predict(file_path):
    output = {}

    # Step 1: Load audio for visualization
    try:
        y, sr_orig = librosa.load(file_path, sr=None)
        output["waveform"] = y[::10].tolist()  # downsample untuk efisiensi
        mfccs_visual = librosa.feature.mfcc(y=y, sr=sr_orig, n_mfcc=13)
        output["mfcc"] = mfccs_visual.tolist()
    except Exception as e:
        return {"status": "error", "message": f"Gagal memuat file audio: {str(e)}"}

    # Step 2: Segmentasi
    segments, sr = segment_audio(file_path)
    if sr == 0 or len(segments) == 0:
        return {"status": "error", "message": "Tidak ada segmen batuk yang valid terdeteksi."}

    print(json.dumps({"status": "info", "message": f"Found {len(segments)} segments"}))

    # Step 3: Validasi batuk dengan YAMNet + Ekstraksi MFCC + Prediksi TB
    segment_details = []
    valid_segments = 0
    tb_probs_all = []
    
    for idx, seg in enumerate(segments):
        # 3.1: Validate cough with YAMNet
        is_cough, cough_score = validate_cough_with_yamnet(seg, threshold=0.5)
        
        if not is_cough:
            # Skip non-cough segments
            segment_details.append({
                "segment_id": idx + 1,
                "is_cough": False,
                "cough_score": round(float(cough_score), 4),
                "tb_probability": 0.0,
                "prediction": "NON-COUGH"
            })
            continue
        
        # 3.2: Extract MFCC features (1 second)
        mfcc_feat = extract_mfcc_1s(seg, sr=SR)
        if mfcc_feat is None:
            segment_details.append({
                "segment_id": idx + 1,
                "is_cough": True,
                "cough_score": round(float(cough_score), 4),
                "tb_probability": 0.0,
                "prediction": "ERROR"
            })
            continue
        
        # 3.3: Reshape for LSTM model (samples, timesteps, features)
        # mfcc_feat shape: (40, TARGET_FRAMES) -> transpose to (TARGET_FRAMES, 40)
        mfcc_reshaped = np.transpose(mfcc_feat, (1, 0))  # (TARGET_FRAMES, 40)
        mfcc_reshaped = mfcc_reshaped.reshape(1, mfcc_reshaped.shape[0], mfcc_reshaped.shape[1])  # (1, TARGET_FRAMES, 40)
        
        # 3.4: Predict TB probability with LSTM
        tb_prob = model_lstm.predict(mfcc_reshaped, verbose=0)[0][0]
        tb_prob = float(tb_prob)
        tb_probs_all.append(tb_prob)
        
        valid_segments += 1
        prediction_label = "TB" if tb_prob > 0.5 else "NON-TB"
        
        segment_details.append({
            "segment_id": idx + 1,
            "is_cough": True,
            "cough_score": round(float(cough_score), 4),
            "tb_probability": round(tb_prob, 4),
            "prediction": prediction_label
        })

    # Step 4: Calculate overall statistics with Soft Voting
    if len(tb_probs_all) == 0:
        return {
            "status": "error", 
            "message": "Tidak ada segmen batuk yang valid ditemukan setelah validasi YAMNet."
        }
    
    # === SOFT VOTING: mean probability ===
    # Menggunakan rata-rata probabilitas dari semua segmen batuk yang valid
    # 
    # Threshold 0.5: lebih sensitif untuk deteksi TB
    #   - Lebih baik untuk skrining TB (mendeteksi lebih banyak kasus positif)
    #   - Sensitivity tinggi, Specificity lebih rendah
    #
    # Threshold 0.7: lebih spesifik untuk non-TB
    #   - Lebih baik untuk testing pada orang sehat/non-TB
    #   - Specificity tinggi, Sensitivity lebih rendah
    
    avg_tb_prob = np.mean(tb_probs_all)
    SOFT_VOTING_THRESHOLD = 0.5  # Default: lebih sensitif untuk deteksi TB
    
    # Count segments based on threshold
    tb_count = sum(1 for p in tb_probs_all if p > SOFT_VOTING_THRESHOLD)
    non_tb_count = len(tb_probs_all) - tb_count
    
    # Final decision based on soft voting (mean probability)
    final_decision = "TB" if avg_tb_prob >= SOFT_VOTING_THRESHOLD else "NON-TB"

    # === OUTPUT ===
    # Structure disesuaikan dengan format yang diharapkan oleh controller
    output.update({
        "status": "success",
        "prediction": final_decision,  # "TB" or "NON-TB"
        "final_decision": final_decision,
        "detail": {
            "total_segments": len(segments),  # Total segmen dari VAD
            "valid_cough_segments": valid_segments,  # Segmen yang tervalidasi sebagai batuk
            "tb_segments": tb_count,  # Jumlah segmen dengan prob > threshold
            "non_tb_segments": non_tb_count,  # Jumlah segmen dengan prob <= threshold
            "average_tb_probability": round(float(avg_tb_prob), 4),  # Soft voting score
            "soft_voting_threshold": SOFT_VOTING_THRESHOLD,
            "confidence_percentage": round(float(avg_tb_prob * 100), 2)
        },
        "segment_details": segment_details,  # Detail per segmen (untuk debugging)
        "waveform": output.get("waveform", []),  # Untuk visualisasi
        "mfcc": output.get("mfcc", [])  # Untuk visualisasi
    })

    return output


# === ENTRY POINT ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        # Path resolution: handle both absolute and relative paths
        if not os.path.isabs(audio_path):
            # Resolve from current working directory (where the command is run)
            # Script dijalankan dari web/, jadi public/ harus diresolve sebagai ../public/
            audio_path = os.path.abspath(audio_path)
        
        # Validasi file exists
        if not os.path.exists(audio_path):
            print(json.dumps({"status": "error", "message": f"File tidak ditemukan: {audio_path}"}))
            sys.exit(1)
        
        try:
            result = process_and_predict(audio_path)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
    else:
        print(json.dumps({"status": "error", "message": "No audio file path provided."}))
