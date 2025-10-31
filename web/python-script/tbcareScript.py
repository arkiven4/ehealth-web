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
   - N_FFT: 100 ms window
   - HOP_LENGTH: 5 ms
   - Duration: 1 second (~199 frames)
4. Prediksi TB dengan LSTM MFCC model (model1_mfcc.keras)
5. Soft Voting: mean probability threshold (default 0.5)
   - Threshold 0.5: lebih sensitif untuk deteksi TB
   - Threshold 0.7: lebih spesifik untuk non-TB testing

Models:
- YAMNet base: yamnet_saved_model (embedding extraction)
- YAMNet classifier: model-1s.keras (cough validation)
- LSTM classifier: model1_mfcc.keras (TB classification)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import librosa
from scipy.signal import lfilter
from copy import deepcopy
import speechproc

# Suppress warnings and errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (avoid CUDA errors if GPU not available)

import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.path.join('python-script', 'models_tbcare')
MODEL_YAMNET_EMBEDDING = os.path.join(BASE_PATH, 'yamnet_saved_model')

# YAMNet classifier - try multiple possible locations
YAMNET_CLASSIFIER_PATHS = [
    os.path.join(BASE_PATH, 'model-1s.keras'),
    os.path.join(BASE_PATH, 'yamnet_classifier', 'model-1s.keras'),
    os.path.join(BASE_PATH, 'yamnet_classifier', 'model.keras'),
]

MODEL_MFCC_LSTM = os.path.join(BASE_PATH, 'model1_mfcc.keras')  # LSTM MFCC model (1 second window)

# === MFCC Parameters (1 second window) ===
SR = 22050
N_FFT = int(0.1 * SR)  # 100 ms window
HOP_LENGTH = int(0.005 * SR)  # 5 ms hop
N_MFCC = 40
SEQ_DURATION = 1.0  # 1 second
TARGET_FRAMES = int((SR * SEQ_DURATION) / HOP_LENGTH)  # ~199 frames


# === LOAD MODEL UTILITY ===
def load_keras_model(model_path):
    """
    Load Keras model with multiple fallback methods:
    1. Try direct load (for .keras files or SavedModel format)
    2. Try loading from folder structure (config.json + model.weights.h5)
    3. Try with compile=False for compatibility issues
    """
    # Method 1: Direct load
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e1:
        print(json.dumps({"status": "debug", "message": f"Direct load failed: {str(e1)[:100]}"}))
    
    # Method 2: Try with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e2:
        print(json.dumps({"status": "debug", "message": f"Load with compile=False failed: {str(e2)[:100]}"}))
    
    # Method 3: Load from folder structure
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, 'config.json')
        weights_path = os.path.join(model_path, 'model.weights.h5')
        
        if os.path.exists(config_path) and os.path.exists(weights_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                model = tf.keras.models.model_from_json(json.dumps(config))
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

    # Load YAMNet classifier (try multiple possible locations)
    yamnet_classifier = None
    for path in YAMNET_CLASSIFIER_PATHS:
        if os.path.exists(path):
            try:
                yamnet_classifier = load_keras_model(path)
                print(json.dumps({"status": "info", "message": f"Loaded YAMNet classifier from: {path}"}))
                break
            except Exception as e:
                print(json.dumps({"status": "debug", "message": f"Failed to load from {path}: {str(e)[:100]}"}))
                continue
    
    if yamnet_classifier is None:
        raise FileNotFoundError("YAMNet classifier not found. Tried: " + ", ".join(YAMNET_CLASSIFIER_PATHS))

    # Load LSTM MFCC model (model1_mfcc.keras)
    model_lstm = load_keras_model(MODEL_MFCC_LSTM)
    print(json.dumps({"status": "info", "message": "Loaded LSTM MFCC model (model1_mfcc.keras)"}))

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
    """Extract YAMNet embedding from audio segment"""
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
        
        waveform = tf.convert_to_tensor(y_resampled, dtype=tf.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        emb_mean = tf.reduce_mean(embeddings, axis=0)  # shape (1024,)
        return emb_mean.numpy()
    except Exception as e:
        print(json.dumps({"status": "warning", "message": f"Error extracting YAMNet embedding: {str(e)}"}))
        return None


def validate_cough_with_yamnet(audio_segment, threshold=0.5):
    """Validate if segment is a cough using YAMNet classifier"""
    emb = extract_yamnet_embedding(audio_segment)
    if emb is None:
        return False, 0.0
    
    emb = emb.reshape(1, -1)
    pred = yamnet_classifier.predict(emb, verbose=0)[0][0]
    is_cough = float(pred) > threshold
    return is_cough, float(pred)


# === FEATURE EXTRACTION (Step 2: Extract MFCC 1s) ===
def extract_mfcc_1s(audio_segment, sr=SR):
    """Extract MFCC features from 1 second audio segment"""
    try:
        y = np.array(audio_segment, dtype=np.float32)
        
        # Pad or truncate to 1 second
        target_len = int(sr * SEQ_DURATION)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        
        # Extract MFCC with new parameters
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Pad or truncate to target frames
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

    # Step 4: Calculate overall statistics
    if len(tb_probs_all) == 0:
        return {
            "status": "error", 
            "message": "Tidak ada segmen batuk yang valid ditemukan setelah validasi YAMNet."
        }
    
    # Soft voting: average probability across all valid cough segments
    # Threshold 0.5 lebih baik untuk mendeteksi TB (sensitif terhadap positif)
    # Threshold 0.7 lebih baik untuk testing pada orang non-TB (lebih spesifik)
    avg_tb_prob = np.mean(tb_probs_all)
    SOFT_VOTING_THRESHOLD = 0.5
    
    tb_count = sum(1 for p in tb_probs_all if p > SOFT_VOTING_THRESHOLD)
    non_tb_count = len(tb_probs_all) - tb_count
    
    # Final decision based on soft voting (mean probability)
    final_decision = "TB" if avg_tb_prob >= SOFT_VOTING_THRESHOLD else "NON-TB"

    # === OUTPUT ===
    output.update({
        "status": "success",
        "prediction": final_decision,
        "final_decision": final_decision,
        "total_segments": len(segments),
        "valid_cough_segments": valid_segments,
        "tb_segments": tb_count,
        "non_tb_segments": non_tb_count,
        "average_tb_probability": round(float(avg_tb_prob), 4),
        "overall_confidence": round(float(avg_tb_prob * 100), 2),  # as percentage
        "segment_details": segment_details
    })

    return output


# === ENTRY POINT ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.isabs(audio_path):
            audio_path = os.path.abspath(audio_path)
        try:
            result = process_and_predict(audio_path)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
    else:
        print(json.dumps({"status": "error", "message": "No audio file path provided."}))
