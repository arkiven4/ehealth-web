import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
# Force librosa to use scipy/kaiser_fast instead of resampy
os.environ['LIBROSA_PREFERRED_RESAMPLER'] = 'kaiser_fast'
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

BASE_PATH = os.path.join('python-script', 'models_tbcare')
MODEL_YAMNET_EMBEDDING = os.path.join(BASE_PATH, 'yamnet_saved_model')
YAMNET_CLASSIFIER_PATH = os.path.join(BASE_PATH, 'model-1s.keras')
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
        # Load audio dengan res_type='kaiser_fast' untuk menghindari error Numba
        X, sample_rate = librosa.load(file_path, sr=22050, mono=True, res_type='kaiser_fast')
        if len(X) == 0:
            print(json.dumps({"status": "debug", "message": "Audio file is empty"}))
            return np.array([], dtype=object), 0
        
        print(json.dumps({"status": "debug", "message": f"Audio loaded: {len(X)} samples, {len(X)/sample_rate:.2f}s duration"}))
    except Exception as e:
        print(json.dumps({"status": "debug", "message": f"Error loading audio: {str(e)}"}))
        # Try scipy resampler as fallback
        try:
            import soundfile as sf
            X, sample_rate_orig = sf.read(file_path)
            
            # Convert stereo to mono if needed
            if len(X.shape) > 1 and X.shape[1] > 1:
                X = np.mean(X, axis=1)
                print(json.dumps({"status": "debug", "message": "Converted stereo to mono"}))
            
            if sample_rate_orig != 22050:
                from scipy import signal
                num_samples = int(len(X) * 22050 / sample_rate_orig)
                X = signal.resample(X, num_samples)
            sample_rate = 22050
            print(json.dumps({"status": "debug", "message": f"Audio loaded with scipy: {len(X)} samples"}))
        except Exception as e2:
            print(json.dumps({"status": "debug", "message": f"All audio loading methods failed: {str(e2)}"}))
            return np.array([], dtype=object), 0

    try:
        fvad = getVad(X, sample_rate)
        print(json.dumps({"status": "debug", "message": f"VAD result: {len(fvad)} frames, sum={sum(fvad)}"}))
    except Exception as e:
        print(json.dumps({"status": "debug", "message": f"VAD failed: {str(e)}, using full audio"}))
        # Fallback: jika VAD gagal, gunakan seluruh audio
        return np.array([X], dtype=object), sample_rate
    
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
    
    # Jika VAD tidak menemukan segmen, gunakan seluruh audio
    if len(list_X) == 0:
        print(json.dumps({"status": "debug", "message": "No VAD segments found, using full audio as single segment"}))
        return np.array([X], dtype=object), sample_rate
    
    print(json.dumps({"status": "debug", "message": f"Segmentation result: {len(list_X)} segments"}))
    return np.array(list_X, dtype=object), sample_rate


# === YAMNET VALIDATION (Step 1: Validate Cough) ===
def extract_yamnet_embedding(audio_segment, target_sr=16000, max_len=1.0):
    try:
        # Resample to 16kHz for YAMNet
        if len(audio_segment) == 0:
            return None
        
        # Ensure audio is 1D (mono)
        audio_segment = np.array(audio_segment, dtype=np.float32)
        if len(audio_segment.shape) > 1:
            audio_segment = np.mean(audio_segment, axis=1)
        
        # Use kaiser_fast to avoid Numba error, atau gunakan scipy
        try:
            y_resampled = librosa.resample(audio_segment, orig_sr=SR, target_sr=target_sr, res_type='kaiser_fast')
        except:
            # Fallback to scipy resampling
            from scipy import signal
            num_samples = int(len(audio_segment) * target_sr / SR)
            y_resampled = signal.resample(audio_segment, num_samples)
        
        # Ensure still 1D after resampling
        if len(y_resampled.shape) > 1:
            y_resampled = y_resampled.flatten()
        
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
    emb = extract_yamnet_embedding(audio_segment)
    if emb is None:
        print(json.dumps({"status": "debug", "message": "YAMNet embedding extraction failed"}))
        return False, 0.0
    
    # Predict with YAMNet classifier
    emb = emb.reshape(1, -1)  # (1, 1024)
    pred = yamnet_classifier.predict(emb, verbose=0)[0][0]
    cough_score = float(pred)
    is_cough = cough_score > threshold
    
    print(json.dumps({"status": "debug", "message": f"YAMNet cough score: {cough_score:.4f}, is_cough: {is_cough}"}))
    
    return is_cough, cough_score


# === FEATURE EXTRACTION (Step 2: Extract MFCC 1s) ===
def extract_mfcc_1s(audio_segment, sr=SR):
    try:
        y = np.array(audio_segment, dtype=np.float32)
        
        # Pad or truncate to 1 second
        target_len = int(sr * SEQ_DURATION)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        
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
        y, sr_orig = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
        output["waveform"] = y[::10].tolist()  # downsample untuk efisiensi
        mfccs_visual = librosa.feature.mfcc(y=y, sr=sr_orig, n_mfcc=13)
        output["mfcc"] = mfccs_visual.tolist()
    except Exception as e:
        # Try alternative loading method
        try:
            import soundfile as sf
            y, sr_orig = sf.read(file_path)
            
            # Convert stereo to mono if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.mean(y, axis=1)
            
            output["waveform"] = y[::10].tolist()
            mfccs_visual = librosa.feature.mfcc(y=y, sr=sr_orig, n_mfcc=13)
            output["mfcc"] = mfccs_visual.tolist()
        except Exception as e2:
            return {"status": "error", "message": f"Gagal memuat file audio: {str(e2)}"}

    # Step 2: Segmentasi
    segments, sr = segment_audio(file_path)
    if sr == 0 or len(segments) == 0:
        return {"status": "error", "message": "Tidak ada segmen batuk yang valid terdeteksi."}

    print(json.dumps({"status": "info", "message": f"Found {len(segments)} segments"}))

    # Step 3: Validasi batuk dengan YAMNet + Ekstraksi MFCC + Prediksi TB
    segment_details = []
    valid_segments = 0
    tb_probs_all = []
    
    # OPSI: Set ke False untuk skip validasi YAMNet (untuk debugging)
    USE_YAMNET_VALIDATION = True
    YAMNET_THRESHOLD = 0.3  # Turunkan threshold dari 0.5 ke 0.3 agar lebih permisif
    
    for idx, seg in enumerate(segments):
        print(json.dumps({"status": "debug", "message": f"Processing segment {idx+1}/{len(segments)}, length: {len(seg)} samples"}))
        
        # 3.1: Validate cough with YAMNet
        if USE_YAMNET_VALIDATION:
            is_cough, cough_score = validate_cough_with_yamnet(seg, threshold=YAMNET_THRESHOLD)
            
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
        else:
            # Skip YAMNet validation (assume all segments are cough)
            is_cough = True
            cough_score = 1.0
            print(json.dumps({"status": "debug", "message": f"YAMNet validation skipped, assuming cough"}))
        
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
        
        print(json.dumps({"status": "debug", "message": f"Segment {idx+1}: TB prob={tb_prob:.4f}, prediction={prediction_label}"}))

    # Step 4: Calculate overall statistics with Soft Voting
    print(json.dumps({"status": "info", "message": f"Valid cough segments: {valid_segments}/{len(segments)}"}))
    
    if len(tb_probs_all) == 0:
        # Hitung berapa segmen yang di-reject oleh YAMNet
        rejected_by_yamnet = sum(1 for s in segment_details if not s.get("is_cough", False))
        return {
            "status": "error", 
            "message": f"Tidak ada segmen batuk yang valid ditemukan setelah validasi YAMNet. Total segmen: {len(segments)}, Rejected: {rejected_by_yamnet}. Coba turunkan threshold YAMNet atau check kualitas audio.",
            "debug_info": {
                "total_segments": len(segments),
                "rejected_by_yamnet": rejected_by_yamnet,
                "segment_details": segment_details
            }
        }
    
    avg_tb_prob = np.mean(tb_probs_all)
    SOFT_VOTING_THRESHOLD = 0.5  

    tb_count = sum(1 for p in tb_probs_all if p > SOFT_VOTING_THRESHOLD)
    non_tb_count = len(tb_probs_all) - tb_count
    final_decision = "TB" if avg_tb_prob >= SOFT_VOTING_THRESHOLD else "NON-TB"

    # === OUTPUT ===
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
        "segment_details": segment_details,
        "waveform": output.get("waveform", []),
        "mfcc": output.get("mfcc", [])
    })

    return output


# === ENTRY POINT ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        print(json.dumps({"status": "debug", "message": f"Input path: {audio_path}"}))
        print(json.dumps({"status": "debug", "message": f"Current working directory: {os.getcwd()}"}))

        # Handle relative path
        if not os.path.isabs(audio_path):
            # Coba beberapa kemungkinan base path
            possible_paths = [
                os.path.abspath(audio_path),  # Relatif terhadap current dir
                os.path.join(os.getcwd(), audio_path),  # Eksplisit dari current dir
                os.path.join(os.path.dirname(os.getcwd()), audio_path.lstrip('../')),  # Parent dir
            ]
            
            # Cari path yang valid
            for path in possible_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            else:
                # Jika tidak ada yang valid, gunakan yang pertama
                audio_path = possible_paths[0]
        
        print(json.dumps({"status": "debug", "message": f"Resolved path: {audio_path}"}))
        
        if not os.path.exists(audio_path):
            print(json.dumps({
                "status": "error", 
                "message": f"File tidak ditemukan: {audio_path}",
                "debug_info": {
                    "input_path": sys.argv[1],
                    "resolved_path": audio_path,
                    "cwd": os.getcwd(),
                    "parent_dir": os.path.dirname(os.getcwd())
                }
            }))
            sys.exit(1)
        
        try:
            result = process_and_predict(audio_path)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
    else:
        print(json.dumps({"status": "error", "message": "No audio file path provided."}))
