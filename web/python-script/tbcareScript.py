import os
import sys
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from scipy.signal import lfilter
from copy import deepcopy
import speechproc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BASE_PATH = os.path.join('python-script', 'models_tbcare')
MODEL_MFCC_PATH = os.path.join(BASE_PATH, 'model1_mfcc.keras')
MODEL_YAMNET_EMBEDDING = os.path.join(BASE_PATH, 'yamnet_saved_model')
MODEL_YAMNET_CLASSIFIER = os.path.join(BASE_PATH, 'yamnet_classifier', 'model.keras')


# === LOAD MODEL UTILITY ===
def load_model_from_folder(model_folder):
    config_path = os.path.join(model_folder, 'config.json')
    weights_path = os.path.join(model_folder, 'model.weights.h5')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing model files in {model_folder}")
    with open(config_path) as f:
        config = json.load(f)
    model = tf.keras.models.model_from_json(json.dumps(config), custom_objects={'KerasLayer': hub.KerasLayer})
    model.load_weights(weights_path)
    return model


# === LOAD MODELS ===
try:
    print(json.dumps({"status": "info", "message": "Loading models..."}))
    model_mfcc = load_model_from_folder(MODEL_MFCC_PATH)
    print(json.dumps({"status": "info", "message": "Loaded MFCC model"}))

    yamnet_model = tf.saved_model.load(MODEL_YAMNET_EMBEDDING)
    print(json.dumps({"status": "info", "message": "Loaded YAMNet embedding model"}))

    classifier_model = load_model_from_folder(MODEL_YAMNET_CLASSIFIER)
    print(json.dumps({"status": "info", "message": "Loaded YAMNet classifier model"}))

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


# === FEATURE EXTRACTION ===
def extract_mfcc_features(segments, sr):
    features = []
    for seg in segments:
        if len(seg) < sr * 0.1:  # skip segmen terlalu pendek
            continue
        mfcc = librosa.feature.mfcc(y=np.array(seg, dtype=np.float32), sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)
    return np.array(features)


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

    # Step 3: Ekstraksi fitur
    features = extract_mfcc_features(segments, sr)
    if features.shape[0] == 0:
        return {"status": "error", "message": "Tidak ada fitur yang berhasil diekstraksi."}

    # Step 4: Prediksi MFCC model
    predictions = []
    for feat in features:
        feat_reshaped = feat.reshape((1, 1, 40))
        pred = model_mfcc.predict(feat_reshaped, verbose=0)[0][0]
        predictions.append(pred)

    tb_count = sum(1 for p in predictions if p > 0.5)
    non_tb_count = len(predictions) - tb_count
    total_count = len(predictions)

    # === OUTPUT ===
    output.update({
        "status": "success",
        "prediction": "TB" if tb_count > 0 else "NON-TB",
        "final_decision": "TB" if tb_count > 0 else "NON-TB",
        "tb_segments": tb_count,
        "non_tb_segments": non_tb_count,
        "total_segments": total_count
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
