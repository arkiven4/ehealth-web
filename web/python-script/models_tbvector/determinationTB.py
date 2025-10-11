import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import librosa
from torchaudio import transforms as T
import warnings
warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        self.sampling_rate = 16000
        self.desired_length = 1.0
        self.fade_samples_ratio = 6
        self.pad_types = "repeat"
        
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.window = 'hann'
        
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0
        self.cough_padding = 0.5
        self.min_cough_len = 0.2
        self.th_l_multiplier = 0.02
        self.th_h_multiplier = 5
        
        self.input_size = 39
        self.hidden_size = 512
        self.output_size = 2
        self.dropout = 0.1
        
        self.model_path = "./python-script/models_tbvector/LSTM_mfcc_model.pth"

def cut_pad_sample(data, sample_rate, desired_length, pad_types='zero'):
    fade_samples_ratio = 6
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = int(desired_length * sample_rate)
    current_length = data.shape[-1]

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        if pad_types == 'zero':
            total_pad = target_duration - current_length
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            data = torch.nn.functional.pad(data, (pad_left, pad_right), mode='constant', value=0.0)
        elif pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    return data

def load_audio_sample(data, sample_rate, desired_length, fade_samples_ratio=6, pad_types='zero'):
    data = torch.from_numpy(data).unsqueeze(0)
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    data = cut_pad_sample(data, sample_rate, desired_length, pad_types=pad_types)
    return data

def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2):
    cough_mask = np.array([False]*len(x))
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding > min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end = i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding > min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i-padding if (i-padding >= 0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

class MFCCProcessor:
    def __init__(self, config):
        self.config = config
        self.n_mfcc = 13  
        self.n_mels = 40

    def extract_mfcc(self, audio_data, sr):
        
        mfcc_features = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=self.n_mfcc,  # 13
            n_mels=self.n_mels,  # 40
            n_fft=self.config.filter_length,    # 1024
            hop_length=self.config.hop_length,  # 256
            win_length=self.config.win_length,  # 1024
            fmin=self.config.mel_fmin,          # 0.0
            fmax=self.config.mel_fmax           # 8000.0
        )
        
        delta_mfcc = librosa.feature.delta(mfcc_features)
        
        delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
        
        combined_features = np.vstack([mfcc_features, delta_mfcc, delta2_mfcc])
        return combined_features.T 
    
    def preprocess_audio(self, wav_path):
        audio_data, sr = librosa.load(wav_path, sr=self.config.sampling_rate)
        
        audio_max = np.abs(audio_data).max()
        if audio_max > 0:
            audio_data = audio_data / audio_max
        
        cough_segments, _ = segment_cough(
            audio_data, 
            self.config.sampling_rate,
            self.config.cough_padding,      # 0.5
            self.config.min_cough_len,      # 0.2
            self.config.th_l_multiplier,    # 0.02
            self.config.th_h_multiplier     # 5
        )
        
        if len(cough_segments) > 0:
            audio_segment = cough_segments[0]
        else:
            audio_segment = audio_data
        
        if len(audio_segment) < 1000:
            audio_segment = audio_data
        
        audio_tensor = load_audio_sample(
            audio_segment, 
            self.config.sampling_rate, 
            self.config.desired_length, 
            self.config.fade_samples_ratio, 
            self.config.pad_types
        )
        
        mfcc_features = self.extract_mfcc(audio_tensor.squeeze().numpy(), self.config.sampling_rate)
        return torch.tensor(mfcc_features, dtype=torch.float32)

class LSTMAudioClassifierMFCC(nn.Module):
    def __init__(self, input_size=39, hidden_size=512, output_size=2, dropout=0.1):
        
        super(LSTMAudioClassifierMFCC, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # [1, 39, time_frames]
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)  # [1, time_frames, 39]
        x, _ = self.lstm1(x)
        
        # x shape after LSTM: [1, time_frames, 512]
        x = x.transpose(1, 2)  # [1, 512, time_frames]
        x = self.batch_norm2(x)
        x = x.transpose(1, 2)  # [1, time_frames, 512]
        x, _ = self.lstm2(x)

        # Take the last time step
        x = x[:, -1, :]  # [1, 512]
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def predict_tb_from_audio(wav_path, model_path="./python-script/models_tbvector/LSTM_mfcc_model.pth", config_path="./python-script/models_tbvector/mfcc_config.json"):
    
    start_time = time.time()
    
    try:    
        
        config = Config()
        
        mfcc_processor = MFCCProcessor(config)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {
                'model_architecture': {
                    'input_size': 39,
                    'hidden_size': 512, 
                    'output_size': 2,
                    'dropout': 0.1
                }
            }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAudioClassifierMFCC(
            input_size=model_config['model_architecture']['input_size'],    # 39
            hidden_size=model_config['model_architecture']['hidden_size'],  # 512
            output_size=model_config['model_architecture']['output_size'],  # 2
            dropout=model_config['model_architecture']['dropout']           # 0.1
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        mfcc_features = mfcc_processor.preprocess_audio(wav_path)
        mfcc_tensor = mfcc_features.unsqueeze(0).to(device) 
        #
        with torch.no_grad():
            output = model(mfcc_tensor)
            prediction = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(prediction, dim=1).item()
            confidence = prediction[0][predicted_class].item()
        
        processing_time = time.time() - start_time
        return predicted_class, confidence, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"ERROR: {str(e)}", file=sys.stderr)
        return -1, 0.0, processing_time

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("ERROR: Usage: python tb_server.py <audio_filename>", file=sys.stderr)
        sys.exit(1)
    
    audio_filename = sys.argv[1]
    
    base_path = "/usr/src/app/public/uploads/batuk/"  # path audio
    audio_path = os.path.join(base_path, audio_filename)
    
    # Alternative paths for development/testing
    if not os.path.exists(audio_path):
        alternative_paths = [
            "lstm_sken3/",  # Model directory from model.ipynb
            "Audio_files_forced/",
            "Data/",
            "./"
        ]
        
        for alt_path in alternative_paths:
            test_path = os.path.join(alt_path, audio_filename)
            if os.path.exists(test_path):
                audio_path = test_path
                break
    
    if not os.path.exists(audio_path):
        sys.exit(1)
    
    model_path = "./python-script/models_tbvector/LSTM_mfcc_model.pth"
    config_path = "./python-script/models_tbvector/mfcc_config.json"
    
    predicted_class, confidence, processing_time = predict_tb_from_audio(
        audio_path, model_path, config_path
    )
    
    if predicted_class != -1:
        print(predicted_class)  
        print(f"Confidence: {confidence:.4f}")
        print(f"Execution TB Script: --- {processing_time:.4f} seconds ---")
    else:
        sys.exit(1)
