import librosa
import numpy as np
import io

# def extract_features(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc_scaled = np.mean(mfcc.T, axis=0)
#     return mfcc_scaled


def extract_features_from_io(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def get_features_inference(data, sample_rate):
    """
    Extract features from audio data without data augmentation.

    Parameters:
    - data (np.ndarray): Audio time series.
    - sample_rate (int): Sampling rate of the audio.

    Returns:
    - np.ndarray: Extracted feature vector.
    """
    return extract_features(data)
