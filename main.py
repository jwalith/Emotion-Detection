# backend/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
# import tensorflow as tf
import io
import joblib
import os
import logging
from keras.models import load_model
from datetime import datetime
from fastapi.logger import logger
import json

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # Frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to model and preprocessing objects
MODEL_PATH = os.path.join('model', 'speech_emotion_recognition_model.h5')
SCALER_PATH = os.path.join('model', 'scaler.save')
ENCODER_PATH = os.path.join('model', 'encoder.save')

# Check if all required files exist
for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
    if not os.path.exists(path):
        logger.error(f"Required file not found: {path}")
        raise FileNotFoundError(f"Required file not found: {path}")

# Load the scaler
try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    raise e

# Load the encoder
try:
    encoder = joblib.load(ENCODER_PATH)
    logger.info("Encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading encoder: {e}")
    raise e

# Load the trained model
try:
    print("Checking Here")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Retrieve emotion labels from encoder
try:
    emotion_labels = encoder.categories_[0].tolist()
    logger.info(f"Emotion labels: {emotion_labels}")
except Exception as e:
    logger.error(f"Error retrieving emotion labels from encoder: {e}")
    raise e

def extract_features(data, sample_rate):
    """
    Extracts audio features from raw audio data.

    Parameters:
    - data (np.ndarray): Audio time series.
    - sample_rate (int): Sampling rate of the audio.

    Returns:
    - np.ndarray: Extracted feature vector.
    """
    try:
        # Zero Crossing Rate (ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

        # Chroma Short-Time Fourier Transform (Chroma_STFT)
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # Mel-Frequency Cepstral Coefficients (MFCC)
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)

        # Root Mean Square (RMS) Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

        # Concatenate all features into a single vector
        feature_vector = np.hstack((zcr, chroma_stft, mfcc, rms, mel))

        return feature_vector
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the emotion from an uploaded audio file.

    Parameters:
    - file (UploadFile): The uploaded audio file.

    Returns:
    - dict: Predicted emotion and confidence score.
    """
    # Validate file type
    if file.content_type not in ["audio/wav","audio/mp3", "audio/mpeg", "audio/flac", "audio/x-m4a"]:
        logger.warning(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Supported types: .wav, .mp3, .flac, .m4a")

    try:
        # Read and validate file size
        contents = await file.read()
        MAX_FILE_SIZE = 15 * 1024 * 1024  # 10 MB
        if len(contents) > MAX_FILE_SIZE:
            logger.error("Uploaded file exceeds size limit.")
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

        # Load the entire audio file
        file_like = io.BytesIO(contents)
        try:
            audio_data, sample_rate = librosa.load(file_like, sr=None)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise HTTPException(status_code=400, detail="Failed to load audio file. Ensure the file is valid.")

        # Calculate and log audio duration
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        logger.info(f"Audio duration: {duration:.2f} seconds")

        # Validate audio data
        if len(audio_data) == 0:
            logger.error("Audio file is empty or invalid.")
            raise HTTPException(status_code=400, detail="Audio file is empty or invalid.")

        # Extract features
        features = extract_features(audio_data, sample_rate)

        # Scale features
        features_scaled = scaler.transform([features])

        # Reshape for Conv1D: (samples, features, 1)
        features_scaled = np.expand_dims(features_scaled, axis=2)
        input_shape = model.input_shape[1:]
        if features_scaled.shape[1:] != input_shape:
            logger.error(f"Incompatible feature shape. Expected {input_shape}, got {features_scaled.shape[1:]}")
            raise HTTPException(status_code=500, detail="Incompatible feature shape.")

        # Predict emotion
        prediction = model.predict(features_scaled)

        # Decode prediction
        predicted_class = encoder.inverse_transform(prediction)[0][0]
        confidence = float(np.max(prediction))

        logger.info(f"Prediction: {predicted_class} with confidence {confidence:.2f}")

        return {"emotion": predicted_class, "confidence": confidence}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the file.")


@app.post("/journal")
async def save_journal_entry(
    file: UploadFile = File(...), 
    emotion: str = Form("unknown"), 
    note: str = Form("")
):
    """
    Saves a journal entry with audio, emotion, and a note.

    Parameters:
    - file (UploadFile): The uploaded audio file.
    - emotion (str): The predicted emotion.
    - note (str): A user-provided note.

    Returns:
    - dict: Success message.
    """
    try:
        # Path to store journal entries
        JOURNAL_FILE = os.path.join("journal_data", "journal.json")

        # Ensure the journal file exists
        if not os.path.exists(JOURNAL_FILE):
            os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
            with open(JOURNAL_FILE, "w") as f:
                json.dump([], f)

        # Generate a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = file.filename
        unique_filename = f"{timestamp}_{original_filename}"
        # Save the audio file
        audio_path = os.path.join("audio_files", unique_filename)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # Save the journal entry
        with open(JOURNAL_FILE, "r+") as f:
            journal = json.load(f)
            journal.append({"audio_file": unique_filename, "emotion": emotion, "note": note, "date": datetime.strptime(timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")})
            f.seek(0)
            json.dump(journal, f, indent=4)

        logger.info(f"Journal entry saved: {audio_path}, {emotion}")
        return {"message": "Journal entry saved successfully."}

    except Exception as e:
        logger.error(f"Error saving journal entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to save journal entry.")
    
@app.get("/journal")
async def get_journal_entries():
    """
    Retrieves all journal entries.

    Returns:
    - List[dict]: A list of journal entries.
    """
    try:
        JOURNAL_FILE = os.path.join("journal_data", "journal.json")

        # Ensure the journal file exists
        if not os.path.exists(JOURNAL_FILE):
            return []

        with open(JOURNAL_FILE, "r") as f:
            journal = json.load(f)
        return journal
    except Exception as e:
        logger.error(f"Error fetching journal entries: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch journal entries.")
    

from fastapi.responses import FileResponse

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serves an audio file by its filename.

    Parameters:
    - filename (str): The name of the audio file.

    Returns:
    - FileResponse: The requested audio file.
    """
    audio_path = os.path.join("audio_files", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


from pydantic import BaseModel

class UpdateNoteRequest(BaseModel):
    note: str

@app.put("/journal/{index}")
async def update_journal_entry(index: int, request: UpdateNoteRequest):
    """
    Updates a journal entry's note.

    Parameters:
    - index (int): Index of the journal entry.
    - request (UpdateNoteRequest): Contains the updated note.

    Returns:
    - dict: Success message.
    """
    try:
        print(f"Trying here for index: {index}")
        JOURNAL_FILE = os.path.join("journal_data", "journal.json")

        if not os.path.exists(JOURNAL_FILE):
            raise HTTPException(status_code=404, detail="Journal file not found.")

        with open(JOURNAL_FILE, "r+") as f:
            journal = json.load(f)

            if index < 0 or index >= len(journal):
                raise HTTPException(status_code=404, detail="Invalid journal entry index.")

            # Update the note
            journal[index]["note"] = request.note
            f.seek(0)
            json.dump(journal, f, indent=4)
            f.truncate()

        logger.info(f"Updated note for journal entry {index}.")
        return {"message": "Journal entry updated successfully."}

    except Exception as e:
        logger.error(f"Error updating journal entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to update journal entry.")
