import os
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import soundfile as sf
import io
import json
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

app = FastAPI(title="StethoAI API",
             description="API for medical sound analysis and diagnosis",
             version="1.0.0")
@app.get("/")
async def root():
    return {"message": "مرحبا! API تعمل الآن بنجاح"}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    class_labels = np.load('class_labels.npy', allow_pickle=True).item()
    diagnoses = np.load('diagnoses.npy', allow_pickle=True).item()
    print("Model and data loaded successfully")
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None
    scaler = None
    class_labels = {}
    diagnoses = {}

def preprocess_audio(audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != 22050:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        return mfccs_scaled
    except Exception as e:
        print(f"Error in audio preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error in audio preprocessing: {str(e)}")

def predict_sound(audio_data: np.ndarray) -> Dict[str, Any]:
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not trained yet")
    
    try:
        # Preprocess audio
        features = preprocess_audio(audio_data)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict_proba(features_scaled)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Get diagnosis
        diagnosis = diagnoses.get(predicted_class, "Unknown")
        sound_type = class_labels.get(predicted_class, "Unknown")
        
        return {
            "sound_type": sound_type,
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "all_predictions": {
                class_labels.get(i, f"Class_{i}"): float(pred) 
                for i, pred in enumerate(prediction)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to StethoAI API. The API is running!"}

@app.post("/analyze-sound", 
         response_model=Dict[str, Any],
         summary="Analyze medical sound",
         description="Upload an audio file to analyze and get diagnosis")
async def analyze_sound(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # طباعة معلومات الملف للمساعدة في التصحيح
        print("filename:", file.filename)
        print("content_type:", file.content_type)
        
        # تحقق من الامتداد
        allowed_ext = ('.wav', '.mp3', '.ogg', '.m4a')
        filename_lower = file.filename.lower()
        if not filename_lower.endswith(allowed_ext):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file (wav, mp3, ogg, m4a)")

        # حاول قراءة الملف حتى لو كان content_type غير صحيح إذا كان الامتداد صحيح
        try:
            audio_bytes = await file.read()
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading audio file: {str(e)}")
        
        # Check if audio data is valid
        if audio_data.size == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
            
        # Predict sound type
        result = predict_sound(audio_data)
        return result
            
    except HTTPException as he:
        raise he
    except Exception as e:
        if "python-multipart" in str(e):
            raise HTTPException(status_code=500, 
                              detail="Server error: python-multipart package is required. Please install it using: pip install python-multipart")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health", 
        response_model=Dict[str, Any],
        summary="Check API health",
        description="Check if the API is running and model is loaded")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "available_diagnoses": list(diagnoses.values()),
        "available_sound_types": list(class_labels.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
