from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import traceback
import io
from dotenv import load_dotenv
from services.gemini import GeminiService
from models.text import TextModel
from models.image import ImageModel
from services.inference import EnsembleInference
from services.explainability import XAIService
import os

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="Multimodal Plant Disease Diagnosis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text", "overhaul", "best_biobert_overhaul.pth")
DATA_PATH = os.path.join(BASE_DIR, "models", "text", "overhaul", "overhaul_metadata.pkl")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PLACEHOLDER")

# Init Services
gemini_service = GeminiService(api_key=GEMINI_API_KEY)
text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=DATA_PATH)
text_model.model.float()

image_model = ImageModel(model_path=IMAGE_MODEL_PATH)
image_model.classifier.float()
image_model.feature_extractor.float()

ensemble = EnsembleInference(alpha=0.65)

from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, FileResponse

class CORSStaticFilesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith('/outputs'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
        return response

app.add_middleware(CORSStaticFilesMiddleware)

@app.get("/")
def read_root():
    return {"status": "online"}

# Serve static files from 'outputs' directory with CORS
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

@app.get("/outputs/{file_path:path}")
async def serve_output_file(file_path: str):
    """Serve files from outputs directory with CORS headers"""
    full_path = os.path.join(OUTPUTS_DIR, file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Determine media type
    media_type = "application/octet-stream"
    if file_path.endswith('.md'):
        media_type = "text/markdown"
    elif file_path.endswith('.txt'):
        media_type = "text/plain"
    elif file_path.endswith('.csv'):
        media_type = "text/csv"
    elif file_path.endswith('.png'):
        media_type = "image/png"
    elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
        media_type = "image/jpeg"
    
    return FileResponse(
        full_path,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/results")
def get_results():
    """
    Returns a structured list of all interpretability assets for the gallery.
    """
    results = {
        "visualizations": [],
        "reports": []
    }
    
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), OUTPUTS_DIR)
            url_path = f"/outputs/{rel_path.replace(os.sep, '/')}"
            
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Group by folder for the gallery categories
                category = os.path.basename(root)
                results["visualizations"].append({
                    "name": file,
                    "url": url_path,
                    "category": category
                })
            elif file.endswith(('.md', '.csv', '.txt')):
                results["reports"].append({
                    "name": file, 
                    "url": url_path,
                    "type": "markdown" if file.endswith('.md') else "text" if file.endswith('.txt') else "csv"
                })
                
    return results

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        # Read file into memory once
        img_bytes = await file.read()
        
        # 1. Gemini Step: Description
        description = gemini_service.generate_description_bytes(img_bytes)
        
        # 2. Image Step: CNN+QNN
        image_probs = image_model.predict_bytes(img_bytes)
        
        # 3. Text Step: Hybrid BioBERT
        text_probs = text_model.predict(description)
        
        # 4. Fusion
        fusion_result = ensemble.fuse(image_probs, text_probs)
        target_idx = fusion_result["predicted_idx"]
        
        # 5. XAI
        heatmap_base64 = XAIService.generate_gradcam_bytes(image_model, img_bytes, target_idx=target_idx)
        text_importance = XAIService.get_feature_importance(description)
        
        label = text_model.label_encoder.classes_[target_idx]
        
        return {
            "disease": label,
            "confidence": f"{fusion_result['confidence']*100:.2f}%",
            "description": description,
            "heatmap": heatmap_base64,
            "importance": text_importance,
            "image_contribution": f"{ensemble.alpha*100:.0f}%",
            "text_contribution": f"{(1-ensemble.alpha)*100:.0f}%",
            "status": "success"
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
