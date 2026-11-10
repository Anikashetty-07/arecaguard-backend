import base64
import io
from typing import Dict, Any, Tuple
import cv2
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI(title="Arecanut Disease & Severity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/arecanut_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = [
    'Bud_Borer_Rot', 
    'Healthy', 
    'Koleroga', 
    'Stem_Bleeding', 
    'Stem_Cracking', 
    'Yellow_Leaf_Disease'
]

MANAGEMENT_GUIDELINES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Healthy": {
        "en": {
            "name": "Healthy Plant",
            "pathogen": "None (Plant is healthy)",
            "action": "Maintain balanced NPK fertilization and clean basin weeding."
        },
        "kn": {
            "name": "ಆರೋಗ್ಯಕರ ಸಸ್ಯ (Healthy)",
            "pathogen": "ಯಾವುದೇ ರೋಗಾಣು ಇಲ್ಲ",
            "action": "ಸರಿಯಾದ ಪ್ರಮಾಣದಲ್ಲಿ ಗೊಬ್ಬರ ಹಾಗೂ ನೀರಾವರಿ ನಿರ್ವಹಣೆ ಮುಂದುವರಿಸಿ."
        }
    },
    "Koleroga": {
        "en": {
            "name": "Koleroga / Fruit Rot",
            "pathogen": "Phytophthora meadii (Fungus)",
            "action": "Spray Bordeaux mixture (1%) before monsoon. Fasten polythene bunch covers."
        },
        "kn": {
            "name": "ಕೊಳೆರೋಗ / ಮಹಾಲಿ (Koleroga)",
            "pathogen": "ಫೈಟೊಫ್ತೋರಾ ಮೀಡಿಯೈ (ಶಿಲೀಂಧ್ರ)",
            "action": "ಮಳೆಗಾಲ ಆರಂಭಕ್ಕೂ ಮುನ್ನ ಶೇ 1% ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಸಿಂಪಡಿಸಿ ಅಥವಾ ಹಿಂಗಾರಿಗೆ ಪಾಲಿಥೀನ್ ಹೊದಿಕೆ ಕಟ್ಟಿ."
        }
    },
    "Yellow_Leaf_Disease": {
        "en": {
            "name": "Yellow Leaf Disease",
            "pathogen": "Phytoplasma Organism",
            "action": "Provide proper drainage. Apply farmyard manure, zinc, and boron micronutrients."
        },
        "kn": {
            "name": "ಹಳದಿ ಎಲೆ ರೋಗ (Yellow Leaf Disease)",
            "pathogen": "ಫೈಟೋಪ್ಲಾಸ್ಮಾ ಸೂಕ್ಷ್ಮಾಣುಜೀವಿ",
            "action": "ತೋಟದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಬಸಿದು ಹೋಗಲು ಕಾಲುವೆ ಮಾಡಿ. ಬೇವಿನ ಹಿಂಡಿ ಮತ್ತು ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶಗಳನ್ನು ನೀಡಿ."
        }
    },
    "Bud_Borer_Rot": {
        "en": {
            "name": "Bud Borer / Crown Rot",
            "pathogen": "Crown Infestation / Decay",
            "action": "Remove affected tissue. Apply copper oxychloride paste to the cut surface."
        },
        "kn": {
            "name": "ಸುಳಿ ಕೊಳೆ ರೋಗ / ಹುಳು ಬಾಧೆ (Bud Rot)",
            "pathogen": "ಸುಳಿ ಕೊಳೆಯುವಿಕೆ ಮತ್ತು ಕೀಟ ಬಾಧೆ",
            "action": "ಕೊಳೆತ ಸುಳಿಯನ್ನು ಕತ್ತರಿಸಿ ತೆಗೆದು, ಆ ಜಾಗಕ್ಕೆ ತಾಮ್ರದ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಪೇಸ್ಟ್ ಹಚ್ಚಿ."
        }
    },
    "Stem_Bleeding": {
        "en": {
            "name": "Stem Bleeding",
            "pathogen": "Thielaviopsis paradoxa (Fungus)",
            "action": "Chisel diseased bark patches. Apply coal tar or Bordeaux paste; drench root basin."
        },
        "kn": {
            "name": "ಕಾಂಡ ಸೋರುವಿಕೆ (Stem Bleeding)",
            "pathogen": "ಥೀಲಾವಿಯೊಪ್ಸಿಸ್ ಪ್ಯಾರಾಡಾಕ್ಸಾ (ಶಿಲೀಂಧ್ರ)",
            "action": "ರೋಗಪೀಡಿತ ಕಾಂಡದ ಭಾಗವನ್ನು ಕೆತ್ತಿ ತೆಗೆದು ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಅಥವಾ ಡಾಂಬರು ಹಚ್ಚಿ."
        }
    },
    "Stem_Cracking": {
        "en": {
            "name": "Stem Cracking",
            "pathogen": "Nutritional Stress / Moisture Fluctuation",
            "action": "Maintain uniform irrigation to prevent moisture stress; apply borax."
        },
        "kn": {
            "name": "ಕಾಂಡ ಬಿರುಕು ಬಿಡುವುದು (Stem Cracking)",
            "pathogen": "ನೀರಿನ ಕೊರತೆ ಮತ್ತು ಬೋರಾನ್ ಪೋಷಕಾಂಶದ ವ್ಯತ್ಯಯ",
            "action": "ತೋಟಕ್ಕೆ ನಿಯಮಿತವಾಗಿ ನೀರು ಹಾಯಿಸಿ, ಶಿಫಾರಸು ಮಾಡಿದ ಪ್ರಮಾಣದಲ್ಲಿ ಬೊರಾಕ್ಸ್ ನೀಡಿ."
        }
    }
}

class ImageData(BaseModel):
    image_base64: str

def compute_severity(image_rgb: np.ndarray, predicted_class: str) -> Tuple[float, str]:
    if predicted_class == "Healthy":
        return 0.0, "Healthy"

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    tissue_mask = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([100, 255, 255]))
    lesion_mask = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([35, 255, 220]))
    lesion_in_tissue = cv2.bitwise_and(lesion_mask, lesion_mask, mask=tissue_mask)

    total_tissue = int(cv2.countNonZero(tissue_mask))
    total_lesions = int(cv2.countNonZero(lesion_in_tissue))

    if total_tissue == 0:
        return 0.0, "Healthy"

    pct = round((total_lesions / total_tissue) * 100.0, 2)
    if pct < 2.0:
        tier = "Healthy"
    elif pct < 15.0:
        tier = "Mild"
    elif pct < 35.0:
        tier = "Moderate"
    else:
        tier = "Severe"

    return pct, tier

def run_model_inference(image_pil: Image.Image) -> Dict[str, Any]:
    image_np = np.array(image_pil.convert("RGB"))

    resized = cv2.resize(image_np, (224, 224)) / 255.0
    input_tensor = np.expand_dims(resized, axis=0)

    # In modern Keras, calling model(tensor) directly is faster and cleaner
    preds = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100.0
    predicted_class = CLASSES[idx]

    severity_pct, severity_tier = compute_severity(image_np, predicted_class)
    advisory = MANAGEMENT_GUIDELINES.get(predicted_class, {})

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "severity_percentage": severity_pct,
        "severity_level": severity_tier,
        "advisory_en": advisory.get("en", {}),
        "advisory_kn": advisory.get("kn", {})
    }

@app.post("/predict_base64")
async def predict_crop_base64(data: ImageData):
    img_bytes = base64.b64decode(data.image_base64)
    image = Image.open(io.BytesIO(img_bytes))
    return run_model_inference(image)

@app.post("/predict")
async def predict_crop(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return run_model_inference(image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)