import gc
import io
import os
import base64
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ArecaGuard - Arecanut Disease Detection & Severity Analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. Model & Paths Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "arecanut_classifier.h5")

def find_last_conv_layer(keras_model) -> str:
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name
    return None

grad_model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    last_conv_name = find_last_conv_layer(model)
    if last_conv_name:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
    gc.collect()
    print(f"SUCCESS: Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"ERROR: Failed to load model -> {e}")

CLASS_NAMES = [
    "Bud_Rot",
    "Foot_Rot",
    "Healthy",
    "Mahali_Koleroga",
    "Stem_Bleeding",
    "Yellow_Leaf_Disease"
]

APP_DISEASE_MAP = {
    "Bud_Rot": "Bud_Borer_Rot",
    "Foot_Rot": "Foot_Rot",
    "Healthy": "Healthy",
    "Mahali_Koleroga": "Koleroga",
    "Stem_Bleeding": "Stem_Bleeding",
    "Yellow_Leaf_Disease": "Yellow_Leaf_Disease",
}

IMAGE_SIZE = (224, 224)

TREATMENTS = {
    "Healthy": {
        "status": "Healthy",
        "description": "No visible signs of pathogenic infection or fungal damage detected.",
        "advisory_en": "Palm appears healthy. Continue routine organic mulching and split NPK fertilization.",
        "advisory_kn": "ಅಡಿಕೆ ಮರ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತ ಸಾವಯವ ಗೊಬ್ಬರ ಮತ್ತು ನೀರಾವರಿ ನಿರ್ವಹಣೆ ಮುಂದುವರಿಸಿ.",
        "management": [
            "Maintain regular nutrient schedules with balanced NPK application.",
            "Ensure proper drainage during heavy monsoon downpours.",
            "Conduct routine crown and stem canopy checks."
        ]
    },
    "Mahali_Koleroga": {
        "status": "Severe Fungal Infection",
        "description": "Fruit rot / Koleroga caused by Phytophthora meadii, common during heavy monsoon.",
        "advisory_en": "Immediate 1% Bordeaux mixture spray recommended during a dry spell to arrest fruit rot.",
        "advisory_kn": "ಕೊಳೆರೋಗ ನಿಯಂತ್ರಣಕ್ಕೆ ಮಳೆ ಬಿಡುವು ನೀಡಿದ ತಕ್ಷಣ ೧% ಬೋರ್ಡೋ ದ್ರಾವಣ ಸಿಂಪಡಿಸಿ.",
        "management": [
            "Spray 1% Bordeaux mixture before monsoon onset and give a second spray 40 days later.",
            "Tie polythene covers over arecanut bunches in endemic areas to prevent rain splash infection.",
            "Collect and destroy fallen infected nuts to curb fungal spore spread."
        ]
    },
    "Yellow_Leaf_Disease": {
        "status": "Phytoplasma / Nutritional Stress",
        "description": "Characterized by yellowing of inner whorl leaves, blackening of root tips, and nut drop.",
        "advisory_en": "Apply Magnesium Sulphate (500g/palm), vermicompost, and improve root-zone drainage.",
        "advisory_kn": "ಪ್ರತಿ ಮರಕ್ಕೆ ೫೦೦ ಗ್ರಾಂ ಮೆಗ್ನೀಶಿಯಂ ಸಲ್ಫೇಟ್, ಎರೆಹುಳು ಗೊಬ್ಬರ ನೀಡಿ ಮತ್ತು ಬಸಿಗಾಲುವೆ ಸರಿಪಡಿಸಿ.",
        "management": [
            "Apply additional dosage of balanced organic manure along with zinc sulphate and magnesium sulphate.",
            "Improve soil drainage and avoid waterlogging around palm roots.",
            "Remove severely stunted, unproductive palms and replant with tolerant cultivars."
        ]
    },
    "Stem_Bleeding": {
        "status": "Thielaviopsis paradoxa Fungal Lesion",
        "description": "Exudation of dark reddish-brown liquid through cracks on the trunk.",
        "advisory_en": "Chisel out infected bark tissue and coat the exposed wound with Coal Tar or Bordeaux paste.",
        "advisory_kn": "ಸೋಂಕಿತ ತೊಗಟೆಯನ್ನು ಕೆತ್ತಿ ತೆಗೆದು ಕಲ್ಲಿದ್ದಲು ಡಾಂಬರು ಅಥವಾ ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಚ್ಚಿ.",
        "management": [
            "Chisel out infected bark tissue cleanly until fresh tissue is exposed.",
            "Swab the wounded trunk area with Coal Tar or 5% Calixin/Hexaconazole paste.",
            "Apply Trichoderma-enriched neem cake (5 kg per palm) in the root basin."
        ]
    },
    "Bud_Rot": {
        "status": "Phytophthora palmivora Infection",
        "description": "Infection of growing bud tissues resulting in rotting and heart leaf shedding.",
        "advisory_en": "Excise decaying spindle tissue immediately and apply 10% Bordeaux paste into the crown.",
        "advisory_kn": "ಕೊಳೆತ ಸುಳಿ ಭಾಗವನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ ಸುಳಿಯ ಒಳಗೆ ೧೦% ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಾಕಿ.",
        "management": [
            "Scoop out infected rotten spindle and bud tissue in the early stage.",
            "Treat crown cavity with 10% Bordeaux paste or apply Mancozeb paste.",
            "Cover treated crown with a protective cap to shield from rainwater."
        ]
    },
    "Foot_Rot": {
        "status": "Anabe Roga / Ganoderma Lucidum",
        "description": "Bracket fungal infection attacking root bases, causing leaf drooping and trunk brittleness.",
        "advisory_en": "Dig isolation trenches around the palm and drench root zone with 0.2% Hexaconazole.",
        "advisory_kn": "ಮರದ ಸುತ್ತ ಬಸಿಗಾಲುವೆ ತೋಡಿ ಬೇರಿಗೆ ಹೆಕ್ಸಾಕೊನಜೋಲ್ (೦.೨%) ದ್ರಾವಣ ಸುರಿಯಿರಿ.",
        "management": [
            "Dig isolation trenches (30 cm wide and 60 cm deep) around diseased palms.",
            "Drench root zones with Hexaconazole (0.2%) or Captan solution.",
            "Uproot and burn dead palms along with root stumps to prevent spore dispersal."
        ]
    }
}

class Base64ImageRequest(BaseModel):
    image_base64: str

# ---------------------------------------------------------
# 2. Memory-Safe Image Preprocessing & Grad-CAM Helpers
# ---------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.draft("RGB", (512, 512))
        resized_image = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0), resized_image


def generate_gradcam_overlay(img_array: np.ndarray, resized_img: Image.Image, pred_index: int) -> tuple[str | None, float]:
    try:
        if grad_model is None:
            return None, 0.0

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        infected_ratio = float(np.mean(heatmap_np > 0.5) * 100)

        orig_np = np.asarray(resized_img, dtype=np.float32)
        heatmap_resized = cv2.resize(heatmap_np, IMAGE_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)

        superimposed_img = np.uint8(orig_np * 0.6 + color_heatmap * 0.4)

        overlay_pil = Image.fromarray(superimposed_img)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="JPEG", quality=80)
        encoded_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_str}", round(infected_ratio, 2)
    except Exception as cam_err:
        print(f"Grad-CAM generation skipped: {cam_err}")
        return None, 0.0


def run_prediction_pipeline(image_bytes: bytes) -> dict:
    try:
        processed_img, resized_img = preprocess_image(image_bytes)

        raw_predictions = model(processed_img, training=False).numpy()
        predicted_idx = int(np.argmax(raw_predictions[0]))
        confidence = float(raw_predictions[0][predicted_idx])

        raw_class_name = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class_{predicted_idx}"
        app_disease_key = APP_DISEASE_MAP.get(raw_class_name, raw_class_name)

        confidence_pct = round(confidence * 100, 2)
        heatmap_b64, infected_pct = generate_gradcam_overlay(processed_img, resized_img, predicted_idx)

        if raw_class_name == "Healthy":
            severity = "Healthy"
            severity_pct = 0.0
        else:
            severity_pct = infected_pct if infected_pct > 0 else round(confidence_pct * 0.65, 2)
            if severity_pct >= 50 or confidence_pct >= 85:
                severity = "Severe"
            elif severity_pct >= 20 or confidence_pct >= 60:
                severity = "Moderate"
            else:
                severity = "Mild"

        all_probabilities = {
            (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"): round(float(prob) * 100, 2)
            for i, prob in enumerate(raw_predictions[0])
        }

        remediation_info = TREATMENTS.get(raw_class_name, {
            "status": "Identified",
            "description": "Refer to an agricultural officer for tailored diagnosis.",
            "advisory_en": "Consult a local horticulture officer for field inspection.",
            "advisory_kn": "ಸ್ಥಳೀಯ ತೋಟಗಾರಿಕಾ ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "management": ["Isolate affected palm and avoid excess surface irrigation."]
        })

        return {
            "disease": app_disease_key,
            "prediction": raw_class_name,
            "confidence": confidence_pct,
            "severity_level": severity,
            "severity": severity,
            "severity_percentage": severity_pct,
            "advisory_en": remediation_info["advisory_en"],
            "advisory_kn": remediation_info["advisory_kn"],
            "all_probabilities": all_probabilities,
            "heatmap_base64": heatmap_b64,
            "treatment": remediation_info
        }
    finally:
        gc.collect()

# ---------------------------------------------------------
# 3. API Routes
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "classes_supported": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Classifier model is not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    try:
        contents = await file.read()
        return run_prediction_pipeline(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict_base64")
async def predict_base64_image(payload: Base64ImageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Classifier model is not loaded.")

    try:
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
        return run_prediction_pipeline(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)