from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage with an upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


def is_likely_medical_image(file_path: str) -> bool:
    """Check if the image is likely a lung X-ray (grayscale or RGB with grayscale content)."""
    try:
        img = Image.open(file_path)
        print(f"Image mode: {img.mode}")
        if img.mode in ["L", "LA"]:
            return True
        elif img.mode == "RGB":
            img_array = np.array(img)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            if np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5):
                return True
        return False
    except Exception as e:
        print(f"Error checking image mode: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def preprocess_image(file_path: str) -> np.ndarray:
    """Preprocess the image to match training settings (256x256, normalized)."""
    try:
        img = Image.open(file_path)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload, validate, preprocess, and predict lung disease."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if not is_likely_medical_image(file_path):
        print("Image rejected as non-lung")
        return JSONResponse(content={"predictions": [{"class": "Not a lung image", "confidence": "N/A"}]})

    img_array = preprocess_image(file_path)
    if img_array is None:
        raise HTTPException(status_code=500, detail="Failed to preprocess image")

    results = model.predict(file_path, imgsz=256, conf=0.5)

    predictions = []
    for result in results:
        predicted_class_index = result.probs.top1
        confidence = result.probs.top1conf.item()

        print("Full probabilities:", result.probs.data)
        print("Top prediction:", result.names[predicted_class_index], confidence)

        if confidence > 0.5 and predicted_class_index in result.names:
            predicted_class_name = result.names[predicted_class_index]
        else:
            predicted_class_name = "Unknown"

        predictions.append({
            "class": predicted_class_name,
            "confidence": round(confidence, 2) if predicted_class_name != "Unknown" else "N/A"
        })

    return JSONResponse(content={"predictions": predictions})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)