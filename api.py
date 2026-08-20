from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from advisory import LocalAdvisoryEngine
from gradcam import generate_gradcam
from predict import predict_image

app = FastAPI(title="Crop Stress Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

advisory_engine = LocalAdvisoryEngine()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "message": "Crop Stress Detection API is running."}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    language: str = Form("English"),
    crop_name: str = Form("Unknown Crop"),
    model_path: str = Form("artifacts/best_model.keras"),
) -> dict:
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = Path(tmp.name)

    pred = predict_image(model_path=model_path, image_path=str(temp_path))

    gradcam_dir = Path("artifacts/gradcam")
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    gradcam_path = gradcam_dir / f"{temp_path.stem}_gradcam.jpg"

    generate_gradcam(
        model_path=model_path,
        image_path=str(temp_path),
        output_path=str(gradcam_path),
        class_index=pred["class_index"],
    )

    advisory_text = advisory_engine.generate(
        crop=crop_name,
        disease=pred["label"],
        confidence=pred["confidence"],
        language=language,
    )

    return {
        "prediction": pred,
        "gradcam_path": str(gradcam_path),
        "advisory": advisory_text,
    }
