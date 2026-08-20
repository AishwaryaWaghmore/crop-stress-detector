# On-Device Crop Stress Detection with Explainable Advisory using Quantized LLMs

This project implements your mini-project pipeline:
- CNN disease detection (EfficientNetB0 transfer learning)
- Grad-CAM explainability heatmaps
- TFLite quantization for edge deployment
- Local advisory generation using quantized GGUF LLMs
- FastAPI backend for inference

## 1. Create environment and install dependencies

Recommended Python version: **3.10 or 3.11** (TensorFlow support).

```bash
py -3.11 -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Kaggle setup

1. Create Kaggle API token from your Kaggle account settings.
2. Place `kaggle.json` at:
   - `C:\\Users\\<your-user>\\.kaggle\\kaggle.json`

## 3. Download datasets (PlantVillage + PlantDoc)

```bash
python download_datasets.py
```

Current defaults:
- PlantVillage: `emmarex/plantdisease`
- PlantDoc: `hasnain4236/plantdoc`

If slugs change on Kaggle, pass custom ones:

```bash
python download_datasets.py --plantvillage <owner/dataset> --plantdoc <owner/dataset>
```

## 4. Prepare combined training dataset

```bash
python prepare_dataset.py --val-size 0.2 --min-images-per-class 20
```

Output:
- `dataset/combined/train/...`
- `dataset/combined/val/...`

## 5. Train model

```bash
python train.py --data-dir dataset/combined --epochs 10 --img-size 224 --batch-size 32
```

Artifacts generated in `artifacts/`:
- `best_model.keras`
- `class_names.txt`
- `training_history.json`

## 6. Predict a sample image

```bash
python predict.py --model artifacts/best_model.keras --image path/to/leaf.jpg
```

## 7. Generate Grad-CAM visualization

```bash
python gradcam.py --model artifacts/best_model.keras --image path/to/leaf.jpg --output artifacts/gradcam/sample_gradcam.jpg
```

## 8. Quantize model for edge deployment (TFLite)

```bash
python quantize.py --model artifacts/best_model.keras --data-dir dataset/combined/train --output artifacts/model_int8.tflite
```

## 9. Run FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Test endpoint:
- `GET /`
- `POST /predict` with multipart form:
  - `file` (image)
  - `language` (optional)
  - `crop_name` (optional)
  - `model_path` (optional)

## 10. Optional: Quantized LLM advisory (local)

Set local GGUF model path before running API:

```bash
set GGUF_MODEL_PATH=C:\\path\\to\\model.gguf
```

If not set, project uses a built-in fallback advisory template.

## Suggested project structure

```
mini project/
  dataset/
    raw/
    combined/
  artifacts/
  api.py
  advisory.py
  download_datasets.py
  prepare_dataset.py
  train.py
  predict.py
  gradcam.py
  quantize.py
  requirements.txt
```
