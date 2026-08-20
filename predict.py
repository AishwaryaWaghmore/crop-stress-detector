from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def require_tensorflow():
	try:
		import importlib

		return importlib.import_module("tensorflow")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"TensorFlow is not installed. Use Python 3.10/3.11 and install requirements.txt."
		) from exc


def load_class_names(model_path: Path) -> List[str]:
	class_file = model_path.parent / "class_names.txt"
	if not class_file.exists():
		raise FileNotFoundError(f"Missing class names file: {class_file}")
	return [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image_path: Path, img_size: int = 224) -> np.ndarray:
	image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
	arr = np.array(image, dtype=np.float32)
	return np.expand_dims(arr, axis=0)


def predict_image(model_path: str, image_path: str, img_size: int = 224) -> Dict:
	tf = require_tensorflow()
	model_path_obj = Path(model_path)
	model = tf.keras.models.load_model(model_path_obj)
	class_names = load_class_names(model_path_obj)

	x = preprocess_image(Path(image_path), img_size=img_size)
	probs = model.predict(x, verbose=0)[0]
	class_idx = int(np.argmax(probs))

	return {
		"class_index": class_idx,
		"label": class_names[class_idx],
		"confidence": float(probs[class_idx]),
		"all_probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict crop disease for one image")
	parser.add_argument("--model", default="artifacts/best_model.keras")
	parser.add_argument("--image", required=True)
	parser.add_argument("--img-size", type=int, default=224)
	args = parser.parse_args()

	result = predict_image(args.model, args.image, img_size=args.img_size)
	print("Prediction:")
	print(f"  Label: {result['label']}")
	print(f"  Confidence: {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
	main()
