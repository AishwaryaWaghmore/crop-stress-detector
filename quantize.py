from __future__ import annotations

import argparse
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image

from dataset_utils import chunked, list_images


def require_tensorflow():
	try:
		import importlib

		return importlib.import_module("tensorflow")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"TensorFlow is not installed. Use Python 3.10/3.11 and install requirements.txt."
		) from exc


def representative_dataset_gen(data_dir: Path, img_size: int = 224, sample_limit: int = 300) -> Generator:
	images = list_images(data_dir)[:sample_limit]
	for batch in chunked(images, 1):
		img_path = batch[0]
		img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
		arr = np.array(img, dtype=np.float32)
		arr = np.expand_dims(arr, axis=0)
		yield [arr]


def quantize_keras_to_tflite(model_path: Path, data_dir: Path, output_path: Path, img_size: int = 224) -> None:
	tf = require_tensorflow()
	model = tf.keras.models.load_model(model_path)

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = lambda: representative_dataset_gen(data_dir, img_size=img_size)
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	tflite_model = converter.convert()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_bytes(tflite_model)


def main() -> None:
	parser = argparse.ArgumentParser(description="Quantize Keras model to INT8 TFLite")
	parser.add_argument("--model", default="artifacts/best_model.keras")
	parser.add_argument("--data-dir", default="dataset/combined/train")
	parser.add_argument("--output", default="artifacts/model_int8.tflite")
	parser.add_argument("--img-size", type=int, default=224)
	args = parser.parse_args()

	model_path = Path(args.model)
	data_dir = Path(args.data_dir)
	output_path = Path(args.output)

	quantize_keras_to_tflite(model_path=model_path, data_dir=data_dir, output_path=output_path, img_size=args.img_size)
	original_size_mb = model_path.stat().st_size / (1024 * 1024)
	quantized_size_mb = output_path.stat().st_size / (1024 * 1024)
	reduction = original_size_mb / max(quantized_size_mb, 1e-6)

	print(f"Quantized model saved to: {output_path}")
	print(f"Original size: {original_size_mb:.2f} MB")
	print(f"Quantized size: {quantized_size_mb:.2f} MB")
	print(f"Compression ratio: {reduction:.2f}x")


if __name__ == "__main__":
	main()
