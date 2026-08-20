from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.cm as cm
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


def preprocess_image(image_path: Path, img_size: int = 224):
	image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
	arr = np.array(image, dtype=np.float32)
	return arr, np.expand_dims(arr, axis=0)


def get_connected_feature_tensor(tf, model, preferred_layer_name: Optional[str] = None):
	if preferred_layer_name:
		layer = model.get_layer(preferred_layer_name)
		return layer.output

	try:
		return model.get_layer("efficientnetb0").output
	except Exception:
		pass

	for layer in reversed(model.layers):
		output = getattr(layer, "output", None)
		if output is None:
			continue
		shape = getattr(output, "shape", None)
		if shape is None or len(shape) != 4:
			continue
		try:
			tf.keras.models.Model([model.inputs], [output, model.output])
			return output
		except Exception:
			continue

	raise ValueError("No connected 4D feature layer found for Grad-CAM")


def make_gradcam_heatmap(tf, img_array: np.ndarray, model, feature_tensor, pred_index: Optional[int] = None):
	grad_model = tf.keras.models.Model(
		model.inputs,
		[feature_tensor, model.output],
	)

	with tf.GradientTape() as tape:
		conv_outputs, predictions = grad_model(img_array)
		if pred_index is None:
			pred_index = tf.argmax(predictions[0])
		class_channel = predictions[:, pred_index]

	grads = tape.gradient(class_channel, conv_outputs)
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
	conv_outputs = conv_outputs[0]
	heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
	heatmap = tf.squeeze(heatmap)

	heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
	return heatmap.numpy()


def make_input_gradient_heatmap(tf, img_array: np.ndarray, model, pred_index: Optional[int] = None):
	inputs = tf.convert_to_tensor(img_array)
	with tf.GradientTape() as tape:
		tape.watch(inputs)
		predictions = model(inputs)
		if pred_index is None:
			pred_index = tf.argmax(predictions[0])
		class_channel = predictions[:, pred_index]

	grads = tape.gradient(class_channel, inputs)
	grads = tf.math.abs(grads[0])
	heatmap = tf.reduce_mean(grads, axis=-1)
	heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
	return heatmap.numpy()


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
	heatmap_uint8 = np.uint8(255 * heatmap)
	colormap = cm.get_cmap("jet")
	jet_colors = colormap(np.arange(256))[:, :3]
	jet_heatmap = jet_colors[heatmap_uint8]
	jet_heatmap = cv2.resize(jet_heatmap, (original.shape[1], original.shape[0]))
	jet_heatmap = np.uint8(255 * jet_heatmap)

	overlay = cv2.addWeighted(original.astype(np.uint8), 1 - alpha, jet_heatmap, alpha, 0)
	return overlay


def generate_gradcam(
	model_path: str,
	image_path: str,
	output_path: str,
	class_index: Optional[int] = None,
	conv_layer_name: Optional[str] = None,
	img_size: int = 224,
) -> str:
	tf = require_tensorflow()
	model = tf.keras.models.load_model(model_path)
	original, x = preprocess_image(Path(image_path), img_size=img_size)

	try:
		feature_tensor = get_connected_feature_tensor(tf, model, conv_layer_name)
		heatmap = make_gradcam_heatmap(tf, x, model, feature_tensor, class_index)
	except Exception:
		# Fallback for nested-model graphs where intermediate conv tensors are disconnected.
		heatmap = make_input_gradient_heatmap(tf, x, model, class_index)
	overlay = overlay_heatmap(original, heatmap)

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
	return str(out_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
	parser.add_argument("--model", default="artifacts/best_model.keras")
	parser.add_argument("--image", required=True)
	parser.add_argument("--output", default="artifacts/gradcam/gradcam.jpg")
	parser.add_argument("--class-index", type=int, default=None)
	parser.add_argument("--conv-layer", default=None)
	parser.add_argument("--img-size", type=int, default=224)
	args = parser.parse_args()

	out = generate_gradcam(
		model_path=args.model,
		image_path=args.image,
		output_path=args.output,
		class_index=args.class_index,
		conv_layer_name=args.conv_layer,
		img_size=args.img_size,
	)
	print(f"Grad-CAM saved to: {out}")


if __name__ == "__main__":
	main()
