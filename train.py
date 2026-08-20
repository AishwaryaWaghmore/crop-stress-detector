from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_tensorflow():
	try:
		import importlib

		tf = importlib.import_module("tensorflow")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"TensorFlow is not installed. Use Python 3.10/3.11 and install requirements.txt."
		) from exc
	return tf


def build_model(tf, num_classes: int, img_size: int = 224):
	keras = tf.keras
	layers = tf.keras.layers
	augment = keras.Sequential(
		[
			layers.RandomFlip("horizontal"),
			layers.RandomRotation(0.05),
			layers.RandomZoom(0.1),
		],
		name="augmentation",
	)
	base = keras.applications.EfficientNetB0(
		include_top=False,
		input_shape=(img_size, img_size, 3),
		weights="imagenet",
	)
	base.trainable = False

	inputs = keras.Input(shape=(img_size, img_size, 3))
	x = augment(inputs)
	x = keras.applications.efficientnet.preprocess_input(x)
	x = base(x, training=False)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dropout(0.3)(x)
	outputs = layers.Dense(num_classes, activation="softmax")(x)
	model = keras.Model(inputs, outputs)
	return model


def make_datasets(data_dir: Path, img_size: int, batch_size: int):
	tf = require_tensorflow()
	keras = tf.keras
	autotune = tf.data.AUTOTUNE

	train_dir = data_dir / "train"
	val_dir = data_dir / "val"

	train_ds = keras.utils.image_dataset_from_directory(
		train_dir,
		image_size=(img_size, img_size),
		batch_size=batch_size,
		shuffle=True,
	)
	val_ds = keras.utils.image_dataset_from_directory(
		val_dir,
		image_size=(img_size, img_size),
		batch_size=batch_size,
		shuffle=False,
	)

	class_names = train_ds.class_names

	train_ds = train_ds.prefetch(autotune)
	val_ds = val_ds.prefetch(autotune)
	return train_ds, val_ds, class_names


def main() -> None:
	parser = argparse.ArgumentParser(description="Train crop disease classifier")
	parser.add_argument("--data-dir", default="dataset/combined", help="Directory with train/ and val/")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--img-size", type=int, default=224)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--fine-tune-epochs", type=int, default=3, help="Extra epochs after unfreezing top EfficientNet layers")
	parser.add_argument("--fine-tune-at", type=int, default=200, help="Layer index to start unfreezing from")
	parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
	parser.add_argument("--output-dir", default="artifacts")
	parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for train batches per epoch")
	parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for validation batches")
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	tf = require_tensorflow()
	keras = tf.keras

	train_ds, val_ds, class_names = make_datasets(
		data_dir=data_dir,
		img_size=args.img_size,
		batch_size=args.batch_size,
	)

	if args.max_train_batches is not None and args.max_train_batches > 0:
		train_ds = train_ds.take(args.max_train_batches)
	if args.max_val_batches is not None and args.max_val_batches > 0:
		val_ds = val_ds.take(args.max_val_batches)

	model = build_model(tf=tf, num_classes=len(class_names), img_size=args.img_size)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	ckpt_path = output_dir / "best_model.keras"
	callbacks = [
		keras.callbacks.ModelCheckpoint(
			filepath=str(ckpt_path),
			monitor="val_accuracy",
			save_best_only=True,
			mode="max",
		),
		keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
		keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
	]

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=args.epochs,
		callbacks=callbacks,
	)

	if args.fine_tune_epochs > 0:
		base_model = model.get_layer("efficientnetb0")
		base_model.trainable = True
		for layer in base_model.layers[: args.fine_tune_at]:
			layer.trainable = False

		model.compile(
			optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"],
		)

		fine_history = model.fit(
			train_ds,
			validation_data=val_ds,
			epochs=args.epochs + args.fine_tune_epochs,
			initial_epoch=args.epochs,
			callbacks=callbacks,
		)

		for k, vals in fine_history.history.items():
			history.history.setdefault(k, [])
			history.history[k].extend(vals)

	with (output_dir / "class_names.txt").open("w", encoding="utf-8") as f:
		for name in class_names:
			f.write(f"{name}\n")

	history_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
	with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
		json.dump(history_json, f, indent=2)

	print(f"Training complete. Best model saved to: {ckpt_path}")


if __name__ == "__main__":
	main()
