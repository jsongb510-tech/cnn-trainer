"""CIFAR-10 classification with the same VGG-style CNN as cifar10_torch.py, written in Keras 3."""
import argparse
import importlib.util
import os
import time

# Backend: KERAS_BACKEND if set, otherwise whichever of torch / tensorflow the active venv has
# (.venv has torch, .venv-tf has tensorflow). Override with e.g. KERAS_BACKEND=tensorflow python cifar10_keras.py
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch" if importlib.util.find_spec("torch") else "tensorflow"

import keras
import numpy as np
from keras import layers

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SHORT = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype="float32")
STD = np.array([0.2470, 0.2435, 0.2616], dtype="float32")


def parse_args():
    p = argparse.ArgumentParser(description="Keras CNN on CIFAR-10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    return p.parse_args()


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D()(x)


def build_model():
    inputs = keras.Input(shape=(32, 32, 3))
    # Augmentation layers are only active during training (identity at test time).
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.125, 0.125, fill_mode="constant")(x)  # up to 4 px, like padding=4 + random crop
    for filters in (64, 128, 256):  # 32 -> 16 -> 8 -> 4
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, dtype="float32")(x)  # keep logits in float32 under mixed precision
    return keras.Model(inputs, outputs)


def print_confusion_matrix(cm):
    print("rows = true class, cols = predicted class")
    print(" " * 7 + "".join(f"{c:>7}" for c in SHORT))
    for name, row in zip(SHORT, cm):
        print(f"{name:>6} " + "".join(f"{v:>7}" for v in row))


def device_info():
    backend = keras.backend.backend()
    if backend == "torch":
        import torch
        return f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    if backend == "tensorflow":
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"gpu ({tf.config.experimental.get_device_details(gpus[0]).get('device_name')})" if gpus else "cpu"
    return "unknown"


def main():
    args = parse_args()
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    print(f"Keras {keras.__version__} | backend: {keras.backend.backend()} | device: {device_info()}")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.ravel(), y_test.ravel()
    x_train = (x_train.astype("float32") / 255.0 - MEAN) / STD
    x_test = (x_test.astype("float32") / 255.0 - MEAN) / STD

    steps = args.epochs * (len(x_train) // args.batch_size)
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps),
            weight_decay=args.weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    t_start = time.time()
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_test, y_test), verbose=2)
    print(f"\nTraining time: {time.time() - t_start:.0f}s")

    y_pred = model.predict(x_test, batch_size=1000, verbose=0).argmax(axis=1)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y_test, y_pred), 1)

    print("\nConfusion matrix (test set):")
    print_confusion_matrix(cm)
    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASSES):
        print(f"  {name:<10} {100 * cm[i, i] / cm[i].sum():6.2f}%")
    correct, total = int(np.trace(cm)), int(cm.sum())
    print(f"\nTest accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

    model.save("cifar10_cnn.keras")
    print("saved model to cifar10_cnn.keras")


if __name__ == "__main__":
    main()
