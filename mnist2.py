import argparse
import base64
import json
import os
import time

# Default to the PyTorch backend; override with e.g. KERAS_BACKEND=tensorflow python mnist2.py
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from keras import layers

NUM_CLASSES = 10
MAX_MISTAKES = 500  # cap on misclassified images sent to the GUI


def parse_args():
    p = argparse.ArgumentParser(description="Keras CNN on MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--json", action="store_true",
                   help="print machine-readable @@{...} progress lines instead of Keras logs (used by the GUI)")
    return p.parse_args()


def emit(kind, **data):
    print("@@" + json.dumps({"type": kind, **data}), flush=True)


class JsonProgress(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch, self.t0 = epoch, time.time()

    def on_train_batch_end(self, batch, logs=None):
        steps = self.params["steps"]
        if batch % 20 == 0 or batch == steps - 1:
            emit("batch", epoch=self.epoch + 1, batch=batch + 1, steps=steps)

    def on_epoch_end(self, epoch, logs=None):
        emit("epoch", epoch=epoch + 1, epochs=self.params["epochs"], seconds=time.time() - self.t0,
             **{k: float(v) for k, v in (logs or {}).items()})


def build_model(dropout):
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),  # 28 -> 14
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),  # 14 -> 7
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES),
    ])


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def print_confusion_matrix(cm):
    width = max(5, len(str(cm.max())) + 1)
    print("rows = true label, cols = predicted label")
    print("     " + "".join(f"{c:>{width}}" for c in range(cm.shape[1])))
    for r, row in enumerate(cm):
        print(f"{r:>4} " + "".join(f"{v:>{width}}" for v in row))


def mistakes(images, y_true, probs, limit):
    """Misclassified test images, most confidently wrong first. Pixels are base64 raw 28x28 uint8."""
    y_pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(y_pred != y_true)
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])]
    return int(len(wrong)), [
        {"index": int(i), "true": int(y_true[i]), "pred": int(y_pred[i]),
         "conf": float(probs[i, y_pred[i]]), "true_prob": float(probs[i, y_true[i]]),
         "img": base64.b64encode(images[i].tobytes()).decode("ascii")}
        for i in wrong[:limit]
    ]


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
    backend, device = keras.backend.backend(), device_info()
    if args.json:
        emit("start", keras=keras.__version__, backend=backend, device=device, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, dropout=args.dropout)
    else:
        print(f"Keras {keras.__version__} | backend: {backend} | device: {device}")

    (x_train, y_train), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
    x_train = ((x_train.astype("float32") / 255.0 - 0.1307) / 0.3081)[..., None]
    x_test = ((x_test_raw.astype("float32") / 255.0 - 0.1307) / 0.3081)[..., None]

    model = build_model(args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    t0 = time.time()
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_test, y_test), verbose=0 if args.json else 2,
              callbacks=[JsonProgress()] if args.json else [])
    train_seconds = time.time() - t0

    logits = model.predict(x_test, batch_size=1000, verbose=0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, NUM_CLASSES)
    correct, total = int(np.trace(cm)), int(cm.sum())

    model.save("mnist2_cnn.keras")
    if args.json:
        n_wrong, items = mistakes(x_test_raw, y_test, probs, MAX_MISTAKES)
        emit("mistakes", total=n_wrong, items=items)
        emit("result", accuracy=correct / total, correct=correct, total=total, confusion=cm.tolist(),
             train_seconds=train_seconds, model_path="mnist2_cnn.keras")
        return

    print("\nConfusion matrix (test set):")
    print_confusion_matrix(cm)
    print(f"\nTest accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
    print("saved model to mnist2_cnn.keras")


if __name__ == "__main__":
    main()
