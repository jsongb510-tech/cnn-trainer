"""Machine-readable progress lines for the GUI (gui/trainer_gui.py).

Each message is one stdout line: "@@" + JSON object with a "type" field.
Types: start, batch, epoch, mistakes, result.
"""
import base64
import json

import numpy as np

MAX_MISTAKES = 3000  # cap on misclassified images sent to the GUI (~4 KB each for 32x32 RGB)


def emit(kind, **data):
    print("@@" + json.dumps({"type": kind, **data}), flush=True)


def mistakes(images, y_true, probs, limit=MAX_MISTAKES):
    """(count, items) for misclassified images, most confidently wrong first.

    images: uint8 array (N, H, W) or (N, H, W, C); pixels are sent as base64 raw bytes.
    """
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(y_pred != y_true)
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])]
    return int(len(wrong)), [
        {"index": int(i), "true": int(y_true[i]), "pred": int(y_pred[i]),
         "conf": float(probs[i, y_pred[i]]), "true_prob": float(probs[i, y_true[i]]),
         "img": base64.b64encode(np.ascontiguousarray(images[i], dtype=np.uint8).tobytes()).decode("ascii")}
        for i in wrong[:limit]
    ]
