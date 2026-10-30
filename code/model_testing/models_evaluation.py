"""
Compute: accuracy, precision, recall and size of the float (Keras) model
and of the quantized int8 TFLite model, on the SAME images.

    python models_evaluation.py \
        --keras tiny_resnet_obstacle.keras \
        --tflite esp_model/tiny_resnet_obstacle_int8.tflite \
        --eval_dir data/val \
        --device_ms 685

--eval_dir must contain the class folders (clear, left_blocked, center_blocked,
right_blocked). For honest results use images that were NOT used to pick the
best epoch or the hyperparameters: a separate TEST set, ideally recorded in
separate sessions with the deployed camera, e.g.

    python camera_capture.py --port COM3 --out_dir ./data_test --subset val
    python models_evaluation.py ... --eval_dir data_test/val

Precision and recall are MACRO averages: computed per class and then averaged
over the classes that occur in the evaluation images. The per-class values and
the confusion matrices are printed as well.

Everything is computed on the PC. The int8 model is run with the TensorFlow
Lite interpreter, which gives (almost) the same predictions as the board. For
the numbers measured ON the board use the live evaluation of camera_capture.py
(key e) and the "inference=" times printed by the firmware.

The last lines are ready to paste into the LaTeX table.
"""

import argparse
import os
import time

import cv2
import numpy as np

CLASS_NAMES = ["clear", "left_blocked", "center_blocked", "right_blocked"]
IMG = 96


# ---------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, n=len(CLASS_NAMES)):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm):
    """cm[true][pred]. Returns accuracy, per-class and macro precision/recall."""
    n = len(cm)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0
    precision, recall = [], []
    for i in range(n):
        col, row = cm[:, i].sum(), cm[i].sum()
        precision.append(float(cm[i, i] / col) if col else 0.0)
        recall.append(float(cm[i, i] / row) if row else 0.0)
    present = [i for i in range(n) if cm[i].sum() > 0]
    macro_p = float(np.mean([precision[i] for i in present])) if present else 0.0
    macro_r = float(np.mean([recall[i] for i in present])) if present else 0.0
    blocked = cm[1:].sum()
    blocked_as_clear = float(cm[1:, 0].sum() / blocked) if blocked else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "macro_precision": macro_p, "macro_recall": macro_r,
            "blocked_as_clear": blocked_as_clear}


def print_report(title, cm):
    m = metrics_from_confusion(cm)
    print(f"\n=== {title} ===")
    print("Confusion matrix (rows = true class, columns = predicted):")
    print(" " * 16 + "".join(f"{c[:12]:>14}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:>15} " + "".join(f"{v:>14}" for v in cm[i]))
    print(f"Accuracy:        {m['accuracy'] * 100:.1f}%")
    print(f"Macro precision: {m['macro_precision'] * 100:.1f}%")
    print(f"Macro recall:    {m['macro_recall'] * 100:.1f}%")
    for i, cls in enumerate(CLASS_NAMES):
        if cm[i].sum() == 0:
            print(f"  {cls:>15}: (no images of this class)")
        else:
            print(f"  {cls:>15}: precision {m['precision'][i] * 100:5.1f}%   recall {m['recall'][i] * 100:5.1f}%")
    print(f"Blocked images classified as clear: {m['blocked_as_clear'] * 100:.1f}%")
    return m


# ---------------------------------------------------------------- data
def load_images(eval_dir):
    xs, ys = [], []
    counts = {}
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(eval_dir, cls)
        if not os.path.isdir(folder):
            print(f"  (no folder for class '{cls}')")
            continue
        n = 0
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG, IMG):
                img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
            xs.append(img.astype(np.float32) / 255.0)
            ys.append(idx)
            n += 1
        counts[cls] = n
    if not xs:
        raise SystemExit(f"No images found under {eval_dir}")
    print("Images per class:", ", ".join(f"{k}={v}" for k, v in counts.items()), f"(total {len(xs)})")
    return np.stack(xs)[..., None], np.array(ys)


# ---------------------------------------------------------------- models
def predict_keras(model, x):
    probs = model.predict(x, batch_size=64, verbose=0)
    return np.argmax(probs, axis=1)


def load_interpreter(tflite_path):
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    with open(tflite_path, "rb") as f:          # bytes: avoids path problems with accented characters
        data = f.read()
    interp = Interpreter(model_content=data)
    interp.allocate_tensors()
    return interp, len(data)


def predict_int8(interp, x):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero = inp["quantization"]
    preds, times = [], []
    for img in x:
        q = np.clip(np.round(img / scale + zero), -128, 127).astype(inp["dtype"])
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], q.reshape(1, IMG, IMG, 1))
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0]
        times.append((time.perf_counter() - t0) * 1000.0)
        preds.append(int(np.argmax(raw)))
    return np.array(preds), float(np.mean(times))


def float_tflite_size(model):
    import tensorflow as tf
    return len(tf.lite.TFLiteConverter.from_keras_model(model).convert())


def pct(v):
    return f"{v * 100:.1f}\\%"


def kb(n):
    return f"{n / 1024:.1f}~KB"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keras", required=True, help="Trained float model (.keras)")
    ap.add_argument("--tflite", required=True, help="Quantized int8 model (.tflite)")
    ap.add_argument("--eval_dir", required=True, help="Folder with the four class folders")
    ap.add_argument("--device_ms", type=float, help="Mean inference time measured on the board, in ms")
    args = ap.parse_args()

    import tensorflow as tf
    x, y = load_images(args.eval_dir)

    model = tf.keras.models.load_model(args.keras)
    pred_f = predict_keras(model, x)
    interp, tflite_bytes = load_interpreter(args.tflite)
    pred_q, pc_ms = predict_int8(interp, x)

    mf = print_report("Float model (Keras)", confusion_matrix(y, pred_f))
    mq = print_report("Quantized int8 model (TFLite interpreter)", confusion_matrix(y, pred_q))

    agree = float(np.mean(pred_f == pred_q))
    print(f"\nFloat and int8 models give the same prediction on {agree * 100:.1f}% of the images.")
    print("A large gap between the two accuracies points to quantization loss "
          "(try a more varied representative dataset in Model_Quantizing.py).")

    keras_kb = os.path.getsize(args.keras)
    float_tflite = float_tflite_size(model)
    print(f"\nParameters: {model.count_params()}")
    print(f"Sizes: .keras file {kb(keras_kb)}, float32 .tflite {kb(float_tflite)}, int8 .tflite {kb(tflite_bytes)}")
    print(f"PC inference time of the int8 model: {pc_ms:.2f} ms/image")

    print(f"Accuracy       & {pct(mf['accuracy'])} & {pct(mq['accuracy'])} \\\\")
    print(f"Precision      & {pct(mf['macro_precision'])} & {pct(mq['macro_precision'])} \\\\")
    print(f"Recall         & {pct(mf['macro_recall'])} & {pct(mq['macro_recall'])} \\\\")
    print(f"Model size     & {kb(float_tflite)} & {kb(tflite_bytes)} \\\\")
    print(f"Inference time & n/a \\\\")
    print("% Precision and recall are macro averages over the classes present in the evaluation images.")


if __name__ == "__main__":
    main()