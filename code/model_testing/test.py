"""
Test the quantized int8 TFLite model on your PC before flashing it.

Three modes (pick one):

  1) Evaluate on a folder of labeled images (accuracy + confusion matrix):
       python test_tflite_model.py --model model/quantized/tiny_resnet_obstacle_int8.tflite --eval_dir data/val

  2) Classify a single image:
       python test_tflite_model.py --model model/quantized/tiny_resnet_obstacle_int8.tflite --image photo.jpg

  3) Live webcam test (press q to quit):
       python test_tflite_model.py --model model/quantized/tiny_resnet_obstacle_int8.tflite --webcam

Optional: add --keras tiny_resnet_obstacle.keras to --eval_dir to compare the
quantized model against the original float model on the same images. A drop of
more than a few percent means quantization is hurting accuracy (usually a
poor representative dataset in quantize_and_convert.py).

Preprocessing here matches training: grayscale, whole frame squashed to
96x96 (INTER_AREA), pixel values scaled to 0-1.
"""

import argparse
import os
import time

import cv2
import numpy as np

CLASS_NAMES = ["clear", "left_blocked", "center_blocked", "right_blocked"]
IMG_SIZE = 96


def load_interpreter(model_path):
    """Load a TFLite model and allocate its tensors."""
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    with open(model_path, "rb") as f:
        interp = Interpreter(model_content=f.read())
    interp.allocate_tensors()
    return interp


def preprocess(img_bgr_or_gray):
    """Any image -> float32 array in [0, 1] of shape (96, 96)."""
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def predict(interp, x01):
    """Run one 96x96 image (floats in [0,1]) through the int8 model.
    Returns (class_index, scores) with scores dequantized to roughly 0-1."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    scale, zero_point = inp["quantization"]
    q = np.round(x01 / scale + zero_point)
    q = np.clip(q, -128, 127).astype(inp["dtype"])
    interp.set_tensor(inp["index"], q.reshape(1, IMG_SIZE, IMG_SIZE, 1))
    interp.invoke()

    raw = interp.get_tensor(out["index"])[0].astype(np.float32)
    o_scale, o_zero = out["quantization"]
    scores = (raw - o_zero) * o_scale
    return int(np.argmax(scores)), scores


def eval_folder(interp, eval_dir, keras_model=None):
    n = len(CLASS_NAMES)
    confusion = np.zeros((n, n), dtype=int)
    keras_correct, total, times = 0, 0, []

    for true_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(eval_dir, cls)
        if not os.path.isdir(folder):
            print(f"  (no folder for class '{cls}', skipping)")
            continue
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            x = preprocess(img)
            t0 = time.perf_counter()
            pred, _ = predict(interp, x)
            times.append((time.perf_counter() - t0) * 1000)
            confusion[true_idx, pred] += 1
            total += 1
            if keras_model is not None:
                kp = int(np.argmax(keras_model.predict(x.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)))
                keras_correct += int(kp == true_idx)

    if total == 0:
        print("No images found. Expected folders like", os.path.join(eval_dir, CLASS_NAMES[0]))
        return

    acc = np.trace(confusion) / total
    print(f"\nQuantized int8 accuracy: {acc * 100:.1f}%  ({np.trace(confusion)}/{total})")
    if keras_model is not None:
        print(f"Float Keras accuracy:    {keras_correct / total * 100:.1f}%  (same images)")

    print("\nConfusion matrix (rows = true class, columns = predicted):")
    header = "".join(f"{c[:12]:>14}" for c in CLASS_NAMES)
    print(" " * 16 + header)
    for i, cls in enumerate(CLASS_NAMES):
        row_total = confusion[i].sum()
        recall = f"{confusion[i, i] / row_total * 100:5.1f}%" if row_total else "   n/a"
        print(f"{cls:>15} " + "".join(f"{v:>14}" for v in confusion[i]) + f"   recall {recall}")

    print(f"\nPC inference time: {np.mean(times):.2f} ms/image on average "
          "(PC speed only; the ESP-EYE will be far slower, measure it on the device).")
    print("Watch the safety-relevant mistakes: blocked classes predicted as 'clear'.")


def classify_image(interp, path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    idx, scores = predict(interp, preprocess(img))
    print(f"Prediction: {CLASS_NAMES[idx]}")
    for name, s in zip(CLASS_NAMES, scores):
        print(f"  {name:>15}: {s:.2f}")


def webcam(interp, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (try --camera 1).")
    print("Webcam running. Press q in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx, scores = predict(interp, preprocess(frame))
        text = f"{CLASS_NAMES[idx]}  ({scores[idx]:.2f})"
        color = (0, 200, 0) if idx == 0 else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # Draw thirds so you can see which zone an obstacle falls in
        h, w = frame.shape[:2]
        for k in (1, 2):
            cv2.line(frame, (w * k // 3, 0), (w * k // 3, h), (255, 255, 255), 1)
        # Show what the model actually sees
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frame[h - IMG_SIZE - 5:h - 5, 5:5 + IMG_SIZE] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        cv2.imshow("tiny-ResNet test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the int8 .tflite file")
    ap.add_argument("--eval_dir", help="Folder with clear/, left_blocked/, ... subfolders (e.g. data/val)")
    ap.add_argument("--image", help="Classify a single image")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index")
    ap.add_argument("--keras", help="Optional float .keras model to compare against (with --eval_dir)")
    args = ap.parse_args()

    interp = load_interpreter(args.model)
    inp = interp.get_input_details()[0]
    print(f"Model input: shape={inp['shape']}, dtype={inp['dtype'].__name__}, quantization={inp['quantization']}")

    if args.eval_dir:
        keras_model = None
        if args.keras:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(args.keras)
        eval_folder(interp, args.eval_dir, keras_model)
    elif args.image:
        classify_image(interp, args.image)
    elif args.webcam:
        webcam(interp, args.camera)
    else:
        ap.error("Choose one of --eval_dir, --image or --webcam")


if __name__ == "__main__":
    main()