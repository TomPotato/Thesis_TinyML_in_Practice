"""
Receive frames from the ESP-EYE over USB, watch the model's live prediction,
and save labeled images straight into the dataset folder layout used by
train_tinyresnet.py:

    <out_dir>/<subset>/{clear,left_blocked,center_blocked,right_blocked}/esp_*.png

The saved images are exactly what the model sees on the device (96x96
grayscale from the ESP-EYE camera), so they match deployment conditions.

Setup:
    python -m pip install pyserial opencv-python numpy
    Flash esp_eye_firmware.ino first. CLOSE the Arduino Serial Monitor
    (only one program can use the port at a time).

Find your port (Windows shows COM3, COM4, ...):
    python -m serial.tools.list_ports

Run:
    python camera_capture.py --port COM3 --out_dir ./data_own --subset train

Keys (click the video window first):
    1 / 2 / 3 / 4   select label: clear / left_blocked / center_blocked / right_blocked
    s               save the current frame with the selected label
    SPACE           start/stop burst saving (one frame every --interval seconds)
    e               start/stop LIVE EVALUATION (see below)
    q               quit

Live evaluation (measures the accuracy of the model running ON the board):
    1. Press 1-4 to select the TRUE class of what the camera is looking at.
    2. Arrange the scene (obstacle on the left / center / right, or nothing).
    3. Press e: every frame the board classifies is counted against that label.
       Wait for about 100 frames (the status line shows the count), press e again.
    4. Repeat for every class and for several arrangements, then press q.
    On exit it prints the confusion matrix, accuracy, precision and recall, and
    saves them to <out_dir>/live_eval_<session>.csv. These are the numbers for
    the "on device" column of the results table.

Workflow that works well: select a label, place the obstacle (or move the
rover/obstacle slightly between shots), then burst-save while moving things
around. The window shows what the model predicts in green if it agrees with
your selected label and orange if not, so mistakes are easy to spot.

Train/validation split: do NOT split randomly by frame (neighbouring frames
are almost identical). Instead record whole sessions: run once with
--subset train for most of your sessions, and with --subset val for a few
separate sessions (different day, lighting or arrangement).

If the image looks upside-down or mirrored, fix it in the firmware
(set_vflip / set_hmirror) and recapture, so training images and the live
camera stay identical. Do not flip images on the PC side.
"""

import argparse
import os
import re
import time

import cv2
import numpy as np

MAGIC = bytes([0xAA, 0x55, 0xA5, 0x5A])
IMG = 96
PIXELS = IMG * IMG
PACKET_LEN = 4 + 1 + 4 + PIXELS + 1   # magic, class, 4 scores, pixels, checksum
CLASS_NAMES = ["clear", "left_blocked", "center_blocked", "right_blocked"]
KEY_TO_CLASS = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3}
SCALE = 5  # display zoom (96 -> 480 px)


class PacketParser:
    """Extracts frame packets from a byte stream that may also contain text
    logs and partial packets. Resynchronises automatically on corruption."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        self.buf.extend(data)
        packets = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                if len(self.buf) > 3:       # keep a tail in case the magic is split
                    del self.buf[:-3]
                break
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < PACKET_LEN:
                break                        # wait for the rest of the packet
            pkt = bytes(self.buf[:PACKET_LEN])
            cls = pkt[4]
            scores = list(pkt[5:9])
            pix = pkt[9:9 + PIXELS]
            checksum = pkt[9 + PIXELS]
            if cls < len(CLASS_NAMES) and (sum(pix) & 0xFF) == checksum:
                frame = np.frombuffer(pix, dtype=np.uint8).reshape(IMG, IMG).copy()
                packets.append((cls, scores, frame))
                del self.buf[:PACKET_LEN]
            else:
                del self.buf[:1]             # false magic or corruption: resync
        return packets


def metrics_from_confusion(conf):
    """conf[true][pred] counts. Returns accuracy and per-class / macro precision
    and recall (macro = mean over the classes that actually occur)."""
    n = len(conf)
    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(n))
    prec, rec = [], []
    for i in range(n):
        col = sum(conf[r][i] for r in range(n))
        row = sum(conf[i])
        prec.append(conf[i][i] / col if col else 0.0)
        rec.append(conf[i][i] / row if row else 0.0)
    present = [i for i in range(n) if sum(conf[i]) > 0]
    macro_p = sum(prec[i] for i in present) / len(present) if present else 0.0
    macro_r = sum(rec[i] for i in present) / len(present) if present else 0.0
    blocked = sum(sum(conf[i]) for i in range(1, n))
    blocked_as_clear = sum(conf[i][0] for i in range(1, n))
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "precision": prec, "recall": rec,
        "macro_precision": macro_p, "macro_recall": macro_r,
        "blocked_as_clear": blocked_as_clear / blocked if blocked else 0.0,
    }


def report_live_eval(live, csv_path):
    m = metrics_from_confusion(live)
    if m["total"] == 0:
        return
    print("\n=== Live evaluation on the device ===")
    print(f"Frames counted: {m['total']}")
    print("Confusion matrix (rows = true class, columns = predicted by the board):")
    print(" " * 16 + "".join(f"{c[:12]:>14}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:>15} " + "".join(f"{v:>14}" for v in live[i]))
    print(f"Accuracy:        {m['accuracy'] * 100:.1f}%")
    print(f"Macro precision: {m['macro_precision'] * 100:.1f}%")
    print(f"Macro recall:    {m['macro_recall'] * 100:.1f}%")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>15}: precision {m['precision'][i] * 100:5.1f}%  recall {m['recall'][i] * 100:5.1f}%")
    print(f"Blocked frames classified as clear: {m['blocked_as_clear'] * 100:.1f}%")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("true\\predicted," + ",".join(CLASS_NAMES) + "\n")
            for i, cls in enumerate(CLASS_NAMES):
                f.write(cls + "," + ",".join(str(v) for v in live[i]) + "\n")
            f.write(f"accuracy,{m['accuracy']:.4f}\n")
            f.write(f"macro_precision,{m['macro_precision']:.4f}\n")
            f.write(f"macro_recall,{m['macro_recall']:.4f}\n")
        print(f"Saved to {csv_path}")
    except OSError as e:
        print(f"(could not save the csv: {e})")


def count_existing(out_dir, subset):
    counts = []
    for cls in CLASS_NAMES:
        folder = os.path.join(out_dir, subset, cls)
        counts.append(len(os.listdir(folder)) if os.path.isdir(folder) else 0)
    return counts


def save_frame(frame, out_dir, subset, cls_idx, session, n):
    folder = os.path.join(out_dir, subset, CLASS_NAMES[cls_idx])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"esp_{session}_{n:05d}.png")
    cv2.imwrite(path, frame)
    return path


def draw_ui(frame, pred, scores, selected, burst, counts, fps, subset, eval_info=None):
    big = cv2.resize(frame, (IMG * SCALE, IMG * SCALE), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((IMG * SCALE + 155, IMG * SCALE, 3), np.uint8)
    canvas[:IMG * SCALE] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    for k in (1, 2):  # left / center / right thirds
        x = IMG * SCALE * k // 3
        cv2.line(canvas, (x, 0), (x, IMG * SCALE), (255, 255, 255), 1)

    y0 = IMG * SCALE
    agree = pred == selected
    color = (0, 200, 0) if agree else (0, 140, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"model: {CLASS_NAMES[pred]} ({scores[pred] / 255:.2f})", (8, y0 + 24), font, 0.65, color, 2)
    cv2.putText(canvas, f"label: [{selected + 1}] {CLASS_NAMES[selected]}   subset={subset}", (8, y0 + 50), font, 0.55, (255, 255, 255), 1)
    state = "BURST SAVING" if burst else "paused (SPACE = burst, s = save one)"
    cv2.putText(canvas, state, (8, y0 + 74), font, 0.55, (0, 0, 255) if burst else (200, 200, 200), 1)
    short = ["clear", "left", "center", "right"]
    cv2.putText(canvas, "saved: " + "  ".join(f"{n}={c}" for n, c in zip(short, counts)), (8, y0 + 98), font, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"{fps:.1f} fps   e = live eval   q = quit", (8, y0 + 120), font, 0.45, (150, 150, 150), 1)
    if eval_info is not None:
        recording, n_frames, acc = eval_info
        text = (f"LIVE EVAL RECORDING: {n_frames} frames, accuracy {acc * 100:.0f}%" if recording
                else f"live eval paused: {n_frames} frames, accuracy {acc * 100:.0f}%")
        cv2.putText(canvas, text, (8, y0 + 144), font, 0.5, (0, 0, 255) if recording else (200, 200, 200), 1)
    return canvas


def readable_text(data, max_chars=400):
    """Printable ASCII fragments in a byte string (boot messages, log lines)."""
    parts = re.findall(rb"[\x20-\x7e]{6,}", data)
    return b" | ".join(parts).decode("ascii", "ignore")[-max_chars:]


def describe_ports():
    """Human-readable list of the serial ports pyserial can see."""
    try:
        from serial.tools import list_ports
        ports = [f"{p.device} ({p.description})" for p in list_ports.comports()]
        return "; ".join(ports) if ports else "none found"
    except Exception:
        return "could not list ports (run: python -m serial.tools.list_ports)"


def diagnose(total_bytes, sample, baud):
    """Explain why no valid frame has arrived yet."""
    if total_bytes == 0:
        return ("No data at all from the board. Ports I can see: " + describe_ports() + ". "
                "Check: the --port is the right one (Bluetooth ports also show up as COM ports but "
                "never send anything; look for USB Serial / CP210x / CH340); the Arduino "
                "Serial Monitor is closed; the board is powered and running the latest firmware; "
                "ESP32-CAM: GPIO0-GND jumper removed and reset pressed; the USB cable carries data.")
    text = readable_text(sample, max_chars=160)
    printable = sum(32 <= b < 127 or b in (10, 13) for b in sample) / max(1, len(sample))
    if printable > 0.8:
        return (f"The board is talking ({total_bytes} bytes) but not streaming frames. It says: \"{text}\"\n"
                "  -> Probably an OLD firmware without streaming (reflash the latest esp_eye_firmware.ino), "
                "or the camera/model failed to start (open the Serial Monitor at 921600 to read the error).")
    return (f"Receiving {total_bytes} bytes but they are not valid frames, and most look like garbage.\n"
            f"  -> Likely a baud mismatch: --baud is {baud}, make it equal SERIAL_BAUD in the firmware. "
            "Also check the power supply (brownouts corrupt output).")


def draw_waiting(waited, total_bytes):
    canvas = np.zeros((IMG * SCALE + 155, IMG * SCALE, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Waiting for frames from the board...", (20, 200), font, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, f"{waited:.0f} s, {total_bytes} bytes received", (20, 235), font, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, "See the terminal for hints.  q = quit", (20, 270), font, 0.5, (150, 150, 150), 1)
    return canvas


def open_serial(port, baud):
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial is missing. Install it with:  python -m pip install pyserial")
    try:
        # Create the port closed, switch DTR/RTS off, THEN open it. By default
        # pyserial asserts both lines on open, which on boards with an
        # auto-reset circuit holds the chip in reset (so it sends nothing).
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baud
        ser.timeout = 0.05
        ser.dtr = False
        ser.rts = False
        ser.open()
        return ser
    except serial.SerialException as e:
        raise SystemExit(f"Could not open {port}: {e}\n"
                         "Close the Arduino Serial Monitor, and check the port with "
                         "python -m serial.tools.list_ports")


def run(args, ser):
    parser = PacketParser()
    session = time.strftime("%Y%m%d_%H%M%S")
    counts = count_existing(args.out_dir, args.subset)
    selected = 0
    burst = False
    evaluating = False
    live = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]   # live[true][predicted by the board]
    last_save_t = 0.0
    save_n = 0
    latest = None
    frame_id = 0
    last_saved_id = -1
    times = []

    # Opening the port usually resets the ESP32: wait for it to boot, then ask
    # for the stream, repeating the request until the first frame arrives.
    print(f"Opened {args.port} at {args.baud} baud. Waiting for the board to boot...")
    if not args.no_reset:
        # Pulse RTS to reset the board (works on boards with an auto-reset
        # circuit, like the ESP32-CAM-MB base). This makes it
        # print its startup messages, which are shown below.
        try:
            ser.dtr = False
            ser.rts = True
            time.sleep(0.1)
            ser.rts = False
        except Exception:
            pass
    time.sleep(args.boot_wait)
    boot = ser.read(ser.in_waiting) if ser.in_waiting else b""
    boot_text = readable_text(boot)
    if boot_text:
        print(f"Board said at startup: {boot_text}")
    last_request = 0.0
    wait_start = time.time()
    last_hint = wait_start
    hints_shown = 0
    total_bytes = 0
    sample = bytearray(boot)

    print("Waiting for frames (press q in the window to quit)...")
    try:
        while True:
            now = time.time()
            if latest is None and now - last_request > 1.0:
                ser.write(b"v")
                last_request = now

            data = ser.read(max(1, ser.in_waiting))
            total_bytes += len(data)
            sample.extend(data)
            del sample[:-600]
            for cls, scores, frame in parser.feed(data):
                latest = (cls, scores, frame)
                frame_id += 1
                if evaluating:
                    live[selected][cls] += 1
                times.append(time.time())
                times[:] = times[-20:]

            if latest is None:
                cv2.imshow("ESP-EYE capture", draw_waiting(now - wait_start, total_bytes))
                if now - last_hint > 8 and hints_shown < 3:
                    print("No valid frame yet. " + diagnose(total_bytes, bytes(sample), args.baud))
                    last_hint = now
                    hints_shown += 1

            if latest is not None:
                cls, scores, frame = latest
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
                n_live = sum(sum(r) for r in live)
                live_acc = sum(live[i][i] for i in range(len(live))) / n_live if n_live else 0.0
                cv2.imshow("ESP-EYE capture", draw_ui(frame, cls, scores, selected, burst, counts, fps,
                                                       args.subset, (evaluating, n_live, live_acc)))

                if burst and frame_id != last_saved_id and now - last_save_t >= args.interval:
                    save_frame(frame, args.out_dir, args.subset, selected, session, save_n)
                    counts[selected] += 1
                    save_n += 1
                    last_save_t, last_saved_id = now, frame_id

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in KEY_TO_CLASS:
                selected = KEY_TO_CLASS[key]
            elif key == ord(" "):
                burst = not burst
            elif key == ord("e"):
                evaluating = not evaluating
            elif key == ord("s") and latest is not None and frame_id != last_saved_id:
                save_frame(latest[2], args.out_dir, args.subset, selected, session, save_n)
                counts[selected] += 1
                save_n += 1
                last_saved_id = frame_id
    finally:
        try:
            ser.write(b"x")  # stop streaming so the Serial Monitor is readable again
        except Exception:
            pass
        ser.close()
        cv2.destroyAllWindows()
    print(f"Done. Saved {save_n} images this session into {os.path.join(args.out_dir, args.subset)}")
    report_live_eval(live, os.path.join(args.out_dir, f"live_eval_{session}.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600, help="Must match SERIAL_BAUD in the firmware")
    ap.add_argument("--out_dir", default="./data_own")
    ap.add_argument("--subset", choices=["train", "val"], default="train")
    ap.add_argument("--interval", type=float, default=0.3, help="Seconds between saved frames in burst mode")
    ap.add_argument("--no_reset", action="store_true", help="Do not reset the board when connecting")
    ap.add_argument("--boot_wait", type=float, default=2.5, help="Seconds to wait after opening the port")
    args = ap.parse_args()
    run(args, open_serial(args.port, args.baud))


if __name__ == "__main__":
    main()