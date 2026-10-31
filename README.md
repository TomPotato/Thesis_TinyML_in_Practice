# TinyML in Practice: Developing and Deploying an Image Recognition Model on custom Embedded Hardware

**Author:** Tommaso Agosti
**Supervisor:** Kasim Sinan Yildirim
**Institution:** University of Trento, Department of Information Engineering and Computer Science
**Degree:** Bachelor's in Computer, Communication and Electronic Engineering
**Year:** 2026

## Overview

This repository contains the code developed for my bachelor thesis:

*"TinyML in Practice: Developing and Deploying an Image Recognition Model on custom Embedded Hardware"*

The thesis follows the whole development process of deploying a neural network directly on a microcontroller (TinyML) through a practical application: a small rover that drives to GPS destinations sent from a smartphone while avoiding obstacles. A camera classifier running on an **ESP32-CAM** labels each frame as `clear`, `left_blocked`, `center_blocked` or `right_blocked`; an **Arduino Mega** combines this result with ultrasonic sensors, a GPS receiver, a gyroscope and wheel encoders, and drives the motors.

The repository contains:

- a very small ResNet-style classifier (about 30.5 thousand trainable parameters, about 9.5 million MACs per image);
- converters that build the dataset from public datasets (NYU Depth V2, COCO 2017);
- training, 8-bit quantization and evaluation scripts;
- the ESP32-CAM firmware, and a PC tool to stream frames from the board, collect labelled images and evaluate the model live;
- the Arduino Mega firmware of the rover.

## Repository Structure

```text
thesis-project/
│
├── code/
│   ├── Arduino_side/          # firmware: ESP32-CAM, Arduino Mega, test sketches
│   ├── ML_model/              # dataset converters, training, quantization
│   ├── model_testing/         # evaluation scripts, PC capture and live-evaluation tool
│   └── preprocessing_scripts  # image processing scripts
│
├── data/                    # compressed in data.zip (too large for the repository)
│   ├── train/
│   │   ├── clear/
│   │   ├── center_blocked/
│   │   ├── left_blocked/
│   │   └── right_blocked/
│   └── val/
│       ├── clear/
│       ├── center_blocked/
│       ├── left_blocked/
│       └── right_blocked/
│
├── data_own/                # images captured with the ESP32-CAM (compressed .zip file)
│   └── val/
│       ├── clear/
│       ├── center_blocked/
│       ├── left_blocked/
│       └── right_blocked/
│
├── images/
│   └── readme/
│
├── model/
│   ├── not_quantized/       # float Keras model
│   └── quantized/           # int8 .tflite and model.h (C array for the firmware)
│
├── data.zip
├── requirements.txt
├── LICENSE
└── README.md
```

### Files and what they do

| File | Folder | Purpose |
|---|---|---|
| `nyu_preprocessing.py` | `code/preprocessing_scripts` | Converts the NYU Depth V2 labeled `.mat` into the four classes, labelling each frame from its depth map |
| `coco_preprocessing.py` | `code/preprocessing_scripts` | Converts COCO 2017 bounding boxes into the four classes |
| `Model_Training.py` | `code/ML_model` | Defines and trains the network (Keras) |
| `Model_Quantizing.py` | `code/ML_model` | 8-bit post-training quantization, writes the `.tflite` file and `model.h` |
| `test.py` | `code/model_testing` | Runs the int8 model on the PC on a folder, an image or a webcam |
| `models_evaluation.py` | `code/model_testing` | Accuracy, macro precision and recall, sizes of the float and int8 models |
| `camera_capture.py` | `code/model_testing` | Receives frames from the board over USB, saves labelled images, live evaluation |
| `esp32_cam_firmware.ino` | `code/Arduino_side` | ESP32-CAM firmware: capture, inference, class byte over UART, frame streaming |
| `mega_side_esp32.ino` | `code/Arduino_side` | Arduino Mega firmware: sensor fusion, navigation, motor control, Bluetooth commands |

## Installation

Clone the repository:

```bash
git clone https://github.com/TomPotato/Thesis_TinyML_in_Practice.git
cd Thesis_TinyML_in_Practice
```

Install the Python dependencies (a virtual environment is recommended):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For the firmware you also need the **Arduino IDE** with:

- the Espressif **esp32** board package version, 2.0.1 used in the thesis;
- the TensorFlow Lite Micro Arduino library;
- the **TinyGPSPlus** library for the Arduino Mega sketch.

## Usage

To reproduce the experiments, follow these steps. The data folder with the converted dataset is available as `data.zip`; if you use it, you can skip steps 1 to 3.

### 1. Download and extract the public datasets

- **COCO 2017**: the `train2017` and `val2017` images and `annotations_trainval2017.zip` from <https://cocodataset.org>.
- **NYU Depth V2**: the *labeled* subset `nyu_depth_v2_labeled.mat` from <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>.

### 2. Convert the datasets to the four classes

```bash
python nyu_preprocessing.py --mat nyu_depth_v2_labeled.mat --out_dir ./data_nyu --add_flips

python coco_preprocessing.py --ann annotations/instances_train2017.json --images train2017 --out_dir ./data_coco --max_per_class 2000 --add_flips
```

Both scripts write `train/` and `val/` folders and a `preview.png` contact sheet: look at it to check that the automatic labels are reasonable. All images are saved as $96 \times 96$ grayscale.

### 3. Merge the datasets
Create a new folder named data and drag and drop the train and val folders from both NYU and COCO into it to merge them. The images will not overlap, as they have different filenames.

Images captured with the ESP32-CAM (see step 8) are added to the same class folders.

### 4. Train the model

```bash
python Model_Training.py --data_dir data --epochs 40 --lr 1e-6 --out model/not_quantized/tiny_resnet_not_quantized.keras                                                          
```

The best model is saved as `tiny_resnet_not_quantized.keras`.

### 5. Quantize and convert

```bash
python Model_Quantizing.py --model model/not_quantized/tiny_resnet_not_quantized.keras --data_dir data --out_dir model/quantized                                                      
```

This writes the int8 `.tflite` file and `model.h`, the C array to be compiled into the firmware. The calibration images are taken from the dataset folder (see the note on the calibration set in the thesis).

### 6. Evaluate the models

```bash
python models_evaluation.py \
    --keras model/not_quantized/tiny_resnet_not_quantized.keras \
    --tflite model/quantized/tiny_resnet_obstacle_int8.tflite \
    --eval_dir data/val --device_ms 685
```

It prints the confusion matrices, per-class and macro precision and recall and the model sizes. `test.py` runs the int8 model on a folder (`--eval_dir`), a single image (`--image`) or a webcam (`--webcam`).

### 7. Deploy on the ESP32-CAM

1. Copy `model.h` next to `esp32_cam_firmware.ino`.
2. In the sketch, modify  `CAMERA_ROTATE_180` if your camera is mounted upside down.
3. Arduino IDE: board *AI Thinker ESP32-CAM*, **PSRAM enabled**. Flashing needs a USB-serial adapter with GPIO0 connected to GND during the upload (remove it afterwards).
4. At start-up the firmware prints how much of the tensor arena it uses, and where it was allocated (internal RAM or PSRAM).

The firmware sends the predicted class (0 to 3) as one byte on UART1 (GPIO14 on the AI-Thinker board) to the Arduino Mega.

### 8. Capture images and evaluate live (optional)

Close the Arduino Serial Monitor first, then:

```bash
python camera_capture.py --port COM3 --out_dir ./data_own --subset val
```

Keys: `1`-`4` select the class, `s` saves a frame, `SPACE` saves bursts, `e` starts/stops the live evaluation (accuracy measured on the board), `q` quits.

### 9. Run the rover

1. Upload `rover_main.ino` to the Arduino Mega (TinyGPSPlus installed). Start with `DRY_RUN` set to 1: the motors do not move and the sketch prints what it would do. Set it to 0 only with the wheels off the ground for the first tests.
2. Connect from a phone with a BLE terminal app to the JDY-09 module (9600 baud) and send text lines:

| Command | Effect |
|---|---|
| `lat,lon` (decimal degrees, dot as separator) | sets the destination and starts the navigation |
| `STOP` | stops the rover |
| `STATUS` | prints position, heading, speeds and distance to the target |
| `CAL` | recalibrates the gyroscope (the rover must be still) |
| `ENC`, `RESETODO` | print / reset the encoder counters |
| `DRY_RUN`, `PRINT_EVERY`, `NMEA` | toggle dry run, printing of every camera result, printing of the raw GPS sentences |
| `LIST` | lists the commands |

Pin assignment of the serial links: camera board TX → Mega RX1 (pin 19); GPS TX → RX2 (pin 17); Bluetooth RX3 / TX3 (pins 15 / 14, through a voltage divider on the Mega TX line); GY-521 on SDA 20 / SCL 21; the other pins (ultrasonic sensors, motors, encoders) are listed in the `#define` block at the top of `rover_main.ino`. Wiring diagrams are in the thesis (Chapter 5).

## Methodology

- **Data.** About 34 thousand $96 \times 96$ grayscale images in four classes, from NYU Depth V2 (labelled from depth: the 5th percentile of the distance in the left/centre/right thirds of the lower 60% of the frame, blocked below 1.0 m), COCO 2017 (labelled from the size, position and overlap of obstacle-class bounding boxes; street scenes discarded) and images captured with the ESP32-CAM itself. Splits avoid leakage: NYU by position in the file, COCO by image identifier, own images by recording session.
- **Model.** A ResNet-style classifier built for the device: a strided stem convolution and three residual blocks of 8, 16 and 32 channels, global average pooling and a 4-way softmax. Trained with the Adam optimizer, cross-entropy loss, brightness / contrast / rotation augmentation and left-right mirroring.
- **Quantization and deployment.** Post-training full 8-bit quantization, converted to a C array and run with TensorFlow Lite Micro on the ESP32-CAM; the tensor arena (100 KB) is allocated at run time.
- **Rover.** The Arduino Mega fuses the sensors with a strict priority: ultrasonic safety first, then the camera (advisory), then the GPS goal. The heading comes from a gyroscope corrected by the GPS course; wheel encoders provide speed, stall detection and straight-line trim.

## Results

The full results, with the confusion matrix and the discussion, are presented in the thesis (Chapters 4, 6 and 7).

The classifier is a weak sensor, which is why the rover gives priority to the ultrasonic sensors. Field tests were short (a few tens of metres) and are described, with their limitations, in the thesis.

## Data Availability

All the datasets used in this thesis are available for free under their own licences (check each one) and must be cited if used:

- **COCO 2017**: annotations under CC-BY-4.0; the images are subject to the Flickr terms of use. <https://cocodataset.org>
- **NYU Depth V2**: N. Silberman, P. Kohli, D. Hoiem, R. Fergus. <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>

`data.zip` contains the images converted to $96 \times 96$ grayscale.

## Reproducibility

- **Python version:** Python 3.11.
- **Package versions:** the latest available at the time of the writing, see `requirements.txt` (run `pip freeze` in the environment used for the thesis).
- **Random seeds:** the dataset scripts accept `--seed` (default 0).
- **Experimental configurations:** the thresholds used for labelling (1.0 m depth threshold, 4% minimum box area, 25% zone overlap), the training parameters of step 4 and the quantization settings (data/val calibration images) are described in the thesis.
- **Hardware used for training:** AMD Ryzen 7 7735HS laptop, CPU only (TensorFlow does not use the GPU on native Windows); about 15 to 20 minutes of training.
- **Hardware of the rover:** ESP32-CAM (AI-Thinker, OV2640), Arduino Mega 2560, GPS module marked NEO-6M-0-001 with ceramic antenna, GY-521 (MPU-6050), 3 × HC-SR04, 2 reflective IR sensors with 20-slot discs, JDY-09 Bluetooth Low Energy module, Osoyoo motor driver, 4x DC gear motors, 2 × 18650 cells.

## Citation

```bibtex
@thesis{agosti2026tinyml,
  author = {Tommaso Agosti},
  title  = {TinyML in Practice: Developing and Deploying an Image Recognition Model on custom Embedded Hardware},
  school = {University of Trento},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```

## License

See the `LICENSE` file. The public datasets keep their own licences.