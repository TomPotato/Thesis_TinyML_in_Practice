/*
  ESP32-CAM obstacle classifier firmware.

  Captures a grayscale frame, runs the quantized tiny-ResNet, and sends a
  single byte over UART to the Arduino Mega:
      0 = clear
      1 = left_blocked
      2 = center_blocked
      3 = right_blocked

  Libraries required (Arduino IDE > Library Manager):
    - "esp32" board package (provides esp_camera.h for ESP-EYE)
    - "TensorFlowLite_ESP32" (or the newer "Arduino_TensorFlowLite" /
      tflite-micro Arduino port)

  Board: AI-Thinker ESP32-CAM camera pin definitions below are the standard

  Place model.h in the same sketch folder before compiling.
*/

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "model.h"

// The "TensorFlowLite_ESP32" Arduino library wants this header included first.
#if __has_include(<TensorFlowLite_ESP32.h>)
#include <TensorFlowLite_ESP32.h>
#endif

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// TFLite Micro API differs between library versions:
//  - older versions ship micro_error_reporter.h and require an ErrorReporter
//    in the MicroInterpreter constructor;
//  - newer versions removed it entirely.
// This sketch detects which one is installed and adapts.
#if __has_include(<tensorflow/lite/micro/micro_error_reporter.h>)
#include <tensorflow/lite/micro/micro_error_reporter.h>
#define TFLM_HAS_ERROR_REPORTER 1
#else
#define TFLM_HAS_ERROR_REPORTER 0
#endif

#if __has_include(<tensorflow/lite/version.h>)
#include <tensorflow/lite/version.h>
#endif
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

// ---------- Config ----------
#define IMG_SIZE 96
#define NUM_CLASSES 4

// Standalone testing (no Mega attached): results always print on USB Serial.
#define SEND_TO_MEGA   1     // set to 0 to disable the UART output entirely
#define DEBUG_SCORES   1     // print all four class scores for every frame
#define LOOP_DELAY_MS  0   // slows the printout so it is readable (0 = full speed)
// Send the character 'a' in the Serial Monitor to print an ASCII preview of
// the next frame (use it to check the image is not flipped or mirrored).
//
// Sending frames to the PC (use esp_eye_capture.py):
//   'v' = start streaming frames    'x' = stop streaming    'f' = send one frame
// Frames are binary packets, so the Serial Monitor shows garbage while
// streaming: close it and use the PC script instead.
// A faster baud rate is needed for streaming (9216 bytes per frame).
// The Serial Monitor / PC script must use the same value.
#define SERIAL_BAUD    921600

// BOARD_AI_THINKER_ESP32CAM

// Image orientation. 1 = rotate 180 degrees (vertical flip + horizontal mirror
// on the sensor). Training images are upright, so the live image must be too.
#define CAMERA_ROTATE_180  0

// UART to the Mega, on UART1 (NOT the USB/programming UART0).
// Wiring: board TX pin -> Mega RX1 (pin 19), plus a common GND.
// The link is one-way (board -> Mega), so the Mega's 5V TX does NOT need to be
// wired to the board (ESP32 pins are not 5V tolerant).
HardwareSerial MegaLink(1);  // UART1
#define MEGA_BAUD 115200

// AI-Thinker ESP32-CAM. Free GPIOs with the camera running: 12 (strapping,
// avoid), 13, 14, 15. GPIO4 is the flash LED, 16 is PSRAM, 0/1/3 are for
// flashing and the USB-serial adapter.
#define MEGA_TX_PIN 14   // -> Mega RX1 (pin 19)
#define MEGA_RX_PIN 13   // left unconnected

#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ---------- TFLite Micro globals ----------
namespace {
#if TFLM_HAS_ERROR_REPORTER
tflite::ErrorReporter* error_reporter = nullptr;
#endif
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena: scratch RAM for TFLM activations. Start here; if you see
// an "arena too small" error at runtime, increase this and re-flash.
// Allocated at runtime (see setup_model), NOT as a static array: static RAM on
// the ESP32 is limited to ~124 KB, which a 100 KB array plus the libraries
// overflows at link time. The heap has far more room.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t* tensor_arena = nullptr;
}  // namespace

bool setup_camera() {
  camera_config_t config = {};  // zero-init: unset fields must not be garbage
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;  // match training preprocessing
  config.frame_size = FRAMESIZE_96X96;        // directly matches model input
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

#if CAMERA_ROTATE_180
  // Vertical flip + horizontal mirror together = 180 degree rotation.
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_vflip(sensor, 1);
    sensor->set_hmirror(sensor, 1);
  }
#endif
  return true;
}

bool setup_model() {
#if TFLM_HAS_ERROR_REPORTER
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
#endif

  // Prefer fast internal RAM; fall back to PSRAM (slower, but plentiful).
  // PSRAM needs "PSRAM: Enabled" in the Arduino Tools menu.
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (tensor_arena) {
    Serial.println("Tensor arena allocated in internal RAM.");
  } else {
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (tensor_arena) {
      Serial.println("Tensor arena allocated in PSRAM (slower inference).");
    } else {
      Serial.println("Could not allocate the tensor arena. Enable PSRAM or lower kTensorArenaSize.");
      return false;
    }
  }

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return false;
  }

  // Only register the ops this architecture actually uses -- keeps the
  // runtime binary and RAM footprint smaller than AllOpsResolver.
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddAdd();
  resolver.AddRelu();
  resolver.AddMean();     // used by GlobalAveragePooling2D
  resolver.AddReshape();

#if TFLM_HAS_ERROR_REPORTER
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
#else
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
#endif
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed (try a larger kTensorArenaSize, or a missing op in the resolver)");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("Tensor arena used: %u of %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
  return true;
}

// Copies a captured grayscale frame buffer into the quantized input tensor.
// Assumes the camera frame is already 96x96 (set via FRAMESIZE_96X96 above)
// and applies the same int8 scale/zero-point used during TFLite conversion.
void fill_input_tensor(camera_fb_t* fb) {
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    float pixel_float = fb->buf[i] / 255.0f;  // match training normalization
    int q = static_cast<int>(roundf(pixel_float / scale + zero_point));
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    input->data.int8[i] = static_cast<int8_t>(q);
  }
}

int run_inference() {
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke() failed");
    return -1;
  }

  // Dequantize output just enough to argmax -- no need for float softmax values.
  int best_class = 0;
  int8_t best_score = output->data.int8[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_class = i;
    }
  }
  return best_class;
}

// Copy of the frame the model just classified, kept for sending to the PC.
static uint8_t last_frame[IMG_SIZE * IMG_SIZE];
static bool streaming = false;
static bool send_one = false;

// Binary packet sent to the PC:
//   AA 55 A5 5A | class (1) | 4 scores 0-255 (4) | 96*96 pixels | checksum (1)
// The checksum is the sum of all pixel bytes modulo 256.
void send_frame_packet(int cls) {
  uint8_t header[9] = {0xAA, 0x55, 0xA5, 0x5A, (uint8_t)cls, 0, 0, 0, 0};
  float scale = output->params.scale;
  int zp = output->params.zero_point;
  for (int i = 0; i < NUM_CLASSES; i++) {
    float sc = (output->data.int8[i] - zp) * scale;
    if (sc < 0.0f) sc = 0.0f;
    if (sc > 1.0f) sc = 1.0f;
    header[5 + i] = (uint8_t)(sc * 255.0f + 0.5f);
  }
  uint8_t sum = 0;
  for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) sum += last_frame[i];

  Serial.write(header, sizeof(header));
  Serial.write(last_frame, IMG_SIZE * IMG_SIZE);
  Serial.write(&sum, 1);
}

void print_scores() {
  const char* names[NUM_CLASSES] = {"clear", "left", "center", "right"};
  float scale = output->params.scale;
  int zp = output->params.zero_point;
  Serial.print("scores:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.printf(" %s=%.2f", names[i], (output->data.int8[i] - zp) * scale);
  }
  Serial.println();
}

// Prints the 96x96 frame as 48x24 ASCII art (top of the image first).
void print_ascii_preview(camera_fb_t* fb) {
  const char ramp[] = " .:-=+*#%@";  // dark -> bright
  Serial.println("--- frame preview (what the model sees) ---");
  for (int y = 0; y < IMG_SIZE; y += 4) {
    for (int x = 0; x < IMG_SIZE; x += 2) {
      int sum = 0;
      for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 2; dx++)
          sum += fb->buf[(y + dy) * IMG_SIZE + (x + dx)];
      Serial.write(ramp[(sum / 8) * 9 / 255]);
    }
    Serial.println();
  }
  Serial.println("--- end preview ---");
}

void setup() {
  Serial.begin(SERIAL_BAUD);  // USB: debug console and frame streaming
  MegaLink.begin(MEGA_BAUD, SERIAL_8N1, MEGA_RX_PIN, MEGA_TX_PIN);

  if (!setup_camera()) {
    Serial.println("Camera setup failed, halting.");
    while (true) delay(1000);
  }
  if (!setup_model()) {
    Serial.println("Model setup failed, halting.");
    while (true) delay(1000);
  }
  Serial.println("ESP-EYE obstacle classifier ready.");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed, skipping.");
    delay(50);
    return;
  }

  if (fb->width != IMG_SIZE || fb->height != IMG_SIZE) {
    Serial.printf("Unexpected frame size %dx%d (expected %dx%d), skipping.\n",
                  fb->width, fb->height, IMG_SIZE, IMG_SIZE);
    esp_camera_fb_return(fb);
    delay(50);
    return;
  }

  bool want_preview = false;
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'a') want_preview = true;
    else if (c == 'v') streaming = true;
    else if (c == 'x') streaming = false;
    else if (c == 'f') send_one = true;
  }
  if (want_preview && !streaming) print_ascii_preview(fb);

  memcpy(last_frame, fb->buf, IMG_SIZE * IMG_SIZE);
  fill_input_tensor(fb);
  esp_camera_fb_return(fb);

  unsigned long t0 = millis();
  int result_class = run_inference();
  unsigned long inference_ms = millis() - t0;

  if (result_class >= 0) {
#if SEND_TO_MEGA
    MegaLink.write(static_cast<uint8_t>(result_class));
#endif
    if (streaming || send_one) {
      send_frame_packet(result_class);
      send_one = false;
    }
    if (!streaming) {  // text logs would clutter the binary stream
      const char* names[NUM_CLASSES] = {"clear", "left_blocked", "center_blocked", "right_blocked"};
      Serial.printf("class=%d (%s)  inference=%lums\n", result_class, names[result_class], inference_ms);
#if DEBUG_SCORES
      print_scores();
#endif
    }
  }
  if (!streaming && LOOP_DELAY_MS > 0) delay(LOOP_DELAY_MS);

  // No fixed delay needed -- inference time itself paces the loop. Add a
  // small delay here only if you want to deliberately cap the frame rate.
}
