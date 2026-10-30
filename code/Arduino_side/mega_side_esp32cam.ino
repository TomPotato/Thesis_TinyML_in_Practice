/*
  Rover main sketch (Arduino Mega).

  Send a destination over Bluetooth and the rover drives to it:
      45.890123,11.040123   -> set destination (decimal degrees, DOT as decimal
                               separator) and start driving
      STOP                  -> stop the rover
      STATUS                -> reply with position / distance to target

  Priority of the inputs (highest first), same as the earlier design:
      1. Ultrasonic sensors : anything closer than TOO_CLOSE_CM -> turn away
      2. Camera result      : obstacle seen left / center / right -> steer away
      3. GPS                : steer toward the destination

  Wiring
      Camera board TX      -> Mega RX1 (pin 19)   + common GND   [115200 baud]
      GPS TX               -> Mega RX2 (pin 17)                   [9600 baud]
      Bluetooth TXD        -> Mega RX3 (pin 15)                   [BT_BAUD below]
      Bluetooth RXD        <- Mega TX3 (pin 14) through a divider: Mega TX3 ->
                              1 kOhm -> BT RXD, and 2 kOhm from BT RXD to GND
                              (BT RXD is 3.3 V, the Mega is 5 V)
      Ultrasonic (HC-SR04) : LEFT  TRIG 24 / ECHO 22
                             CENTER TRIG 28 / ECHO 26
                             RIGHT  TRIG 32 / ECHO 30
      Motors               : direction pins 40-43, speed (PWM) pins 6 and 5.
                             The two SPEED pins MUST be PWM pins (Mega: 2-13,
                             44-46); pins 38-43 cannot do PWM.
      Wheel encoders       : left sensor OUT -> pin 2, right sensor OUT -> pin 3
                             (+5 V and GND from the Mega). Disc on the motor
                             shaft, one pulse per slot.

  ENCODERS: set GEAR_RATIO and WHEEL_DIAMETER_MM below (the defaults are only
  typical values). To calibrate by hand: send RESETODO, push the rover exactly
  1.00 m, send ENC, then set METERS_PER_PULSE_OVERRIDE = 1.0 / (average of the
  two pulse counts). The encoders measure distance and speed, detect a stalled
  wheel, and keep the rover driving straight while it searches for its heading.
  They cannot tell forward from reverse: the direction comes from the command.

  Classic Bluetooth modules (HC-05 / HC-06 / JDY-09) work with Android serial-terminal
  apps. iPhones need a BLE module (e.g. HM-10) and a BLE terminal app.

  HEADING (GY-521 gyro + GPS course)
      GY-521: SDA -> Mega pin 20, SCL -> Mega pin 21, VCC -> 5 V or 3.3 V (the
      board has a regulator), GND, AD0 -> GND (I2C address 0x68). Mount it FLAT
      (components facing up), fixed firmly to the frame, away from the motors.
  The gyro measures how fast the rover turns and is integrated into a heading.
  It has no compass, so it does not know where north is: the first time the
  rover moves straight at MIN_COURSE_SPEED_MPS or more, the GPS "course over
  ground" gives it north, and every later GPS course keeps correcting the
  gyro drift. Until that first alignment the rover drives straight to get one.
  Without a working gyro the sketch falls back to GPS course only.
  Check the sign once: with DRY_RUN 1, turn the rover clockwise by hand; the
  "heading" in the status line must INCREASE. If it decreases, flip GYRO_SIGN.
  Keep the rover STILL during the startup calibration. Send CAL to recalibrate.

  !! DRY_RUN is 1 by default: the motors are NOT driven, the sketch only prints
  what it would do. Test everything that way first, then set it to 0, and run
  the first real tests with the wheels off the ground.

  Serial Monitor: 115200 baud.
*/

#include <TinyGPS++.h>
#include <Wire.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ------------------------------ settings ------------------------------
int DRY_RUN =            1;     // 1 = do not drive the motors, just print
#define BT_BAUD        9600     // Bluetooth module speed (find it with bt_baud_finder.ino)
int PRINT_EVERY_RESULT = 0;     // 1 = print every camera result
int RAW_NMEA =           0;     // 1 = copy the GPS sentences to the monitor

#define USE_CAMERA        1    // 0 = ignore the camera (GPS + ultrasonic only)
#define US_CONFIRM_SCANS  2    // a sensor must read "too close" this many scans in a row
int closeCount[3] = {0, 0, 0};
const char* navMode = "idle";
double navErr = 0.0;

#define TOO_CLOSE_CM         40
#define ECHO_TIMEOUT_US      12000UL    // about 2 m; keeps each scan short
#define SCAN_INTERVAL_MS     120

#define ARRIVE_RADIUS_M      3.0        // GPS is only accurate to a few meters
#define MAX_TARGET_DIST_M    50.0     // reject targets farther than this (typos)
#define BASE_SPEED           180        // 0-255
#define TURN_SPEED           120
#define TURN_GAIN            1.6        // PWM units per degree of heading error
#define MIN_COURSE_SPEED_MPS 0.6

#define USE_ENCODERS         1
#define ENC_L_PIN            2       // interrupt pins on the Mega: 2, 3, 18, 19, 20, 21
#define ENC_R_PIN            3
#define ENC_SLOTS            20      // slots on the disc = pulses per motor-shaft turn
#define GEAR_RATIO           48.0    // motor-shaft turns per wheel turn
#define WHEEL_DIAMETER_MM    72.0    
#define METERS_PER_PULSE_OVERRIDE 0.0   // > 0: use this (from the hand calibration) instead
#define ENC_MIN_PULSE_US     100     // ignore pulses closer together than this (noise)
#define ENC_UPDATE_MS        200
#define STALL_MIN_PWM        100     // only check for a stall when driving at least this hard
#define STALL_MS             1500    // motor on this long with no pulses = stalled
#define TRIM_GAIN            15.0    // straight-line correction strength
#define TRIM_MAX             40.0

#define USE_GYRO             1       // 1 = use the GY-521 (MPU-6050) for heading
#define GYRO_SIGN            +1      // -1: board flat, components up. +1: upside down.
#define GYRO_LSB_PER_DPS     65.5    // sensitivity at +-500 deg/s
#define GYRO_DEADBAND_DPS    0.15    // ignore tiny rates (noise) when standing still
#define GYRO_CAL_SAMPLES     500
#define GYRO_CAL_MAX_SPREAD  100     // max raw spread allowed during calibration (= rover still)
#define FUSION_ALPHA         0.8     // share of each GPS-course error corrected per update
#define MAX_FUSION_RATE_DPS  120.0   // GPS course lags while turning: skip corrections then
#define PIVOT_ABOVE_DEG      15.0    // heading error above this: turn on the spot first

#define GPS_FRESH_MS         1000
#define CAMERA_FRESH_MS      1000
#define CAMERA_TIMEOUT_MS    2000

// Motor driver pins
#define dir1PinL  43   // left side forward IN1
#define dir2PinL  41   // left side backwards IN2
#define speedPinL 5    // PWM pin required ENB
#define dir1PinR  40   // right side forward IN3
#define dir2PinR  42   // right side backwards IN4
#define speedPinR 6    // PWM pin required ENA

// ------------------------------ state ---------------------------------
TinyGPSPlus gps;

const char* CLASS_NAMES[] = {"clear", "left_blocked", "center_blocked", "right_blocked"};

const int trigPIN[] = {24, 28, 32};   // left, center, right
const int echoPIN[] = {22, 24, 30};
long distCm[3] = {-1, -1, -1};        // -1 = no echo (nothing within ~2 m)

int camClass = 0;
unsigned long camTime = 0;            // 0 = no camera result received yet
int lastClass = -1;
unsigned long resultCount = 0;
unsigned long lastByteTime = 0;
bool camWarned = false;

bool navigating = false;
double targetLat = 0.0, targetLon = 0.0;
double navDist = 0.0, navBearing = 0.0, navHeading = 0.0;
bool navHeadingValid = false;

char btBuf[48];
byte btLen = 0;

unsigned long lastScan = 0, lastStatus = 0, lastBtStatus = 0;

// Wheel encoders
const double M_PER_PULSE = (METERS_PER_PULSE_OVERRIDE > 0.0)
    ? METERS_PER_PULSE_OVERRIDE
    : (3.14159265 * (WHEEL_DIAMETER_MM / 1000.0) / (ENC_SLOTS * GEAR_RATIO));

volatile long encCountL = 0, encCountR = 0;        // signed pulse counts
volatile unsigned long lastPulseUsL = 0, lastPulseUsR = 0;
volatile int8_t dirL = 1, dirR = 1;                // last commanded direction (+1 / -1)
int cmdL = 0, cmdR = 0;                            // last commanded speed, -255..255
double speedL = 0.0, speedR = 0.0;                 // measured, m/s (signed)
long prevCountL = 0, prevCountR = 0;
unsigned long lastEncUpdateMs = 0;
unsigned long noPulseSinceL = 0, noPulseSinceR = 0;
bool everSeenL = false, everSeenR = false;
bool warnedNoEncL = false, warnedNoEncR = false;
double straightTrim = 0.0;                         // + = left side was too fast
bool straightActive = false;
unsigned long straightLastMs = 0;

// ------------------------------ motors --------------------------------
// left/right: -255..255, positive = forward.
void drive(int left, int right) {
  left  = constrain(left,  -255, 255);
  right = constrain(right, -255, 255);

  // Remember the command: the encoders take their direction from it, and the
  // stall check needs to know when the motors started to be driven.
  if (left  > 0) dirL = 1; else if (left  < 0) dirL = -1;
  if (right > 0) dirR = 1; else if (right < 0) dirR = -1;
  if (abs(left)  >= STALL_MIN_PWM && abs(cmdL) < STALL_MIN_PWM) noPulseSinceL = millis();
  if (abs(right) >= STALL_MIN_PWM && abs(cmdR) < STALL_MIN_PWM) noPulseSinceR = millis();
  cmdL = left;
  cmdR = right;
  if(DRY_RUN){
    static unsigned long lastShown = 0;
    if (millis() - lastShown > 500) {
      lastShown = millis();
      Serial.print("[dry run] left=");
      Serial.print(left);
      Serial.print(" right=");
      Serial.println(right);
    }
    return;
  }
  else{
    digitalWrite(dir1PinL, (left  > 0)? HIGH : LOW);
    digitalWrite(dir2PinL, (left  < 0)? HIGH : LOW);
    digitalWrite(dir1PinR, (right > 0)? HIGH : LOW);
    digitalWrite(dir2PinR, (right < 0)? HIGH : LOW);
    analogWrite(speedPinL, abs(left));
    analogWrite(speedPinR, abs(right));
  }
}

void Stop() { drive(0, 0); }

// ------------------------------ sensors -------------------------------
// Distance in cm, or -1 if no echo came back within ECHO_TIMEOUT_US.
long readDistanceCm(int sensor) {
  digitalWrite(trigPIN[sensor], LOW);
  delayMicroseconds(2);
  digitalWrite(trigPIN[sensor], HIGH);
  delayMicroseconds(5);
  digitalWrite(trigPIN[sensor], LOW);
  unsigned long duration = pulseIn(echoPIN[sensor], HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) return -1;
  return duration / 58;
}

long nearOrFar(long d) { return d < 0 ? 999 : d; }   // "no echo" = far away

void scanUltrasonics() {
  if (millis() - lastScan < SCAN_INTERVAL_MS) return;
  lastScan = millis();
  for (int i = 0; i < 3; i++) {
    distCm[i] = readDistanceCm(i);
    closeCount[i] = (distCm[i] >= 0 && distCm[i] < TOO_CLOSE_CM) ? closeCount[i] + 1 : 0;
    delay(5);          // let this sensor's echo die out before the next one fires
  }
}

void readCamera() {
  while (Serial1.available() > 0) {
    int c = Serial1.read();
    lastByteTime = millis();
    camWarned = false;
    if (c < 0 || c > 3) {
      Serial.print("Unexpected byte: ");
      Serial.println(c);
      continue;
    }
    camClass = c;
    camTime = millis();
    resultCount++;
    if (PRINT_EVERY_RESULT || c != lastClass) {
      Serial.print("[");
      Serial.print(resultCount);
      Serial.print("] camera: ");
      Serial.println(CLASS_NAMES[c]);
    }
    lastClass = c;
  }
  if (!camWarned && millis() - lastByteTime > CAMERA_TIMEOUT_MS) {
    Serial.println("No data from the camera board for 2 s (check TX -> pin 19 and GND).");
    camWarned = true;
  }
}

void readGps() {
  while (Serial2.available() > 0) {     // feed EVERY byte to the parser
    char ch = Serial2.read();
 if(RAW_NMEA) Serial.write(ch);
    gps.encode(ch);
  }
}

bool haveFix() {
  return gps.location.isValid() && gps.location.age() < GPS_FRESH_MS;
}

// ------------------------------ wheel encoders ------------------------
void isrLeft() {
  unsigned long now = micros();
  if (now - lastPulseUsL < ENC_MIN_PULSE_US) return;   // noise
  lastPulseUsL = now;
  encCountL += dirL;
}

void isrRight() {
  unsigned long now = micros();
  if (now - lastPulseUsR < ENC_MIN_PULSE_US) return;
  lastPulseUsR = now;
  encCountR += dirR;
}

double odometerMeters() {
  noInterrupts();
  long cL = encCountL, cR = encCountR;
  interrupts();
  return 0.5 * (cL + cR) * M_PER_PULSE;
}

void printEnc(Print &out) {
  noInterrupts();
  long cL = encCountL, cR = encCountR;
  interrupts();
  out.print("encoders: L ");
  out.print(cL);
  out.print(" pulses, R ");
  out.print(cR);
  out.print(" pulses, odometer ");
  out.print(0.5 * (cL + cR) * M_PER_PULSE, 3);
  out.print(" m (");
  out.print(M_PER_PULSE * 1000.0, 4);
  out.println(" mm per pulse)");
}

void say3(const char* a, const char* b, const char* c) {
  Serial.print(a);  Serial.print(b);  Serial.println(c);
  Serial3.print(a); Serial3.print(b); Serial3.println(c);
}

// True if this side has been driven hard for STALL_MS without a single pulse.
// If its sensor has never produced a pulse, only warn: it is probably not wired.
bool sideStalled(const char* name, int cmd, unsigned long noPulseSince, bool everSeen,
                 bool &warned, unsigned long now) {
  if (abs(cmd) < STALL_MIN_PWM) return false;
  if (now - noPulseSince <= STALL_MS) return false;
  if (everSeen) return true;
  if (!warned) {
    warned = true;
    say3("Warning: no encoder pulses from the ", name,
         " side while driving. Check the sensor wiring and pin (stall detection stays off until a pulse is seen).");
  }
  return false;
}

void updateEncoders() {
#if USE_ENCODERS
  unsigned long now = millis();
  if (now - lastEncUpdateMs < ENC_UPDATE_MS) return;
  double dt = (now - lastEncUpdateMs) / 1000.0;
  lastEncUpdateMs = now;

  noInterrupts();
  long cL = encCountL, cR = encCountR;
  interrupts();
  long dL = cL - prevCountL, dR = cR - prevCountR;
  prevCountL = cL;
  prevCountR = cR;
  speedL = dL * M_PER_PULSE / dt;
  speedR = dR * M_PER_PULSE / dt;
  if (dL != 0) { noPulseSinceL = now; everSeenL = true; }
  if (dR != 0) { noPulseSinceR = now; everSeenR = true; }

  // Straight-line trim: if the left side runs faster than the right, slow it
  // down and speed the right one up (and the other way round).
  if (straightActive && now - straightLastMs < 300 && speedL > 0.03 && speedR > 0.03) {
    double err = (speedL - speedR) / (0.5 * (speedL + speedR));
    straightTrim = constrain(straightTrim + TRIM_GAIN * err, -TRIM_MAX, TRIM_MAX);
  }

  if(!DRY_RUN) {
    bool stallL = sideStalled("left", cmdL, noPulseSinceL, everSeenL, warnedNoEncL, now);
    bool stallR = sideStalled("right", cmdR, noPulseSinceR, everSeenR, warnedNoEncR, now);
    if (stallL || stallR) {
      navigating = false;
      Stop();
      say3("Stopped: the ", stallL ? "left" : "right",
          " wheel is stalled (motor on, no encoder pulses). Check for a jam or an obstacle, then send the target again.");
    }
  }
#endif
}

// Drive straight ahead, using the encoders to even out the two sides.
void driveStraight(int speed) {
#if USE_ENCODERS
  straightActive = true;
  straightLastMs = millis();
  drive(speed - (int)straightTrim, speed + (int)straightTrim);
#else
  drive(speed, speed);
#endif
}

// ------------------------------ heading (gyro + GPS) ------------------
// Angle helpers. angleDiff: smallest signed angle from `current` to `target`,
// in [-180, 180]; positive = the target is to the right (clockwise).
double wrap360(double a) {
  a = fmod(a, 360.0);
  if (a < 0.0) a += 360.0;
  return a;
}

double angleDiff(double target, double current) {
  return fmod(target - current + 540.0, 360.0) - 180.0;
}

bool gyroOk = false;
uint8_t mpuAddr = 0x68;
double gyroBias = 0.0;          // raw counts at standstill
double gyroRate = 0.0;          // deg/s, positive = clockwise
double gyroYaw = 0.0;           // integrated turning in degrees (relative), 0..360
double headingOffset = 0.0;     // gyroYaw + headingOffset = heading relative to north
bool headingKnown = false;      // true once the GPS course has aligned the gyro to north
unsigned long lastGyroUs = 0;

bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool readGyroZ(int16_t &raw) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(0x47);                                  // GYRO_ZOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)mpuAddr, 2) != 2) return false;
  int hi = Wire.read();
  int lo = Wire.read();
  raw = (int16_t)((hi << 8) | lo);
  return true;
}

bool initGyro() {
  for (uint8_t addr = 0x68; addr <= 0x69; addr++) {  // AD0 low = 0x68, high = 0x69
    mpuAddr = addr;
    if (mpuWrite(0x6B, 0x00)) {                      // wake the chip up
      delay(100);
      mpuWrite(0x1A, 0x03);                          // low-pass filter ~44 Hz (vibration)
      mpuWrite(0x1B, 0x08);                          // +-500 deg/s full scale
      return true;
    }
  }
  return false;
}

// Measures the gyro's zero offset. The rover must not move meanwhile.
bool calibrateGyro() {
  for (int attempt = 0; attempt < 3; attempt++) {
    double sum = 0.0;
    int16_t mn = 32767, mx = -32768;
    for (int i = 0; i < GYRO_CAL_SAMPLES; i++) {
      int16_t raw;
      if (!readGyroZ(raw)) return false;
      sum += raw;
      if (raw < mn) mn = raw;
      if (raw > mx) mx = raw;
      delay(3);
    }
    gyroBias = sum / GYRO_CAL_SAMPLES;
    lastGyroUs = micros();
    gyroRate = 0.0;
    if ((mx - mn) <= GYRO_CAL_MAX_SPREAD) return true;
    Serial.println("Gyro calibration: the rover moved, retrying...");
  }
  Serial.println("Gyro calibration: still moving after 3 tries, heading may drift. Send CAL when the rover is still.");
  return true;
}

void updateGyro() {
  if (!gyroOk) return;
  unsigned long nowUs = micros();
  double dt = (nowUs - lastGyroUs) * 1e-6;
  lastGyroUs = nowUs;
  int16_t raw;
  if (!readGyroZ(raw)) return;
  double rate = GYRO_SIGN * (raw - gyroBias) / GYRO_LSB_PER_DPS;
  if (fabs(rate) < GYRO_DEADBAND_DPS) rate = 0.0;
  if (dt > 0.0 && dt < 0.2) {
    gyroYaw = wrap360(gyroYaw + 0.5 * (rate + gyroRate) * dt);   // trapezoid integration
  }
  gyroRate = rate;
}

// Uses each new GPS course (valid only while moving) to give the gyro its
// north reference, and to pull back the slow gyro drift.
void fuseGpsCourse() {
  if (!gyroOk) return;
  if (!gps.course.isUpdated()) return;
  double course = gps.course.deg();                  // reading it clears the "updated" flag
  if (!gps.course.isValid() || !gps.speed.isValid()) return;
  if (gps.speed.mps() < MIN_COURSE_SPEED_MPS) return;
  if (fabs(gyroRate) > MAX_FUSION_RATE_DPS) return;  // GPS course lags during turns

  if (!headingKnown) {
    headingOffset = wrap360(course - gyroYaw);
    headingKnown = true;
    return;
  }
  double fused = wrap360(gyroYaw + headingOffset);
  headingOffset = wrap360(headingOffset + FUSION_ALPHA * angleDiff(course, fused));
}

// Heading in degrees (0 = north, 90 = east, clockwise). Returns false while unknown.
bool currentHeading(double &h) {
  if (gyroOk) {
    if (!headingKnown) return false;
    h = wrap360(gyroYaw + headingOffset);
    return true;
  }
  // No gyro: GPS course only, valid while moving.
  if (gps.course.isValid() && gps.course.age() < 2000 &&
      gps.speed.isValid() && gps.speed.mps() >= MIN_COURSE_SPEED_MPS) {
    h = gps.course.deg();
    return true;
  }
  return false;
}

// ------------------------------ navigation ----------------------------

void say(const char* msg) {
  Serial.println(msg);
  Serial3.println(msg);
}

void printStatus(Print &out) {
  if (gps.location.isValid()) {
    out.print("pos ");
    out.print(gps.location.lat(), 6);
    out.print(",");
    out.print(gps.location.lng(), 6);
    out.print(" sats ");
    out.print(gps.satellites.value());
  } else {
    out.print("no GPS fix (chars ");
    out.print(gps.charsProcessed());
    out.print(", valid sentences ");
    out.print(gps.passedChecksum());
    out.print(")");
  }
  double h;
  if (currentHeading(h)) {
    out.print(" | heading ");
    out.print(h, 0);
  } else {
    out.print(gyroOk ? " | heading unknown (drive straight once to align)" : " | heading unknown");
  }
#if USE_ENCODERS
  out.print(" | L ");
  out.print(speedL, 2);
  out.print(" R ");
  out.print(speedR, 2);
  out.print(" m/s, odo ");
  out.print(odometerMeters(), 2);
  out.print(" m");
#endif
  if (navigating) {
    out.print(" | to target: ");
    out.print(navDist, 1);
    out.print(" m, bearing ");
    out.print(navBearing, 0);
  } else {
    out.print(" | idle");
  }
  out.println();
}

void printNavDebug() {
  Serial.print("   mode: ");           Serial.print(navMode);
  Serial.print(" | us L/C/R: ");       Serial.print(distCm[0]); Serial.print("/");
                                       Serial.print(distCm[1]); Serial.print("/"); Serial.print(distCm[2]);
  Serial.print(" | cam: ");            Serial.print(camClass);
  Serial.print(" | gps speed: ");      Serial.print(gps.speed.mps(), 2);
  Serial.print(" m/s | heading error: "); Serial.println(navErr, 0);
}

void navigate() {
  if (!navigating) { navMode = "idle"; Stop(); return; }

  long l = nearOrFar(distCm[0]);
  long c = nearOrFar(distCm[1]);
  long r = nearOrFar(distCm[2]);

  // ---- Priority 1: ultrasonic safety ----
  bool tooL = closeCount[0] >= US_CONFIRM_SCANS;
  bool tooC = closeCount[1] >= US_CONFIRM_SCANS;
  bool tooR = closeCount[2] >= US_CONFIRM_SCANS;
  if (tooL || tooC || tooR) {
    if (tooL && tooC && tooR) { navMode = "ULTRASONIC: all three close, STOPPED"; Stop(); return; }
    if (tooC) {
      navMode = "ULTRASONIC: centre close, pivoting";
      if (l >= r) drive(-TURN_SPEED, TURN_SPEED);
      else        drive(TURN_SPEED, -TURN_SPEED);
    } else if (tooL) {
      navMode = "ULTRASONIC: left close, pivot right";
      drive(TURN_SPEED, -TURN_SPEED);
    } else {
      navMode = "ULTRASONIC: right close, pivot left";
      drive(-TURN_SPEED, TURN_SPEED);
    }
    return;
  }

  // ---- Priority 2: camera ----
  bool camFresh = camTime != 0 && (millis() - camTime) < CAMERA_FRESH_MS;
  if (USE_CAMERA && headingKnown && camFresh && camClass != 0) {
    if (camClass == 1)      { navMode = "CAMERA: left blocked, arc right"; drive(BASE_SPEED, BASE_SPEED / 3); }
    else if (camClass == 3) { navMode = "CAMERA: right blocked, arc left"; drive(BASE_SPEED / 3, BASE_SPEED); }
    else {
      navMode = "CAMERA: centre blocked, pivoting";
      if (l >= r) drive(-TURN_SPEED, TURN_SPEED);
      else        drive(TURN_SPEED, -TURN_SPEED);
    }
    return;
  }

  // ---- Priority 3: GPS goal ----
  if (!haveFix()) { navMode = "WAITING: no GPS fix (or older than 3 s)"; Stop(); return; }

  double lat = gps.location.lat();
  double lon = gps.location.lng();
  navDist = TinyGPSPlus::distanceBetween(lat, lon, targetLat, targetLon);
  navBearing = TinyGPSPlus::courseTo(lat, lon, targetLat, targetLon);

  if (navDist <= ARRIVE_RADIUS_M) {
    navMode = "ARRIVED";
    Stop();
    navigating = false;
    say("Arrived at the destination.");
    return;
  }

  navHeadingValid = currentHeading(navHeading);
  if (!navHeadingValid) {
    navMode = "GPS: heading unknown, driving straight";
    driveStraight(BASE_SPEED);
    return;
  }

  double error = angleDiff(navBearing, navHeading);
  navErr = error;
  if (gyroOk && fabs(error) > PIVOT_ABOVE_DEG) {
    navMode = (error > 0) ? "GPS: pivot right" : "GPS: pivot left";
    if (error > 0) drive(TURN_SPEED, -TURN_SPEED);
    else           drive(-TURN_SPEED, TURN_SPEED);
    return;
  }
  navMode = "GPS: steering";
  double corr = constrain(TURN_GAIN * error, -BASE_SPEED, BASE_SPEED);
  drive((int)(BASE_SPEED + corr + 0.5), (int)(BASE_SPEED - corr + 0.5));
}

// ------------------------------ bluetooth -----------------------------
// Returns 0 if OK, otherwise: 1 no comma, 2 unreadable latitude,
// 3 unreadable longitude, 4 latitude out of range, 5 longitude out of range.
int parseCoordinates(const char* s, double &lat, double &lon) {
  const char* comma = strchr(s, ',');
  if (!comma) return 1;
  char* end1;
  char* end2;
  lat = strtod(s, &end1);
  lon = strtod(comma + 1, &end2);
  if (end1 == s || end1 > comma) return 2;
  if (end2 == comma + 1) return 3;
  if (lat < -90.0 || lat > 90.0) return 4;
  if (lon < -180.0 || lon > 180.0) return 5;
  return 0;
}

void reportBadCommand(Print &out, const char* cmd, int err, double lat, double lon) {
  out.print("Error: ");
  switch (err) {
    case 1: out.print("no comma found."); break;
    case 2: out.print("could not read the latitude."); break;
    case 3: out.print("could not read the longitude."); break;
    case 4: out.print("latitude must be between -90 and 90, got "); out.print(lat, 6); break;
    case 5: out.print("longitude must be between -180 and 180, got "); out.print(lon, 6); break;
  }
  int commas = 0;
  for (const char* q = cmd; *q; q++) if (*q == ',') commas++;
  if (commas > 1) out.print(" Use a DOT, not a comma, as the decimal separator.");
  out.print(" Received: [");
  out.print(cmd);
  out.println("]  Expected: lat,lon  e.g. 45.890123,11.040123");
}

void reportTooFar(Print &out, double meters) {
  out.print("Error: the target is ");
  out.print(meters / 1000.0, 1);
  out.println(" km from the rover (limit 5 km). Check the numbers: latitude first, then longitude.");
}

void handleCommand(char* cmd) {
  // Some BLE modules print their own status lines ("OK+CONN", "+CONNECTED"...).
  if (cmd[0] == '+' || strncmp(cmd, "OK+", 3) == 0 || strcasecmp(cmd, "CONNECTED") == 0) {
    say("use LIST to list all the available commands");
    return;
  }

  if (strcasecmp(cmd, "LIST") == 0) {
    say("List of commands and their function:");
    say("CAL(recalibrates gyro position and navigating is set to false)");
    say("ENC(prints virtual odometer and L/R IR  received pulses)");
    say("RESETODO(resets the virtual odometer)");
    say("STATUS(prints the status of the rover)");
    say("DRY_RUN(sets the variable dry run to 1 or 0 depending on its first value, so virtualizes the movement of the rover or not)");
    say("PRINT_EVERY(prints every result the camera is reading)");
    say("NMEA(prints everything the gps module si receiving or saying)");
    return;
  }  
  if (strcasecmp(cmd, "STOP") == 0) {
    navigating = false;
    Stop();
    say("Stopped.");
    return;
  }
  if (strcasecmp(cmd, "CAL") == 0) {
    navigating = false;
    Stop();
    if (!gyroOk) {
      say("No gyro found.");
      return;
    }
    say("Navigation stopped. Calibrating the gyro: keep the rover still...");
    calibrateGyro();
    say("Gyro calibrated.");
    return;
  }
  if (strcasecmp(cmd, "ENC") == 0) {
    printEnc(Serial3);
    printEnc(Serial);
    return;
  }
  if (strcasecmp(cmd, "RESETODO") == 0) {
    noInterrupts();
    encCountL = 0;
    encCountR = 0;
    interrupts();
    prevCountL = 0;
    prevCountR = 0;
    say("Odometer reset.");
    return;
  }
  if (strcasecmp(cmd, "STATUS") == 0) {
    printStatus(Serial3);
    printStatus(Serial);
    return;
  }
  if (strcasecmp(cmd, "DRY_RUN") == 0){
    DRY_RUN = (DRY_RUN) ? 0 : 1;
    (DRY_RUN) ? say("motors stopped until changed") : say("motors online");
    return;
  }
  if (strcasecmp(cmd, "PRINT_EVERY") == 0){
    PRINT_EVERY_RESULT = (PRINT_EVERY_RESULT) ? 0 : 1;
    (PRINT_EVERY_RESULT) ? say("print every result") : say("stops printing every result");
    return;
  }
  if (strcasecmp(cmd, "NMEA") == 0){
    RAW_NMEA = (RAW_NMEA) ? 0 : 1;
    (RAW_NMEA) ? say("print every gps char") : say("stops print all gps chars");
    return;
  }

  double lat = 0.0, lon = 0.0;
  int err = parseCoordinates(cmd, lat, lon);
  if (err != 0) {
    reportBadCommand(Serial, cmd, err, lat, lon);
    reportBadCommand(Serial3, cmd, err, lat, lon);
    return;
  }
  if (haveFix()) {
    double d = TinyGPSPlus::distanceBetween(gps.location.lat(), gps.location.lng(), lat, lon);
    if (d > MAX_TARGET_DIST_M) {
      reportTooFar(Serial, d);
      reportTooFar(Serial3, d);
      return;
    }
  }
  targetLat = lat;
  targetLon = lon;
  navigating = true;
  say(haveFix() ? "Target set, navigating." : "Target set. Waiting for a GPS fix before moving.");
}

void readBluetooth() {
  while (Serial3.available() > 0) {
    char ch = Serial3.read();
    if (ch == '\n' || ch == '\r') {
      if (btLen > 0) {
        btBuf[btLen] = '\0';
        handleCommand(btBuf);
        btLen = 0;
      }
    } else if (btLen < sizeof(btBuf) - 1) {
      btBuf[btLen++] = ch;
    } else {
      btLen = 0;                         // line too long: discard it
    }
  }
}

// ------------------------------ main ----------------------------------
void setup() {
  Serial.begin(115200);    // USB -> Serial Monitor
  Serial1.begin(115200);   // RX1 = pin 19, from the camera board
  Serial2.begin(9600);     // RX2 = pin 17, from the GPS
  Serial3.begin(BT_BAUD);  // RX3 = pin 15 / TX3 = pin 14, Bluetooth module

  pinMode(dir1PinL, OUTPUT);
  pinMode(dir2PinL, OUTPUT);
  pinMode(speedPinL, OUTPUT);
  pinMode(dir1PinR, OUTPUT);
  pinMode(dir2PinR, OUTPUT);
  pinMode(speedPinR, OUTPUT);
  Stop();

  for (int i = 0; i < 3; i++) {
    pinMode(trigPIN[i], OUTPUT);
    pinMode(echoPIN[i], INPUT);
    digitalWrite(trigPIN[i], LOW);
  }

#if USE_ENCODERS
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), isrLeft, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), isrRight, RISING);
  lastEncUpdateMs = millis();
#endif

#if USE_GYRO
  Wire.begin();
  Wire.setClock(400000);
  gyroOk = initGyro();
  if (gyroOk) {
    Serial.println("Gyro found. Calibrating: keep the rover still...");
    say("Gyro found. Calibrating: keep the rover still...");
    gyroOk = calibrateGyro();
    Serial.println(gyroOk ? "Gyro ready." : "Gyro read error: using GPS course only.");
    say(gyroOk ? "Gyro ready." : "Gyro read error: using GPS course only.");  
  } else {
    Serial.println("No gyro found (check SDA 20, SCL 21, power). Using GPS course only.");
    say("No gyro found (check SDA 20, SCL 21, power). Using GPS course only.");
  }
#endif

  lastByteTime = millis();
  if(DRY_RUN)
  Serial.println("DRY RUN: the motors will NOT move.");
  else
  say("Rover ready. Send a destination as: lat,lon");
}

void loop() {
  readBluetooth();
  readCamera();
  readGps();
  updateGyro();
  fuseGpsCourse();
  updateEncoders();
  scanUltrasonics();
  navigate();

  if(!RAW_NMEA){
    if (millis() - lastStatus >= 1000) {
      lastStatus = millis();
      printStatus(Serial);
    }
    if (navigating && millis() - lastBtStatus >= 3000) {
      lastBtStatus = millis();
      printStatus(Serial3);
    }
  }
}
