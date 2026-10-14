# HydroTrack ESP32 Hardware Reader Station (`hydrotrack_integrated_light`)

## 1. Overview
The **HydroTrack ESP32 Hardware Reader Station** is a specialized optical readout and environmental telemetry unit designed for passive lead-acetate ($Pb(Ac)_2$) chemical badges.

Instead of relying solely on RGB camera sensors, the station uses an **AMS AS7343 12-channel multispectral sensor** to capture high-resolution optical reflectance fingerprints ($405\text{ nm}$ to $855\text{ nm}$ + $Clear$) alongside a **Bosch BME688** environmental sensor ($T$, $RH$, Barometric Pressure, Gas Resistance).

---

## 2. Hardware Pinout & Wiring

| Component | Interface | ESP32 GPIO Pin | Function |
| :--- | :--- | :--- | :--- |
| **AS7343 Multispectral** | $I^2C$ | `GPIO 21` (SDA), `GPIO 22` (SCL) | 12 spectral channels + Clear reference |
| **BME688 Environmental** | $I^2C$ | `GPIO 21` (SDA), `GPIO 22` (SCL) | Temperature, Humidity, Pressure, VOC Gas |
| **ST7735 TFT Display** | SPI | `GPIO 5` (CS), `GPIO 4` (RST), `GPIO 2` (DC) | Real-time on-device UI |
| **AS7343 Illumination LED** | Internal Driver | Controlled via I2C | Adjustable drive current ($4\text{ mA}$ to $258\text{ mA}$) |

---

## 3. Required Arduino Libraries
- `Adafruit_GFX`
- `Adafruit_ST7735`
- `Adafruit_Sensor`
- `Adafruit_BME680`
- `Adafruit_AS7343`
- `AsyncTCP`
- `ESPAsyncWebServer`
- `Preferences`

---

## 4. Optical Spectrum & Ratio Channels

The AS7343 sensor extracts 12 optical channels normalized against the broadband Clear channel ($Channel / Clear$):

| Channel | Peak Wavelength | Optical Band | Feature Index |
| :--- | :---: | :--- | :---: |
| **F1** | $405\text{ nm}$ | Violet | `input[0]` |
| **F2** | $425\text{ nm}$ | Indigo | `input[1]` |
| **FZ** | $450\text{ nm}$ | Blue | `input[2]` |
| **F3** | $475\text{ nm}$ | Cyan | `input[3]` |
| **F4** | $515\text{ nm}$ | Teal / Green | `input[4]` |
| **F5** | $550\text{ nm}$ | Green | `input[5]` |
| **FY** | $555\text{ nm}$ | Yellow-Green | `input[6]` |
| **FXL** | $600\text{ nm}$ | Amber | `input[7]` |
| **F6** | $640\text{ nm}$ | Orange-Red | `input[8]` |
| **F7** | $690\text{ nm}$ | Deep Red | `input[9]` |
| **F8** | $745\text{ nm}$ | Far-Red | `input[10]` |
| **NIR** | $855\text{ nm}$ | Near-Infrared | `input[11]` |
| **Clear** | $550\text{ nm}$ | Broadband VIS | *Normalization Base* |

---

## 5. Serial & Web Controls

### Serial Interface (115200 Baud)
- `1` : AS7343 LED ON
- `0` : AS7343 LED OFF
- `+` : Increase LED brightness ($+20\text{ mA}$)
- `-` : Decrease LED brightness ($-20\text{ mA}$)
- `T` : Print raw sensor reading
- `B` : Activate badge shift timer
- `S` : Trigger optical scan & compute cumulative dose ($ppm\cdot h$)

### REST API Endpoints
- `GET /` : Embedded web dashboard
- `GET /api/read` : Full JSON telemetry dump
- `GET /api/scan` : Run optical scan & return dose
- `GET /api/activate` : Start shift timer
- `GET /api/led?state=1|0` : Illumination control
- `GET /api/led/current?ma=X` : Set drive current ($4$ to $258\text{ mA}$)
