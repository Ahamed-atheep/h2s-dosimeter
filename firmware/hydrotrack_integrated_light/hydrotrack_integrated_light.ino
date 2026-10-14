/*
  HydroTrack — ESP32 H2S Dosimeter Reader Station
  ------------------------------------------------------------
  Base: your working TFT + WiFi + BME688 + AS7343 firmware.
  Added on top (without touching what already worked):
    - AS7343 ratio-model inference (12 AS7343 channels / Clear)
    - No camera-trained model, no reference-tile normalization, and no
      21-feature camera/spectral feature vector are used for dose inference
    - A new TFT page showing dose (ppm.h) + risk tier
    - h2sPpm now actually populated in the /api/read JSON
      (was hardcoded to 0 before)

  Controls (serial + web dashboard):
    B  -> start the optional exposure timer for a freshly-issued badge
    S  -> read the badge + compute dose using the AS7343 ratio model
    T  -> print a raw BME688 + AS7343 reading
    1/0 -> AS7343 LED ON/OFF
    +/- -> change AS7343 LED brightness

  IMPORTANT — confirmed on your installed library (Adafruit AS7343 v1.1.0):
    - There is no AS7343_CHANNEL_CLEAR. The Clear/VIS reading is
      AS7343_CHANNEL_VIS_TL_0 (per Adafruit's own basic_readings.ino
      example) -- already wired up correctly below.
*/

#include <Wire.h>
#include <SPI.h>

#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_AS7343.h>

// ========== WiFi & Web Server ==========
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

// ========== AS7343 ratio model + LED persistence ==========
#include <Preferences.h>
#include "as7343_ratio_model.h"

// ---- WiFi credentials (CHANGE THESE) ----
const char *ssid = "Md";
const char *password = "12345678";

// ---- AsyncWebServer ----
AsyncWebServer server(80);

// =====================================================
//        EMBEDDED DASHBOARD (served at "/")
// =====================================================
// Self-contained: no external CSS/JS/CDN calls (the badge reader has no
// internet access, only a local WiFi link to whatever is browsing it).
// Polls /api/read every second and re-renders. Activate Badge / Scan
// controls hit the same firmware functions as serial commands
// -- the exact same C functions the serial commands C / B / S call, so the
// TFT and the website always show the same state no matter which side
// triggered the action.
const char INDEX_HTML[] PROGMEM = R"HTMLPAGE(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HydroTrack Dosimeter</title>
<style>
  :root{
    --bg:#0b1220; --panel:#121b2e; --panel2:#0f1a2b; --line:#22334d;
    --text:#e7edf7; --muted:#8fa2c2; --accent:#3fd0c9;
    --safe:#39d98a; --warn:#f5c542; --danger:#ff8a3d; --critical:#ff4d5e;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; font-family:'Segoe UI',Roboto,Arial,sans-serif;
    background:radial-gradient(circle at top,#132038,#070c15 70%);
    color:var(--text); min-height:100vh; padding:24px;
  }
  h1{font-size:22px; margin:0 0 4px 0; letter-spacing:.5px;}
  .sub{color:var(--muted); font-size:13px; margin-bottom:20px;}
  .dot{display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px;}
  .dot.on{background:var(--safe); box-shadow:0 0 8px var(--safe);}
  .dot.off{background:var(--critical); box-shadow:0 0 8px var(--critical);}
  .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px;}
  .card{
    background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line); border-radius:14px; padding:18px;
  }
  .card h2{font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); margin:0 0 14px 0;}
  .metric{display:flex; align-items:baseline; justify-content:space-between; margin:8px 0;}
  .metric .label{color:var(--muted); font-size:13px;}
  .metric .val{font-size:20px; font-weight:600;}
  .big{font-size:44px; font-weight:700; line-height:1;}
  .tier{
    display:inline-block; padding:6px 16px; border-radius:999px; font-weight:700;
    font-size:15px; letter-spacing:1px; margin-top:8px;
  }
  .tier.SAFE{background:rgba(57,217,138,.15); color:var(--safe); border:1px solid var(--safe);}
  .tier.WARNING{background:rgba(245,197,66,.15); color:var(--warn); border:1px solid var(--warn);}
  .tier.DANGER{background:rgba(255,138,61,.15); color:var(--danger); border:1px solid var(--danger);}
  .tier.CRITICAL{background:rgba(255,77,94,.15); color:var(--critical); border:1px solid var(--critical);}
  .tier.NA{background:rgba(143,162,194,.15); color:var(--muted); border:1px solid var(--muted);}
  button{
    background:var(--accent); color:#062521; border:none; border-radius:10px;
    padding:10px 16px; font-weight:700; font-size:13px; cursor:pointer; margin:4px 6px 0 0;
    transition:transform .1s ease, opacity .1s ease;
  }
  button:hover{opacity:.9;}
  button:active{transform:scale(.97);}
  button.secondary{background:transparent; color:var(--text); border:1px solid var(--line);}
  button:disabled{opacity:.4; cursor:not-allowed;}
  table{width:100%; border-collapse:collapse; font-size:12px;}
  td,th{padding:4px 6px; text-align:left; border-bottom:1px solid var(--line); color:var(--muted);}
  td.v{color:var(--text); font-weight:600; text-align:right;}
  .bars{display:flex; align-items:flex-end; gap:4px; height:90px; margin-top:8px;}
  .bar{flex:1; background:linear-gradient(180deg,var(--accent),#1c6f6a); border-radius:3px 3px 0 0; min-height:2px; position:relative;}
  .barlabels{display:flex; gap:4px; margin-top:4px;}
  .barlabels span{flex:1; text-align:center; font-size:9px; color:var(--muted);}
  .toast{
    position:fixed; top:18px; right:18px; background:var(--panel); border:1px solid var(--accent);
    color:var(--text); padding:10px 16px; border-radius:10px; font-size:13px; opacity:0;
    transition:opacity .3s ease; pointer-events:none;
  }
  .toast.show{opacity:1;}
  .badge-row{display:flex; align-items:center; gap:8px; margin-top:6px;}
  footer{text-align:center; color:var(--muted); font-size:11px; margin-top:28px;}
</style>
</head>
<body>
  <h1>&#129517; HydroTrack &mdash; H&#8322;S Dosimeter Reader</h1>
  <div class="sub"><span id="connDot" class="dot off"></span><span id="connText">Connecting...</span></div>

  <div class="grid">

    <div class="card">
      <h2>Environment (BME688)</h2>
      <div class="metric"><span class="label">Temperature</span><span class="val" id="temp">-- &deg;C</span></div>
      <div class="metric"><span class="label">Humidity</span><span class="val" id="hum">-- %</span></div>
      <div class="metric"><span class="label">Pressure</span><span class="val" id="pres">-- hPa</span></div>
      <div class="metric"><span class="label">Gas resistance</span><span class="val" id="gas">-- k&Omega;</span></div>
    </div>

    <div class="card">
      <h2>Dose Result</h2>
      <div class="big" id="dose">--</div>
      <div class="sub" style="margin:0 0 6px 0;">ppm&middot;h (AS7343 spectral model)</div>
      <div id="tier" class="tier NA">NO SCAN YET</div>
      <div style="margin-top:14px;">
        <button id="btnScan">Scan badge (S)</button>
      </div>
    </div>

    <div class="card">
      <h2>AS7343 ML Model</h2>
      <div class="metric"><span class="label">Input</span><span class="val">12 × (channel / Clear)</span></div>
      <div class="metric"><span class="label">Model range</span><span class="val">5–19 ppm.h</span></div>
      <div class="metric"><span class="label">Status</span><span class="val" id="modelStatus">--</span></div>
    </div>

    <div class="card">
      <h2>Badge exposure timer</h2>
      <div class="metric"><span class="label">Status</span><span class="val" id="badgeStatus">--</span></div>
      <div class="metric"><span class="label">Elapsed</span><span class="val" id="elapsed">-- h</span></div>
      <button id="btnBadge">Activate badge (B)</button>
    </div>

    <div class="card" style="grid-column:1/-1;">
      <h2>AS7343 spectral channels (raw counts)</h2>
      <div id="satWarning" style="display:none; background:rgba(255,77,94,.15); border:1px solid var(--critical); color:var(--critical); padding:8px 12px; border-radius:8px; font-size:12px; font-weight:700; margin-bottom:10px;">
        &#9888; SATURATED &mdash; channels are pinned at max, not measuring the strip. Lower the LED brightness below.
      </div>
      <div class="bars" id="bars"></div>
      <div class="barlabels" id="barlabels"></div>
      <div class="badge-row">
        <span class="label" style="color:var(--muted); font-size:12px;">Sensor LED</span>
        <button id="btnLedOn" class="secondary">LED ON (1)</button>
        <button id="btnLedOff" class="secondary">LED OFF (0)</button>
      </div>
      <div class="badge-row" style="margin-top:12px;">
        <span class="label" style="color:var(--muted); font-size:12px; min-width:90px;">Brightness</span>
        <input type="range" id="ledSlider" min="4" max="258" step="20" value="25" style="flex:1;">
        <span class="val" id="ledMaLabel" style="min-width:60px; text-align:right;">-- mA</span>
      </div>
    </div>

  </div>

  <footer>Serial commands (C / B / S / 1 / 0) and this page control the same firmware state &mdash; whichever you use, the TFT and this dashboard stay in sync.</footer>
  <div class="toast" id="toast"></div>

<script>
const CH_LABELS = ["F1","F2","FZ","F3","F4","FY","F5","FXL","F6","F7","F8","NIR","Clear","-"];
let failCount = 0;
let sliderDragging = false;
let sliderDebounce = null;

function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h);
  t._h = setTimeout(()=>t.classList.remove('show'), 2200);
}

function setConn(ok){
  document.getElementById('connDot').className = 'dot ' + (ok?'on':'off');
  document.getElementById('connText').textContent = ok ? 'Live' : 'Reconnecting...';
}

function renderBars(channels){
  const bars = document.getElementById('bars');
  const labels = document.getElementById('barlabels');
  const max = Math.max(1, ...channels);
  bars.innerHTML = ''; labels.innerHTML = '';
  channels.forEach((v,i)=>{
    const h = Math.max(2, Math.round((v/max)*90));
    const b = document.createElement('div');
    b.className = 'bar'; b.style.height = h+'px'; b.title = (CH_LABELS[i]||i) + ': ' + v;
    bars.appendChild(b);
    const l = document.createElement('span');
    l.textContent = CH_LABELS[i] || i;
    labels.appendChild(l);
  });
}

async function refresh(){
  try{
    const r = await fetch('/api/read', {cache:'no-store'});
    if(!r.ok) throw new Error('bad status');
    const d = await r.json();
    failCount = 0; setConn(true);

    document.getElementById('temp').textContent = Number(d.temperature).toFixed(1) + ' \u00b0C';
    document.getElementById('hum').textContent  = Number(d.humidity).toFixed(1) + ' %';
    document.getElementById('pres').textContent = Number(d.pressure).toFixed(0) + ' hPa';
    document.getElementById('gas').textContent  = Number(d.gasResistance).toFixed(1) + ' k\u03a9';

    document.getElementById('modelStatus').textContent = d.as7343Found ? 'Ready ✓' : 'Sensor missing';

    document.getElementById('badgeStatus').textContent = d.badgeActive ? 'Active' : 'Not started';
    document.getElementById('elapsed').textContent = Number(d.elapsedHours).toFixed(2) + ' h';

    renderBars(d.channels);

    if(!sliderDragging){
      document.getElementById('ledSlider').value = d.ledCurrentMa;
    }
    document.getElementById('ledMaLabel').textContent = d.ledCurrentMa + ' mA';
    document.getElementById('satWarning').style.display = d.saturated ? 'block' : 'none';

    if(d.h2sPpm !== undefined && d.h2sPpm >= 0){
      document.getElementById('dose').textContent = Number(d.h2sPpm).toFixed(2);
    } else {
      document.getElementById('dose').textContent = '--';
    }
    const tierEl = document.getElementById('tier');
    const tier = d.riskTier || '--';
    tierEl.textContent = (tier === '--') ? 'NO SCAN YET' : tier;
    tierEl.className = 'tier ' + (['SAFE','WARNING','DANGER','CRITICAL'].includes(tier) ? tier : 'NA');

  } catch(e){
    failCount++;
    if(failCount > 2) setConn(false);
  }
}

async function hit(path, okMsg){
  try{
    const r = await fetch(path, {cache:'no-store'});
    const d = await r.json().catch(()=>({}));
    if(!r.ok){ toast(d.error || ('Error: '+path)); return; }
    toast(okMsg);
    refresh();
  } catch(e){ toast('Request failed: '+path); }
}

document.getElementById('btnBadge').onclick = ()=>hit('/api/activate', 'Badge timer started');
document.getElementById('btnScan').onclick  = ()=>hit('/api/scan', 'Scan complete');
document.getElementById('btnLedOn').onclick  = ()=>hit('/api/led?state=1', 'LED ON');
document.getElementById('btnLedOff').onclick = ()=>hit('/api/led?state=0', 'LED OFF');

const ledSlider = document.getElementById('ledSlider');
ledSlider.addEventListener('input', ()=>{
  sliderDragging = true;
  document.getElementById('ledMaLabel').textContent = ledSlider.value + ' mA';
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(()=>{
    hit('/api/led/current?ma=' + ledSlider.value, 'LED brightness -> ' + ledSlider.value + ' mA');
  }, 200);
});
ledSlider.addEventListener('change', ()=>{ sliderDragging = false; });

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
)HTMLPAGE";

// =====================================================
//                    TFT PINS
// =====================================================
#define TFT_CS 5
#define TFT_RST 4
#define TFT_DC 2

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// =====================================================
//                    I2C PINS
// =====================================================
#define SDA_PIN 21
#define SCL_PIN 22

// =====================================================
//                    SENSORS
// =====================================================
Adafruit_BME680 bme;
Adafruit_AS7343 as7343;

Preferences prefs;  // ESP32 flash storage for LED brightness

// =====================================================
//                    VARIABLES
// =====================================================
bool bmeFound = false;
bool as7343Found = false;

bool ledState = false;

// NEW: illumination LED drive current, in mA. The AS7343's driver LED can
// be set from 4mA to 258mA in 2mA steps (Adafruit_AS7343::setLEDCurrent).
// Your readings pinning at 17999 on almost anything is classic ADC/analog
// saturation from too much reflected light -- this lets you turn the LED
// down (or the gain/integration time) until strips
// actually spread out across the range instead of all clipping to max.
uint16_t led_current_ma = 25;  // conservative default; adjust with +/- or the dashboard slider

float temperature = 0;
float humidity = 0;
float pressure = 0;
float gasResistance = 0;

uint16_t readings[18];

int currentPage = 0;
unsigned long previousPageTime = 0;
const unsigned long PAGE_TIME = 4000;

// ---- AS7343 ratio-model + dose state -----------------------------------
// EXACT model input order: F1,F2,FZ,F3,F4,F5,FY,FXL,F6,F7,F8,NIR.
// Each feature is the raw AS7343 channel divided by Clear from the same
// measurement. No reference-tile calibration is used by this model.
const char *RATIO_CHANNEL_KEYS[12] = {
  "F1", "F2", "FZ", "F3", "F4", "F5",
  "FY", "FXL", "F6", "F7", "F8", "NIR"
};

double last_dose_ppmh = -1.0;  // -1 = no scan yet
const char *last_risk_tier = "--";

unsigned long badge_activated_millis = 0;
bool badge_active = false;

// =====================================================
//                    SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // ---------------------------------------------------
  // I2C
  // ---------------------------------------------------
  Wire.begin(SDA_PIN, SCL_PIN);

  // ---------------------------------------------------
  // TFT
  // ---------------------------------------------------
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextWrap(false);

  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(20, 20);
  tft.println("HydroTrack");
  tft.setTextSize(1);
  tft.setCursor(25, 55);
  tft.println("Sensor System");
  tft.setCursor(25, 75);
  tft.println("BME688 + AS7343");
  delay(2000);

  // ===================================================
  //                    BME688
  // ===================================================
  Serial.println();
  Serial.println("Checking BME688...");

  if (bme.begin(0x77)) {
    bmeFound = true;
    Serial.println("BME688 FOUND at 0x77");
  } else if (bme.begin(0x76)) {
    bmeFound = true;
    Serial.println("BME688 FOUND at 0x76");
  } else {
    Serial.println("BME688 NOT FOUND!");
  }

  if (bmeFound) {
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);
  }

  // ===================================================
  //                    AS7343
  // ===================================================
  Serial.println();
  Serial.println("Checking AS7343...");

  if (as7343.begin()) {
    as7343Found = true;
    Serial.println("AS7343 FOUND");
    as7343.setGain(AS7343_GAIN_256X);
    as7343.setATIME(29);
    as7343.setASTEP(599);
    as7343.enableLED(true);  // LED initially ON (as in your code)
    ledState = true;
  } else {
    Serial.println("AS7343 NOT FOUND!");
  }

  // Preferences is used only for the persisted AS7343 LED brightness.
  prefs.begin("dosimeter", false);

  // Load stored LED brightness (falls back to the 25mA default above
  // the very first time, before anything has ever been saved).
  led_current_ma = prefs.getUShort("led_ma", led_current_ma);
  if (as7343Found) {
    as7343.setLEDCurrent(led_current_ma);
    Serial.print("LED current set to ");
    Serial.print(led_current_ma);
    Serial.println(" mA");
  }

  // ===================================================
  //                    SERIAL CONTROLS
  // ===================================================
  Serial.println();
  Serial.println("======================================");
  Serial.println("          SENSOR SYSTEM READY");
  Serial.println("======================================");
  Serial.println("1 = AS7343 LED ON");
  Serial.println("0 = AS7343 LED OFF");
  Serial.println("+ = Increase LED brightness (+20mA)");
  Serial.println("- = Decrease LED brightness (-20mA)");
  Serial.println("T = Take sensor reading");
    Serial.println("B = BADGE ACTIVATE (start exposure timer)");
  Serial.println("S = SCAN badge + compute dose");
  Serial.println("======================================");

  // ===================================================
  //                    WIFI & WEB SERVER
  // ===================================================
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println();
  Serial.println(">> Idle. Waiting for a command (1 0 T B S)...");

  // ---- Dashboard page ----
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    request->send_P(200, "text/html", INDEX_HTML);
  });

  // ---- API endpoint: /api/read ----
  server.on("/api/read", HTTP_GET, [](AsyncWebServerRequest *request) {
    // Read fresh sensor data (uses your existing function) -- silent,
    // since this fires on every dashboard poll and shouldn't spam Serial.
    readSensors(false);

    // Build JSON response
    String json = "{";
    json += "\"temperature\":" + String(temperature) + ",";
    json += "\"humidity\":" + String(humidity) + ",";
    json += "\"pressure\":" + String(pressure) + ",";
    json += "\"gasResistance\":" + String(gasResistance) + ",";
    json += "\"ledOn\":" + String(ledState ? "true" : "false") + ",";
    json += "\"ledCurrentMa\":" + String(led_current_ma) + ",";
    // NEW: the library's own saturation flags -- this is what's actually
    // behind readings pinned at 17999. If either is true, the fix is to
    // lower ledCurrentMa (or gain/ATIME in setup()), not to trust the number.
    bool currentlySaturated = as7343Found && (as7343.isAnalogSaturated() || as7343.isDigitalSaturated());
    json += "\"saturated\":" + String(currentlySaturated ? "true" : "false") + ",";
    // NEW: h2sPpm is now the real model output (dose in ppm.h) from the
    // most recent SCAN, not a live per-request inference -- a scan is a
    // deliberate "read this badge" action, not a background poll.
    json += "\"h2sPpm\":" + String(last_dose_ppmh, 2) + ",";
    json += "\"riskTier\":\"" + String(last_risk_tier) + "\",";
    // The AS7343 ratio model needs no stored reference calibration.
    json += "\"modelReady\":" + String(as7343Found ? "true" : "false") + ",";

    // Badge exposure timer is informational only; it is NOT an ML input.
    json += "\"badgeActive\":" + String(badge_active ? "true" : "false") + ",";
    float elapsed_h_now = badge_active
                            ? (millis() - badge_activated_millis) / 3600000.0f
                            : 0.0f;
    json += "\"elapsedHours\":" + String(elapsed_h_now, 3) + ",";

    json += "\"bmeFound\":" + String(bmeFound ? "true" : "false") + ",";
    json += "\"as7343Found\":" + String(as7343Found ? "true" : "false") + ",";

    json += "\"channels\":[";

    // Add all 14 AS7343 channels (F1, F2, FZ, F3, F4, F5, FY, FXL, F6, F7, F8, NIR, plus two extras)
    // We'll send the first 14 entries from the readings array.
    for (int i = 0; i < 14; i++) {
      json += String(readings[i]);
      if (i < 13) json += ",";
    }
    json += "]";
    json += "}";

    request->send(200, "application/json", json);
  });

  // ---- API endpoint: /api/activate (mirrors serial 'B') ----
  server.on("/api/activate", HTTP_GET, [](AsyncWebServerRequest *request) {
    activateBadge();
    String json = "{\"badgeActive\":true}";
    request->send(200, "application/json", json);
  });

  // ---- NEW API endpoint: /api/led (mirrors serial '1' / '0') ----
  server.on("/api/led", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!as7343Found) {
      request->send(400, "application/json", "{\"error\":\"AS7343 not found\"}");
      return;
    }
    if (request->hasParam("state")) {
      bool on = request->getParam("state")->value() == "1";
      as7343.enableLED(on);
      ledState = on;
    }
    String json = "{\"ledOn\":" + String(ledState ? "true" : "false") + "}";
    request->send(200, "application/json", json);
  });

  // ---- NEW API endpoint: /api/led/current (mirrors serial '+' / '-') ----
  server.on("/api/led/current", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!as7343Found) {
      request->send(400, "application/json", "{\"error\":\"AS7343 not found\"}");
      return;
    }
    if (!request->hasParam("ma")) {
      request->send(400, "application/json", "{\"error\":\"missing ma param\"}");
      return;
    }
    int requested = request->getParam("ma")->value().toInt();
    setLedBrightness(requested);

    bool sat = as7343.isAnalogSaturated() || as7343.isDigitalSaturated();
    String json = "{\"ledCurrentMa\":" + String(led_current_ma) + ",\"saturated\":" + String(sat ? "true" : "false") + "}";
    request->send(200, "application/json", json);
  });

  // ---- NEW API endpoint: /api/scan (trigger a real badge scan) ----
  server.on("/api/scan", HTTP_GET, [](AsyncWebServerRequest *request) {
    String msg;
    runScan();
    showDosePage();  // keep TFT in sync with a scan triggered from the website
    msg = "{\"dose_ppmh\":" + String(last_dose_ppmh, 2) + ",\"riskTier\":\"" + String(last_risk_tier) + "\"}";
    request->send(200, "application/json", msg);
  });

  server.begin();
  Serial.println("HTTP server started at /api/read and /api/scan");

  // Initial TFT page
  showBMEPage();
}

// =====================================================
//                    MAIN LOOP
// =====================================================
void loop() {
  // ===================================================
  // Serial command control
  // ===================================================
  if (Serial.available()) {
    char command = Serial.read();

    // Serial Monitor line endings (CR/LF) show up here too -- ignore
    // them silently instead of treating them as unknown commands.
    if (command == '\r' || command == '\n') {
      // no-op
    } else {
      Serial.print("\n>> Command received: '");
      Serial.print(command);
      Serial.println("'");

      if (command == '1') {
        if (as7343Found) {
          as7343.enableLED(true);
          ledState = true;
          Serial.println("AS7343 LED -> ON");
        } else {
          Serial.println("AS7343 not found -- cannot control LED.");
        }
      } else if (command == '0') {
        if (as7343Found) {
          as7343.enableLED(false);
          ledState = false;
          Serial.println("AS7343 LED -> OFF");
        } else {
          Serial.println("AS7343 not found -- cannot control LED.");
        }
      } else if (command == '+') {
        setLedBrightness(led_current_ma + 20);
      } else if (command == '-') {
        setLedBrightness((led_current_ma >= 20) ? (led_current_ma - 20) : 4);
      } else if (command == 'T' || command == 't') {
        Serial.println("Taking a sensor reading...");
        readSensors();  // verbose = true: prints the full BME688 + AS7343 dump
        showBMEPage();
        delay(2000);
        showAS7343Page();
      } else if (command == 'B' || command == 'b') {
        activateBadge();
      } else if (command == 'S' || command == 's') {
        runScan();
        showDosePage();
      } else {
        Serial.println("Unknown command. Valid commands: 1 0 T B S");
      }

      Serial.println(">> Ready for next command.");
    }
  }

  // ===================================================
  // NOTE: sensors are now ONLY read when you send a command
  // (T, S) or hit /api/read or /api/scan -- not on every
  // pass through loop(). This is what was flooding the
  // Serial Monitor and burying your command output.
  // ===================================================

  // ===================================================
  // Automatically change TFT page
  // ===================================================
  if (millis() - previousPageTime >= PAGE_TIME) {
    previousPageTime = millis();

    if (currentPage == 0) {
      currentPage = 1;
      showAS7343Page();
    } else {
      currentPage = 0;
      showBMEPage();
    }
  }

  // Small delay to prevent watchdog (server runs asynchronously)
  delay(100);
}

// =====================================================
//              READ ALL SENSOR DATA
// =====================================================
void readSensors(bool verbose) {
  // ===================================================
  //                    BME688
  // ===================================================
  if (bmeFound) {
    if (bme.performReading()) {
      temperature = bme.temperature;
      humidity = bme.humidity;
      pressure = bme.pressure / 100.0;
      gasResistance = bme.gas_resistance / 1000.0;

      if (verbose) {
        Serial.println();
        Serial.println("----------- BME688 -----------");
        Serial.print("Temperature : ");
        Serial.print(temperature);
        Serial.println(" C");
        Serial.print("Humidity    : ");
        Serial.print(humidity);
        Serial.println(" %");
        Serial.print("Pressure    : ");
        Serial.print(pressure);
        Serial.println(" hPa");
        Serial.print("Gas         : ");
        Serial.print(gasResistance);
        Serial.println(" KOhm");
      }
    }
  }

  // ===================================================
  //                    AS7343
  // ===================================================
  if (as7343Found) {
    as7343.startMeasurement();
    while (!as7343.dataReady()) {
      delay(1);
    }
    as7343.readAllChannels(readings);

    if (verbose) {
      Serial.println();
      Serial.println("----------- AS7343 -----------");
      Serial.print("LED : ");
      Serial.println(ledState ? "ON" : "OFF");
      Serial.print("F1  405nm : ");
      Serial.println(readings[AS7343_CHANNEL_F1]);
      Serial.print("F2  425nm : ");
      Serial.println(readings[AS7343_CHANNEL_F2]);
      Serial.print("FZ  450nm : ");
      Serial.println(readings[AS7343_CHANNEL_FZ]);
      Serial.print("F3  475nm : ");
      Serial.println(readings[AS7343_CHANNEL_F3]);
      Serial.print("F4  515nm : ");
      Serial.println(readings[AS7343_CHANNEL_F4]);
      Serial.print("F5  550nm : ");
      Serial.println(readings[AS7343_CHANNEL_F5]);
      Serial.print("FY  555nm : ");
      Serial.println(readings[AS7343_CHANNEL_FY]);
      Serial.print("FXL 600nm : ");
      Serial.println(readings[AS7343_CHANNEL_FXL]);
      Serial.print("F6  640nm : ");
      Serial.println(readings[AS7343_CHANNEL_F6]);
      Serial.print("F7  690nm : ");
      Serial.println(readings[AS7343_CHANNEL_F7]);
      Serial.print("F8  745nm : ");
      Serial.println(readings[AS7343_CHANNEL_F8]);
      Serial.print("NIR 855nm : ");
      Serial.println(readings[AS7343_CHANNEL_NIR]);
      Serial.print("Clear     : ");
      Serial.println(readings[AS7343_CHANNEL_VIS_TL_0]);
    }
  }
}

// Convenience overload so existing call sites that just want a normal,
// printed reading (e.g. the 'T' command) don't need to pass an argument.
void readSensors() {
  readSensors(true);
}

// =====================================================
//       AS7343 RATIO FEATURES FOR THE ML MODEL
// =====================================================
// The trained model expects exactly 12 values:
// F1/Clear, F2/Clear, FZ/Clear, F3/Clear, F4/Clear, F5/Clear,
// FY/Clear, FXL/Clear, F6/Clear, F7/Clear, F8/Clear, NIR/Clear.
// These must come from the SAME AS7343 measurement.
bool buildAS7343RatioFeatures(double *features) {
  if (!as7343Found) {
    Serial.println("ERROR: AS7343 not found -- cannot build ML features.");
    return false;
  }

  const double clear = (double)readings[AS7343_CHANNEL_VIS_TL_0];
  if (clear <= 0.0) {
    Serial.println("ERROR: AS7343 Clear is zero -- refusing to run model.");
    return false;
  }

  features[0]  = (double)readings[AS7343_CHANNEL_F1]  / clear;
  features[1]  = (double)readings[AS7343_CHANNEL_F2]  / clear;
  features[2]  = (double)readings[AS7343_CHANNEL_FZ]  / clear;
  features[3]  = (double)readings[AS7343_CHANNEL_F3]  / clear;
  features[4]  = (double)readings[AS7343_CHANNEL_F4]  / clear;
  features[5]  = (double)readings[AS7343_CHANNEL_F5]  / clear;
  features[6]  = (double)readings[AS7343_CHANNEL_FY]  / clear;
  features[7]  = (double)readings[AS7343_CHANNEL_FXL] / clear;
  features[8]  = (double)readings[AS7343_CHANNEL_F6]  / clear;
  features[9]  = (double)readings[AS7343_CHANNEL_F7]  / clear;
  features[10] = (double)readings[AS7343_CHANNEL_F8]  / clear;
  features[11] = (double)readings[AS7343_CHANNEL_NIR] / clear;

  return true;
}

// =====================================================
//        NEW: LED BRIGHTNESS CONTROL
// =====================================================
// Centralizes clamping + applying + persisting the LED current so serial
// (+/-) and the web slider (/api/led/current) can't disagree with each
// other. Datasheet/library range is 4-258mA in 2mA steps; anything outside
// that gets clamped instead of silently failing.
void setLedBrightness(int requested_ma) {
  if (!as7343Found) {
    Serial.println("AS7343 not found -- cannot control LED brightness.");
    return;
  }
  int clamped = requested_ma;
  if (clamped < 4) clamped = 4;
  if (clamped > 258) clamped = 258;
  clamped -= (clamped % 2);  // driver only accepts 2mA steps

  led_current_ma = (uint16_t)clamped;
  as7343.setLEDCurrent(led_current_ma);
  prefs.putUShort("led_ma", led_current_ma);

  Serial.print("LED brightness -> ");
  Serial.print(led_current_ma);
  Serial.println(" mA");

  // If a strip/tile is currently under the sensor, a fresh reading right
  // after changing brightness lets you see immediately whether channels
  // are still pinned at max (saturated) or have come off the ceiling.
  readSensors(false);
  bool sat = as7343.isAnalogSaturated() || as7343.isDigitalSaturated();
  if (sat) {
    Serial.println("WARNING: still saturated at this brightness -- try lower current, or reduce gain/ATIME in setup().");
  }
}

// =====================================================
//                 BADGE ACTIVATION
// =====================================================
void activateBadge() {
  badge_activated_millis = millis();
  badge_active = true;
  Serial.println("Badge activated. Exposure timer started.");
  // NOTE: millis()-based -- resets on power loss. Fine for a demo; for
  // real deployment log activation time to your FastAPI backend and pull
  // elapsed time from server time instead (same caveat as before).
}

// =====================================================
//                    SCANNING
// =====================================================
void runScan() {
  if (!as7343Found) {
    Serial.println("ERROR: AS7343 not found -- cannot scan.");
    return;
  }

  // One fresh measurement supplies all 12 channels and Clear.
  readSensors(false);

  double ratio_features[12];
  if (!buildAS7343RatioFeatures(ratio_features)) {
    last_dose_ppmh = -1.0;
    last_risk_tier = "--";
    return;
  }

  last_dose_ppmh = score_as7343_ratio(ratio_features);
  last_risk_tier = riskTier(last_dose_ppmh);

  Serial.println("---- AS7343 ML scan result ----");
  Serial.print("Clear: ");
  Serial.println(readings[AS7343_CHANNEL_VIS_TL_0]);
  Serial.println("12 channel/Clear ratios:");
  for (int i = 0; i < 12; i++) {
    Serial.print("  ");
    Serial.print(RATIO_CHANNEL_KEYS[i]);
    Serial.print("/Clear = ");
    Serial.println(ratio_features[i], 6);
  }
  printDoseResult();
}

const char *riskTier(double dose_ppmh) {
  if (dose_ppmh < 10) return "SAFE";
  else if (dose_ppmh < 40) return "WARNING";
  else if (dose_ppmh < 80) return "DANGER";
  else return "CRITICAL";
}

void printDoseResult() {
  Serial.println("---- Scan result ----");
  Serial.print("Dose: ");
  Serial.print(last_dose_ppmh, 2);
  Serial.println(" ppm.h");
  Serial.print("Risk tier: ");
  Serial.println(last_risk_tier);
  Serial.print("Temp: ");
  Serial.print(temperature, 1);
  Serial.println(" C");
  Serial.print("Humidity: ");
  Serial.print(humidity, 1);
  Serial.println(" %");
  Serial.println("----------------------");
}

// =====================================================
//                TFT BME688 PAGE
// =====================================================
void showBMEPage() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor(5, 5);
  tft.println("BME688");
  tft.drawLine(0, 25, 160, 25, ST77XX_WHITE);

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 35);
  tft.print("Temperature:");
  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(85, 32);
  tft.print(temperature, 1);
  tft.setTextSize(1);
  tft.print(" C");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 60);
  tft.print("Humidity:");
  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(85, 57);
  tft.print(humidity, 1);
  tft.setTextSize(1);
  tft.print(" %");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 85);
  tft.print("Pressure:");
  tft.setTextColor(ST77XX_YELLOW);
  tft.setTextSize(2);
  tft.setCursor(70, 82);
  tft.print(pressure, 0);
  tft.setTextSize(1);
  tft.print(" hPa");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 110);
  tft.print("Gas:");
  tft.setTextColor(ST77XX_MAGENTA);
  tft.setTextSize(2);
  tft.setCursor(50, 107);
  tft.print(gasResistance, 1);
  tft.setTextSize(1);
  tft.print(" K");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 140);
  tft.print("HydroTrack");
}

// =====================================================
//                TFT AS7343 PAGE
// =====================================================
void showAS7343Page() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor(5, 2);
  tft.println("AS7343");

  tft.setTextSize(1);
  tft.setCursor(90, 8);
  tft.setTextColor(ST77XX_WHITE);
  tft.print("LED:");
  if (ledState) {
    tft.setTextColor(ST77XX_GREEN);
    tft.print("ON");
  } else {
    tft.setTextColor(ST77XX_RED);
    tft.print("OFF");
  }
  tft.drawLine(0, 25, 160, 25, ST77XX_WHITE);

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 32);
  tft.print("405:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F1]);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(85, 32);
  tft.print("425:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F2]);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 52);
  tft.print("475:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F3]);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(85, 52);
  tft.print("515:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F4]);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 72);
  tft.print("550:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F5]);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(85, 72);
  tft.print("555:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_FY]);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 92);
  tft.print("600:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_FXL]);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(85, 92);
  tft.print("640:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F6]);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 112);
  tft.print("690:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F7]);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(85, 112);
  tft.print("745:");
  tft.setTextColor(ST77XX_GREEN);
  tft.print(readings[AS7343_CHANNEL_F8]);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 132);
  tft.print("NIR:");
  tft.setTextColor(ST77XX_YELLOW);
  tft.print(readings[AS7343_CHANNEL_NIR]);
}

// =====================================================
//        NEW: TFT DOSE PAGE -- shown right after a scan
// =====================================================
void showDosePage() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor(5, 2);
  tft.println("DOSE");
  tft.drawLine(0, 25, 160, 25, ST77XX_WHITE);

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 35);
  tft.print("Dose:");
  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(60, 32);
  tft.print(last_dose_ppmh, 1);
  tft.setTextSize(1);
  tft.print(" ppm.h");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 65);
  tft.print("Risk:");

  uint16_t tierColor = ST77XX_GREEN;
  if (strcmp(last_risk_tier, "WARNING") == 0) tierColor = ST77XX_YELLOW;
  else if (strcmp(last_risk_tier, "DANGER") == 0) tierColor = ST77XX_MAGENTA;
  else if (strcmp(last_risk_tier, "CRITICAL") == 0) tierColor = ST77XX_RED;

  tft.setTextColor(tierColor);
  tft.setTextSize(2);
  tft.setCursor(50, 62);
  tft.print(last_risk_tier);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 140);
  tft.print("HydroTrack");
}
