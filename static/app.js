import { RoadScene } from "./road_scene.js";
import { MotoCockpit } from "./moto_cockpit.js";

const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");

const onOffBtn = document.getElementById("onOffBtn");
const powerStateText = document.getElementById("powerStateText");
const dwellArc = document.getElementById("dwellArc");
const camStatusPill = document.getElementById("camStatusPill");

const helmetBanner = document.getElementById("helmetBanner");
const bannerIcon = document.getElementById("bannerIcon");
const bannerTitle = document.getElementById("bannerTitle");
const bannerSub = document.getElementById("bannerSub");

const pipFrame = document.getElementById("pipFrame");
const leftHandBadge = document.getElementById("leftHandBadge");
const rightHandBadge = document.getElementById("rightHandBadge");

const badgeLeftFist = document.getElementById("badgeLeftFist");
const badgeRightFist = document.getElementById("badgeRightFist");
const badgeBothFists = document.getElementById("badgeBothFists");
const badgeBothOpen = document.getElementById("badgeBothOpen");
const badgeIndex = document.getElementById("badgeIndex");

const demoPanel = document.getElementById("demoPanel");
const demoHelmetBtn = document.getElementById("demoHelmetBtn");
const retryCamBtn = document.getElementById("retryCamBtn");
const browserCamBtn = document.getElementById("browserCamBtn");
const scanTimerText = document.getElementById("scanTimerText");
const demoSpeedBtn = document.getElementById("demoSpeedBtn");

const indexCursor = document.getElementById("indexCursor");
const cursorArc = document.getElementById("cursorArc");

const roadScene = new RoadScene();
const cockpit = new MotoCockpit();

const CIRCUMFERENCE = 2 * Math.PI * 28; // matches SVG r=28
const CURSOR_CIRCUMFERENCE = 2 * Math.PI * 19; // matches cursor SVG r=19

// Live telemetry from Python MediaPipe + Ultralytics YOLO backend
let telemetry = {
  system_on: false,
  helmet_detected: false,
  helmet_score: 0,
  yolo_class: "none",
  yolo_conf: 0,
  compliance_status: "NO_HEAD",
  compliance_label: "NO HELMET DETECTED",
  mid_ride_buffering: false,
  no_helmet_sustained_sec: 0,
  face_detected: false,
  left_hand: "NONE",
  right_hand: "NONE",
  active_gesture: "NONE",
  index_progress: 0,
  index_hand: "NONE",
  cursor_active: false,
  cursor_x: 0.5,
  cursor_y: 0.5,
  camera_available: false,
  camera_label: "Initializing...",
  fps: 0,
  demo_state: { helmet: "no_helmet", left_hand: "NONE", right_hand: "NONE" },
};

// Alert banner override & Web Audio warning state
let alertOverride = null; // { type: 'danger'|'safe', title: string, sub: string, untilMs: number }
let audioCtx = null;

// 1-minute (60s) periodic compliance scan timer + Demo 10x Speed toggle
let demoFastScan = false;
let scanRemainingSec = 60;
let scanViolationBuffering = false;

// Smooth screen-space cursor position & ignition-button dwell state
let smoothCursorX = window.innerWidth * 0.5;
let smoothCursorY = window.innerHeight * 0.5;
let cursorDwellSec = 0;
let cursorCooldownSec = 0;
let toggleInFlight = false;
const DWELL_THRESHOLD_SEC = 0.50;
let browserCamStream = null;
let autoCamTried = false;

// Local smooth riding physics state
let speedKmh = 0;
let lanePos = 1.0; // 0 = left lane, 1 = center lane, 2 = right lane
let rollRad = 0;
let lastFrameMs = performance.now();

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function playAlertBeep() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!audioCtx) audioCtx = new Ctx();
    if (audioCtx.state === "suspended") audioCtx.resume();

    const now = audioCtx.currentTime;
    for (let i = 0; i < 2; i++) {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(i === 0 ? 880 : 660, now + i * 0.18);
      gain.gain.setValueAtTime(0.001, now + i * 0.18);
      gain.gain.exponentialRampToValueAtTime(0.16, now + i * 0.18 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.18 + 0.15);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now + i * 0.18);
      osc.stop(now + i * 0.18 + 0.16);
    }
  } catch (_) {
    // ignore Web Audio autoplay restrictions
  }
}

function setBannerAlert(type, title, sub, durationMs = 4200) {
  alertOverride = {
    type,
    title,
    sub,
    untilMs: performance.now() + durationMs,
  };
  syncHud();
}

async function toggleIgnition(clearDemoIndex = false) {
  if (toggleInFlight) return;
  toggleInFlight = true;
  try {
    const res = await fetch("/api/toggle", { method: "POST" });
    const data = await res.json();
    telemetry.system_on = Boolean(data.system_on);

    if (data.blocked) {
      speedKmh = 0;
      playAlertBeep();
      setBannerAlert(
        "danger",
        data.message || "Helmet not detected! Vehicle ignition prohibited!",
        "PRE-RIDE GATE LOCKED • WEAR HELMET OR TURBAN / PAGRI (>= 60% CONFIDENCE)",
        4500
      );
    } else if (telemetry.system_on) {
      scanRemainingSec = demoFastScan ? 10 : 60;
      scanViolationBuffering = false;
      const isTurban = data.compliance_status === "TURBAN_EXEMPT";
      setBannerAlert(
        "safe",
        data.message || (isTurban ? "IGNITION ON - TURBAN EXEMPT" : "IGNITION ON - HELMET VERIFIED"),
        `YOLO CLASS: ${(data.yolo_class || "helmet").toUpperCase()} (${Math.round((data.yolo_conf || 0) * 100)}%) • FULL THROTTLE UNLOCKED`,
        3200
      );
    } else {
      scanViolationBuffering = false;
      alertOverride = null;
    }

    if (clearDemoIndex && !telemetry.camera_available) {
      await fetch("/api/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ left_hand: "NONE", right_hand: "NONE" }),
      });
    }
    syncHud();
  } catch (err) {
    console.error("Toggle error:", err);
  } finally {
    toggleInFlight = false;
  }
}

// Toggle IGNITE via mouse click
onOffBtn.addEventListener("click", () => toggleIgnition(false));

// Toggle DEMO 10x SPEED (switches 60s scan cycle to 10s for live judging)
demoSpeedBtn.addEventListener("click", () => {
  demoFastScan = !demoFastScan;
  demoSpeedBtn.classList.toggle("is-active", demoFastScan);
  demoSpeedBtn.setAttribute("aria-pressed", String(demoFastScan));
  scanRemainingSec = demoFastScan ? Math.min(scanRemainingSec, 10) : 60;
  updateScanTimerDisplay();
});

// Demo mode helper buttons (cycle no_helmet -> helmet -> special_case)
demoHelmetBtn.addEventListener("click", async () => {
  const cur = String(telemetry.demo_state?.helmet || "no_helmet");
  const nextHelmet =
    cur === "no_helmet" ? "helmet" : cur === "helmet" ? "special_case" : "no_helmet";
  await fetch("/api/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ helmet: nextHelmet }),
  });
});

retryCamBtn.addEventListener("click", async () => {
  await fetch("/api/retry_camera", { method: "POST" });
});

async function startBrowserCam() {
  if (browserCamStream || !navigator.mediaDevices?.getUserMedia) return;
  try {
    browserCamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360, facingMode: "user" },
      audio: false,
    });

    // Attach live mirrored video directly inside pipFrame so your face shows at 60 FPS
    const liveVideo = document.createElement("video");
    liveVideo.srcObject = browserCamStream;
    liveVideo.muted = true;
    liveVideo.playsInline = true;
    liveVideo.autoplay = true;
    Object.assign(liveVideo.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "100%",
      objectFit: "cover",
      transform: "scaleX(-1)",
      borderRadius: "50%",
      zIndex: "5",
    });

    // Hide any placeholder img/canvas inside pipFrame and mount liveVideo
    Array.from(pipFrame.children).forEach((el) => {
      el.style.display = "none";
    });
    pipFrame.style.position = "relative";
    pipFrame.appendChild(liveVideo);
    await liveVideo.play();

    const capCanvas = document.createElement("canvas");
    capCanvas.width = 320;
    capCanvas.height = 240;
    const capCtx = capCanvas.getContext("2d");

    browserCamBtn.textContent = "Browser Cam Active";

    // Single-flight upload loop (prevents overload on Render's 0.1 CPU)
    let uploadInFlight = false;
    setInterval(() => {
      if (!browserCamStream || uploadInFlight || document.hidden) return;
      uploadInFlight = true;
      capCtx.drawImage(liveVideo, 0, 0, capCanvas.width, capCanvas.height);
      capCanvas.toBlob(
        async (blob) => {
          try {
            if (blob) {
              await fetch("/api/upload_frame", {
                method: "POST",
                headers: { "Content-Type": "image/jpeg" },
                body: blob,
              });
            }
          } catch (_) {
            // ignore transient network errors
          } finally {
            uploadInFlight = false;
          }
        },
        "image/jpeg",
        0.60
      );
    }, 450);
  } catch (err) {
    camStatusPill.textContent = "Browser cam denied/unavailable";
    console.error("Browser cam error:", err);
  }
}

browserCamBtn.addEventListener("click", startBrowserCam);

// If OpenCV didn't find a camera after 3s, automatically try browser webcam fallback
setTimeout(() => {
  if (!telemetry.camera_available && !autoCamTried) {
    autoCamTried = true;
    startBrowserCam();
  }
}, 3000);

document.querySelectorAll("[data-demo-gesture]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const g = btn.getAttribute("data-demo-gesture");
    let left_hand = "NONE";
    let right_hand = "NONE";
    if (g === "OPEN") {
      left_hand = "OPEN";
      right_hand = "OPEN";
    } else if (g === "LEFT_FIST") {
      left_hand = "FIST";
      right_hand = "OPEN";
    } else if (g === "RIGHT_FIST") {
      left_hand = "OPEN";
      right_hand = "FIST";
    } else if (g === "BOTH_FISTS") {
      left_hand = "FIST";
      right_hand = "FIST";
    } else if (g === "INDEX") {
      right_hand = "INDEX_ONLY";
    }
    await fetch("/api/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ left_hand, right_hand }),
    });
  });
});

// Connect to live Server-Sent Events telemetry stream
function connectTelemetry() {
  const es = new EventSource("/stream");
  es.onmessage = (event) => {
    try {
      telemetry = JSON.parse(event.data);
      syncHud();
    } catch (_) {
      // ignore malformed packet
    }
  };
  es.onerror = () => {
    camStatusPill.textContent = "Reconnecting telemetry...";
  };
}
connectTelemetry();

function updateScanTimerDisplay() {
  const totalSec = Math.max(0, Math.ceil(scanRemainingSec));
  const mins = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const secs = String(totalSec % 60).padStart(2, "0");
  scanTimerText.textContent = `NEXT 1-MIN SCAN: ${mins}:${secs}`;
}

function updatePeriodicScan(dt) {
  if (!telemetry.system_on) {
    scanRemainingSec = demoFastScan ? 10 : 60;
    scanViolationBuffering = false;
    updateScanTimerDisplay();
    return;
  }

  if (telemetry.helmet_detected) {
    scanViolationBuffering = false;
  }

  scanRemainingSec -= dt;
  if (scanRemainingSec <= 0) {
    scanRemainingSec = demoFastScan ? 10 : 60;
    if (telemetry.helmet_detected) {
      scanViolationBuffering = false;
      setBannerAlert(
        "safe",
        "1-MIN COMPLIANCE SCAN: PASSED ✔",
        `${telemetry.compliance_label} (${Math.round((telemetry.yolo_conf || 0) * 100)}%) • FULL THROTTLE MAINTAINED`,
        3000
      );
    } else {
      scanViolationBuffering = true;
      playAlertBeep();
      setBannerAlert(
        "danger",
        "HELMET REMOVED - PROGRESSIVE SPEED BUFFERING",
        "1-MIN SCAN VIOLATION • DECELERATING TO 20 KM/H SAFETY CAP",
        4000
      );
    }
  }
  updateScanTimerDisplay();
}

function syncHud() {
  // Power IGNITE button visual state
  onOffBtn.classList.toggle("is-on", telemetry.system_on);
  onOffBtn.classList.toggle("is-off", !telemetry.system_on);
  onOffBtn.setAttribute("aria-pressed", String(telemetry.system_on));
  powerStateText.textContent = telemetry.system_on ? "ON" : "OFF";

  // Camera status & optional Demo controls visibility
  camStatusPill.textContent = `${telemetry.camera_label} • ${Math.round(telemetry.fps || 0)} FPS`;
  demoPanel.classList.toggle("hidden", Boolean(telemetry.camera_available));

  // Helmet / Turban compliance & alert banner
  const nowMs = performance.now();
  if (alertOverride && nowMs > alertOverride.untilMs) {
    alertOverride = null;
  }

  const safe = Boolean(telemetry.helmet_detected);
  const buffering = Boolean(telemetry.system_on && (telemetry.mid_ride_buffering || scanViolationBuffering));
  const confPct = Math.round((telemetry.yolo_conf || 0) * 100);

  pipFrame.classList.toggle("is-safe", safe);
  pipFrame.classList.toggle("is-danger", !safe);

  if (alertOverride) {
    helmetBanner.classList.toggle("status-safe", alertOverride.type === "safe");
    helmetBanner.classList.toggle("status-danger", alertOverride.type !== "safe");
    bannerIcon.textContent = alertOverride.type === "safe" ? "✔" : "⚠";
    bannerTitle.textContent = alertOverride.title;
    bannerSub.textContent = alertOverride.sub;
  } else if (buffering) {
    helmetBanner.classList.remove("status-safe");
    helmetBanner.classList.add("status-danger");
    bannerIcon.textContent = "⚠";
    bannerTitle.textContent = "HELMET REMOVED - PROGRESSIVE SPEED BUFFERING";
    bannerSub.textContent = "SPEED CAPPING SMOOTHLY TO 20 KM/H • PUT HELMET ON TO RESTORE FULL SPEED";
  } else if (safe) {
    helmetBanner.classList.add("status-safe");
    helmetBanner.classList.remove("status-danger");
    bannerIcon.textContent = "✔";
    const isTurban = telemetry.compliance_status === "TURBAN_EXEMPT";
    if (telemetry.system_on) {
      bannerTitle.textContent = isTurban
        ? "IGNITION ON - TURBAN EXEMPT"
        : "IGNITION ON - HELMET VERIFIED";
    } else {
      bannerTitle.textContent = isTurban
        ? "TURBAN / PAGRI EXEMPT - READY TO IGNITE"
        : "HELMET DETECTED - READY TO IGNITE";
    }
    bannerSub.textContent = `YOLO (${telemetry.yolo_class.toUpperCase()} ${confPct}%) • FULL THROTTLE UNLOCKED (UP TO 85 KM/H)`;
  } else {
    helmetBanner.classList.remove("status-safe");
    helmetBanner.classList.add("status-danger");
    bannerIcon.textContent = "⚠";
    bannerTitle.textContent = telemetry.system_on
      ? "NO HELMET DETECTED - COMPLIANCE WARNING"
      : "NO HELMET DETECTED - IGNITION LOCKED";
    bannerSub.textContent = telemetry.system_on
      ? "SPEED BUFFERING ENGAGES IN 2.0s IF HELMET REMAINS OFF"
      : `YOLO (${telemetry.yolo_class.toUpperCase()} ${confPct}%) • WEAR HELMET OR TURBAN / PAGRI TO IGNITE`;
  }

  // Hand chips under circular PiP
  leftHandBadge.textContent = `L: ${telemetry.left_hand.replace("_ONLY", "")}`;
  rightHandBadge.textContent = `R: ${telemetry.right_hand.replace("_ONLY", "")}`;
  leftHandBadge.classList.toggle("active", telemetry.left_hand !== "NONE");
  rightHandBadge.classList.toggle("active", telemetry.right_hand !== "NONE");

  // Gesture HUD badges
  const g = telemetry.active_gesture;
  badgeLeftFist.classList.toggle("is-active", g === "LEFT_FIST");
  badgeRightFist.classList.toggle("is-active", g === "RIGHT_FIST");
  badgeBothFists.classList.toggle("is-active", g === "BOTH_FISTS");
  badgeBothFists.classList.add("is-brake");
  badgeBothOpen.classList.toggle("is-active", g === "BOTH_OPEN");
  badgeBothOpen.classList.add("is-go");
  badgeIndex.classList.toggle("is-active", g === "INDEX_TOGGLE");
}

function updatePhysics(dt) {
  const on = Boolean(telemetry.system_on);
  const buffering = Boolean(telemetry.mid_ride_buffering || scanViolationBuffering);
  const g = telemetry.active_gesture;

  // Full speed allowed while compliant OR before 2.0s mid-ride buffering triggers;
  // caps smoothly to 20 km/h when buffering is active.
  const maxSpeed = buffering ? 20 : 85;

  if (!on) {
    // Ignition OFF -> locked at 0 km/h
    speedKmh = Math.max(0, speedKmh - 45 * dt);
    rollRad += (0 - rollRad) * Math.min(1, 7 * dt);
    return;
  }

  // Speed dynamics
  if (g === "BOTH_FISTS") {
    // Apply strong braking toward 0 km/h
    speedKmh = Math.max(0, speedKmh - 54 * dt);
  } else if (g === "BOTH_OPEN") {
    // Smooth acceleration up to cruising speed (capped at 20 if no helmet)
    if (speedKmh < maxSpeed) {
      speedKmh = Math.min(maxSpeed, speedKmh + 28 * dt);
    } else {
      // Buffer down smoothly if helmet was removed mid-ride
      speedKmh = Math.max(maxSpeed, speedKmh - 30 * dt);
    }
  } else if (g === "LEFT_FIST" || g === "RIGHT_FIST") {
    // Maintain gentle rolling momentum while steering, respecting helmet cap
    if (speedKmh > maxSpeed) {
      speedKmh = Math.max(maxSpeed, speedKmh - 26 * dt);
    } else {
      speedKmh = Math.max(0, speedKmh - 4 * dt);
    }
  } else {
    // Coasting friction + helmet speed cap enforcement
    if (speedKmh > maxSpeed) {
      speedKmh = Math.max(maxSpeed, speedKmh - 26 * dt);
    } else {
      speedKmh = Math.max(0, speedKmh - 7 * dt);
    }
  }

  // Steering / Lane shifting dynamics (Left Fist -> Left, Right Fist -> Right)
  let targetRoll = 0;
  const steerSpeed = 1.15; // lanes per second
  if (g === "LEFT_FIST") {
    lanePos = Math.max(-0.15, lanePos - steerSpeed * dt);
    targetRoll = -0.115;
  } else if (g === "RIGHT_FIST") {
    lanePos = Math.min(2.15, lanePos + steerSpeed * dt);
    targetRoll = 0.115;
  } else {
    // Softly center toward nearest valid lane [0, 1, 2] when not steering
    const nearest = Math.round(Math.max(0, Math.min(2, lanePos)));
    lanePos += (nearest - lanePos) * Math.min(1, 2.2 * dt);
  }

  rollRad += (targetRoll - rollRad) * Math.min(1, 8.5 * dt);
}

function updateAirCursor(dt) {
  cursorCooldownSec = Math.max(0, cursorCooldownSec - dt);
  const cActive = Boolean(telemetry.cursor_active);
  indexCursor.classList.toggle("hidden", !cActive);

  let hoveringIgnition = false;
  if (cActive) {
    const targetX = (telemetry.cursor_x ?? 0.5) * window.innerWidth;
    const targetY = (telemetry.cursor_y ?? 0.5) * window.innerHeight;
    smoothCursorX += (targetX - smoothCursorX) * Math.min(1, 16 * dt);
    smoothCursorY += (targetY - smoothCursorY) * Math.min(1, 16 * dt);
    indexCursor.style.transform = `translate3d(${smoothCursorX.toFixed(1)}px, ${smoothCursorY.toFixed(1)}px, 0)`;

    const rect = onOffBtn.getBoundingClientRect();
    const pad = 22;
    hoveringIgnition =
      smoothCursorX >= rect.left - pad &&
      smoothCursorX <= rect.right + pad &&
      smoothCursorY >= rect.top - pad &&
      smoothCursorY <= rect.bottom + pad;
  }

  onOffBtn.classList.toggle("is-hovered", hoveringIgnition);

  // ONLY turn ON/OFF when cursor is actually at the ignition button
  if (cActive && hoveringIgnition && cursorCooldownSec <= 0) {
    cursorDwellSec += dt;
    if (cursorDwellSec >= DWELL_THRESHOLD_SEC) {
      cursorDwellSec = 0;
      cursorCooldownSec = 1.6;
      toggleIgnition(true);
    }
  } else {
    cursorDwellSec = Math.max(0, cursorDwellSec - dt * 3.0);
  }

  const prog = Math.max(0, Math.min(1, cursorDwellSec / DWELL_THRESHOLD_SEC));
  dwellArc.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - prog));
  cursorArc.style.strokeDashoffset = String(CURSOR_CIRCUMFERENCE * (1 - prog));
}

function renderLoop(nowMs) {
  const dt = Math.max(0.001, Math.min(0.05, (nowMs - lastFrameMs) / 1000));
  lastFrameMs = nowMs;

  updatePeriodicScan(dt);
  updateAirCursor(dt);
  updatePhysics(dt);
  roadScene.update(dt, speedKmh, lanePos);

  const W = window.innerWidth;
  const H = window.innerHeight;
  ctx.clearRect(0, 0, W, H);

  roadScene.draw(ctx, W, H, lanePos, speedKmh, rollRad);
  cockpit.draw(ctx, W, H, {
    speedKmh,
    rollRad,
    systemOn: telemetry.system_on,
    helmetDetected: telemetry.helmet_detected,
    activeGesture: telemetry.active_gesture,
    timeSec: nowMs / 1000,
  });

  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
