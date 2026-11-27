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
const pipStream = document.getElementById("pipStream");
const pipOverlay = document.getElementById("pipOverlay");
const pipCtx = pipOverlay.getContext("2d");
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
let liveVideoEl = null;
let snapCanvas = null;
let snapCtx = null;
let autoCamTried = false;
let scanInFlight = false;
const keysPressed = new Set();

// Real-time client-side MediaPipe Hands state (active from start, zero backend frame polling)
let clientHandsActive = false;
let clientHandsState = {
  left_hand: "NONE",
  right_hand: "NONE",
  active_gesture: "NONE",
  cursor_active: false,
  cursor_x: 0.5,
  cursor_y: 0.5,
  hands_landmarks: [],
};

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

function captureSnapshotB64() {
  if (!liveVideoEl || liveVideoEl.readyState < 2) return null;
  if (!snapCanvas) {
    snapCanvas = document.createElement("canvas");
    snapCanvas.width = 320;
    snapCanvas.height = 240;
    snapCtx = snapCanvas.getContext("2d");
  }
  snapCtx.drawImage(liveVideoEl, 0, 0, snapCanvas.width, snapCanvas.height);
  return snapCanvas.toDataURL("image/jpeg", 0.75);
}

async function toggleIgnition(clearDemoIndex = false) {
  if (toggleInFlight) return;
  toggleInFlight = true;
  try {
    const turningOn = !telemetry.system_on;
    let image_b64 = null;
    if (turningOn) {
      setBannerAlert(
        "safe",
        "CAPTURING IMAGE & VERIFYING HELMET...",
        "RUNNING LOCAL YOLO HELMET / TURBAN COMPLIANCE CHECK",
        2500
      );
      if (!liveVideoEl || liveVideoEl.readyState < 2) {
        await startBrowserCam();
      }
      image_b64 = captureSnapshotB64();
    }
    const res = await fetch("/api/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: turningOn, image_b64 }),
    });
    const data = await res.json();
    if (data?.telemetry) {
      telemetry = data.telemetry;
    }
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

    // Hide raw MJPEG placeholder img while keeping pipOverlay canvas above liveVideo
    pipStream.style.display = "none";
    pipFrame.style.position = "relative";
    pipFrame.insertBefore(liveVideo, pipOverlay);
    await liveVideo.play();
    liveVideoEl = liveVideo;

    browserCamBtn.textContent = "Browser Cam Active";
    camStatusPill.textContent = "LIVE HANDS • YOLO SNAPSHOT ON IGNITE & 1-MIN";
    startBrowserHandTracker(liveVideo);
  } catch (err) {
    camStatusPill.textContent = "Browser cam denied/unavailable";
    console.error("Browser cam error:", err);
  }
}

browserCamBtn.addEventListener("click", startBrowserCam);

function dist2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function classifyHandPoseJS(lms) {
  const wrist = lms[0];
  const isExtended = (tip, pip, mcp) =>
    dist2D(wrist, lms[tip]) > dist2D(wrist, lms[pip]) * 1.12 &&
    dist2D(lms[mcp], lms[tip]) > dist2D(lms[mcp], lms[pip]) * 1.08;
  const isCurled = (tip, pip) =>
    dist2D(wrist, lms[tip]) < dist2D(wrist, lms[pip]) * 1.07;

  const indexUp = isExtended(8, 6, 5);
  const middleUp = isExtended(12, 10, 9);
  const ringUp = isExtended(16, 14, 13);
  const pinkyUp = isExtended(20, 18, 17);

  const indexCurled = isCurled(8, 6);
  const middleCurled = isCurled(12, 10);
  const ringCurled = isCurled(16, 14);
  const pinkyCurled = isCurled(20, 18);

  const upCount = Number(indexUp) + Number(middleUp) + Number(ringUp) + Number(pinkyUp);
  const curledCount =
    Number(indexCurled) + Number(middleCurled) + Number(ringCurled) + Number(pinkyCurled);

  if (indexUp && middleCurled && ringCurled && pinkyCurled) return "INDEX_ONLY";
  if (upCount === 0 && curledCount >= 3) return "FIST";
  if (upCount >= 3) return "OPEN";
  return "NEUTRAL";
}

function onBrowserHandResults(results) {
  let leftState = "NONE";
  let rightState = "NONE";
  let cursorActive = false;
  let cursorX = clientHandsState.cursor_x;
  let cursorY = clientHandsState.cursor_y;
  const handsPayload = [];

  const allLms = results?.multiHandLandmarks || [];
  const allHanded = results?.multiHandedness || [];

  if (allLms.length > 0) {
    const detected = [];
    for (let i = 0; i < allLms.length; i++) {
      const lms = allLms[i];
      const rawLabel = (allHanded[i]?.label || "Right").toUpperCase();
      const pose = classifyHandPoseJS(lms);
      detected.push({ wristX: lms[0].x, side: rawLabel, lms, pose });
    }
    if (detected.length === 2) {
      detected.sort((a, b) => a.wristX - b.wristX);
      detected[0].side = "LEFT";
      detected[1].side = "RIGHT";
    }
    for (const item of detected) {
      if (item.side === "LEFT") leftState = item.pose;
      else rightState = item.pose;
      handsPayload.push({
        side: item.side,
        pose: item.pose,
        points: item.lms.map((p) => [p.x, p.y]),
      });
      if (item.pose === "INDEX_ONLY" && !cursorActive) {
        const tip = item.lms[8];
        cursorX = Math.max(0, Math.min(1, (tip.x - 0.14) / 0.72));
        cursorY = Math.max(0, Math.min(1, (tip.y - 0.14) / 0.72));
        cursorActive = true;
      }
    }
  }

  let idxHand = "NONE";
  if (leftState === "INDEX_ONLY" && rightState !== "FIST") idxHand = "LEFT";
  else if (rightState === "INDEX_ONLY" && leftState !== "FIST") idxHand = "RIGHT";

  let gesture = "NONE";
  if (idxHand !== "NONE") gesture = "INDEX_TOGGLE";
  else if (leftState === "FIST" && rightState === "FIST") gesture = "BOTH_FISTS";
  else if (leftState === "FIST") gesture = "LEFT_FIST";
  else if (rightState === "FIST") gesture = "RIGHT_FIST";
  else if (leftState === "OPEN" && rightState === "OPEN") gesture = "BOTH_OPEN";

  clientHandsState = {
    left_hand: leftState,
    right_hand: rightState,
    active_gesture: gesture,
    cursor_active: cursorActive,
    cursor_x: cursorX,
    cursor_y: cursorY,
    hands_landmarks: handsPayload,
  };
  syncHandUiOnly();
}

function startBrowserHandTracker(videoEl) {
  if (typeof window.Hands === "undefined") {
    camStatusPill.textContent = "HAND TRACKER CDN FAILED • CHECK INTERNET";
    console.warn("MediaPipe Hands CDN script not loaded; using keyboard/demo gestures.");
    return;
  }
  const hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 0,
    minDetectionConfidence: 0.45,
    minTrackingConfidence: 0.45,
    selfieMode: true,
  });
  hands.onResults(onBrowserHandResults);
  clientHandsActive = true;
  syncHud();

  let sending = false;
  let lastSentMs = 0;
  async function pumpHands(nowMs) {
    if (!browserCamStream) return;
    if (!sending && !document.hidden && videoEl.readyState >= 2 && nowMs - lastSentMs >= 45) {
      sending = true;
      lastSentMs = nowMs;
      try {
        await hands.send({ image: videoEl });
      } catch (err) {
        console.warn("Hands frame error:", err);
      } finally {
        sending = false;
      }
    }
    requestAnimationFrame(pumpHands);
  }
  requestAnimationFrame(pumpHands);
}

// Initialize live camera preview + client-side hand tracking at startup (no per-second server image uploads)
setTimeout(() => {
  if (!browserCamStream && !autoCamTried) {
    autoCamTried = true;
    startBrowserCam();
  }
}, 800);

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

async function runPeriodicComplianceScan() {
  if (scanInFlight || !telemetry.system_on) return;
  scanInFlight = true;
  try {
    const image_b64 = captureSnapshotB64();
    const res = await fetch("/api/compliance_scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64 }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data?.telemetry) {
        telemetry = data.telemetry;
      }
    }
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
    syncHud();
  } catch (_) {
    // ignore transient network errors
  } finally {
    scanInFlight = false;
  }
}

function updatePeriodicScan(dt) {
  if (!telemetry.system_on) {
    scanRemainingSec = demoFastScan ? 10 : 60;
    scanViolationBuffering = false;
    updateScanTimerDisplay();
    return;
  }

  scanRemainingSec -= dt;
  if (scanRemainingSec <= 0) {
    scanRemainingSec = demoFastScan ? 10 : 60;
    runPeriodicComplianceScan();
  }
  updateScanTimerDisplay();
}

function syncHandUiOnly() {
  const leftHand =
    clientHandsActive && clientHandsState.left_hand !== "NONE"
      ? clientHandsState.left_hand
      : clientHandsActive && telemetry.left_hand === "NONE"
      ? "NONE"
      : telemetry.left_hand;
  const rightHand =
    clientHandsActive && clientHandsState.right_hand !== "NONE"
      ? clientHandsState.right_hand
      : clientHandsActive && telemetry.right_hand === "NONE"
      ? "NONE"
      : telemetry.right_hand;

  leftHandBadge.textContent = `L: ${leftHand.replace("_ONLY", "")}`;
  rightHandBadge.textContent = `R: ${rightHand.replace("_ONLY", "")}`;
  leftHandBadge.classList.toggle("active", leftHand !== "NONE");
  rightHandBadge.classList.toggle("active", rightHand !== "NONE");

  syncGestureBadges();
  drawPipOverlay();
}

function syncHud() {
  // Power IGNITE button visual state
  onOffBtn.classList.toggle("is-on", telemetry.system_on);
  onOffBtn.classList.toggle("is-off", !telemetry.system_on);
  onOffBtn.setAttribute("aria-pressed", String(telemetry.system_on));
  powerStateText.textContent = telemetry.system_on ? "ON" : "OFF";

  // Camera status & optional Demo controls visibility
  if (clientHandsActive) {
    camStatusPill.textContent = `LIVE HANDS • ${telemetry.camera_label}`;
  } else {
    camStatusPill.textContent = `${telemetry.camera_label}`;
  }
  demoPanel.classList.toggle("hidden", Boolean(telemetry.camera_available || browserCamStream));

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

  // Hand chips, Gesture HUD badges, and PiP landmark overlay
  syncHandUiOnly();
}

function syncGestureBadges() {
  const g = getEffectiveGesture();
  badgeLeftFist.classList.toggle("is-active", g === "LEFT_FIST");
  badgeRightFist.classList.toggle("is-active", g === "RIGHT_FIST");
  badgeBothFists.classList.toggle("is-active", g === "BOTH_FISTS");
  badgeBothFists.classList.add("is-brake");
  badgeBothOpen.classList.toggle("is-active", g === "BOTH_OPEN");
  badgeBothOpen.classList.add("is-go");
  badgeIndex.classList.toggle("is-active", g === "INDEX_TOGGLE");
}

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function mapCoverPoint(nx, ny, size, aspect) {
  if (aspect >= 1) {
    return [(nx * aspect - (aspect - 1) * 0.5) * size, ny * size];
  }
  const inv = 1 / Math.max(0.1, aspect);
  return [nx * size, (ny * inv - (inv - 1) * 0.5) * size];
}

function drawPipOverlay() {
  const S = pipOverlay.width;
  pipCtx.clearRect(0, 0, S, S);
  // Only draw client canvas overlay when live browser <video> is active
  if (!browserCamStream) return;

  const aspect =
    liveVideoEl && liveVideoEl.videoWidth && liveVideoEl.videoHeight
      ? liveVideoEl.videoWidth / liveVideoEl.videoHeight
      : telemetry.frame_aspect || 4 / 3;

  // 1. Head / Face Bounding Box + YOLO label
  const hb = telemetry.head_box;
  if (hb) {
    const [x1, y1] = mapCoverPoint(hb.x1, hb.y1, S, aspect);
    const [x2, y2] = mapCoverPoint(hb.x2, hb.y2, S, aspect);
    const col = hb.compliant ? "#3ef09b" : "#ff5252";
    const cLen = Math.max(10, Math.min(20, (x2 - x1) * 0.22));

    pipCtx.strokeStyle = col;
    pipCtx.lineWidth = 2.2;
    pipCtx.beginPath();
    pipCtx.moveTo(x1, y1 + cLen); pipCtx.lineTo(x1, y1); pipCtx.lineTo(x1 + cLen, y1);
    pipCtx.moveTo(x2 - cLen, y1); pipCtx.lineTo(x2, y1); pipCtx.lineTo(x2, y1 + cLen);
    pipCtx.moveTo(x1, y2 - cLen); pipCtx.lineTo(x1, y2); pipCtx.lineTo(x1 + cLen, y2);
    pipCtx.moveTo(x2 - cLen, y2); pipCtx.lineTo(x2, y2); pipCtx.lineTo(x2, y2 - cLen);
    pipCtx.stroke();

    const lbl = hb.label || (hb.compliant ? "HELMET OK" : "NO HELMET");
    pipCtx.font = "bold 10px Inter, sans-serif";
    const tw = pipCtx.measureText(lbl).width + 10;
    const ly = Math.max(18, y1 - 4);
    pipCtx.fillStyle = "rgba(12, 18, 28, 0.82)";
    pipCtx.fillRect(x1, ly - 13, tw, 15);
    pipCtx.fillStyle = col;
    pipCtx.fillText(lbl, x1 + 5, ly - 2);
  }

  // 2. 21-Point MediaPipe Hand Skeleton Landmarks (prefer live client hands when active)
  const hands =
    clientHandsActive && clientHandsState.hands_landmarks.length > 0
      ? clientHandsState.hands_landmarks
      : telemetry.hands_landmarks || [];
  for (const hand of hands) {
    const pts = hand.points || [];
    if (pts.length < 21) continue;

    let lineCol = "#8fe2ff";
    if (hand.pose === "FIST") lineCol = "#ffaa2b";
    else if (hand.pose === "OPEN") lineCol = "#4df59e";
    else if (hand.pose === "INDEX_ONLY") lineCol = "#ffe042";

    const mapped = pts.map(([px, py]) => mapCoverPoint(px, py, S, aspect));

    pipCtx.strokeStyle = lineCol;
    pipCtx.lineWidth = 2.6;
    pipCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      pipCtx.moveTo(mapped[a][0], mapped[a][1]);
      pipCtx.lineTo(mapped[b][0], mapped[b][1]);
    }
    pipCtx.stroke();

    pipCtx.fillStyle = "#ffffff";
    for (const [mx, my] of mapped) {
      pipCtx.beginPath();
      pipCtx.arc(mx, my, 3.1, 0, Math.PI * 2);
      pipCtx.fill();
    }

    const [wx, wy] = mapped[0];
    const tag = `${(hand.side || "H")[0]}:${(hand.pose || "").replace("_ONLY", "")}`;
    pipCtx.font = "bold 9px Inter, sans-serif";
    const tx = Math.max(8, Math.min(S - 54, wx - 20));
    const ty = Math.max(14, Math.min(S - 8, wy + 16));
    pipCtx.fillStyle = "rgba(12, 18, 28, 0.84)";
    pipCtx.fillRect(tx - 3, ty - 10, 48, 13);
    pipCtx.fillStyle = lineCol;
    pipCtx.fillText(tag, tx, ty);
  }
}

function getEffectiveGesture() {
  if (keysPressed.has("Space") || keysPressed.has("ArrowDown") || keysPressed.has("KeyS")) {
    return "BOTH_FISTS";
  }
  if (keysPressed.has("ArrowLeft") || keysPressed.has("KeyA")) {
    return "LEFT_FIST";
  }
  if (keysPressed.has("ArrowRight") || keysPressed.has("KeyD")) {
    return "RIGHT_FIST";
  }
  if (keysPressed.has("ArrowUp") || keysPressed.has("KeyW")) {
    return "BOTH_OPEN";
  }
  if (clientHandsActive && clientHandsState.active_gesture !== "NONE") {
    return clientHandsState.active_gesture;
  }
  return telemetry.active_gesture;
}

window.addEventListener("keydown", (e) => {
  keysPressed.add(e.code);
  syncGestureBadges();
});
window.addEventListener("keyup", (e) => {
  keysPressed.delete(e.code);
  syncGestureBadges();
});

function updatePhysics(dt) {
  const on = Boolean(telemetry.system_on);
  const buffering = Boolean(telemetry.mid_ride_buffering || scanViolationBuffering);
  const g = getEffectiveGesture();

  // Full speed allowed while compliant; caps smoothly to 20 km/h when buffering is active.
  const maxSpeed = buffering ? 20 : 85;

  // Steering & handlebar roll work from the very start (even when OFF or no helmet)
  let targetRoll = 0;
  const steerSpeed = 1.15;
  if (g === "LEFT_FIST") {
    lanePos = Math.max(-0.15, lanePos - steerSpeed * dt);
    targetRoll = -0.115;
  } else if (g === "RIGHT_FIST") {
    lanePos = Math.min(2.15, lanePos + steerSpeed * dt);
    targetRoll = 0.115;
  } else {
    const nearest = Math.round(Math.max(0, Math.min(2, lanePos)));
    lanePos += (nearest - lanePos) * Math.min(1, 2.2 * dt);
  }
  rollRad += (targetRoll - rollRad) * Math.min(1, 8.5 * dt);

  if (!on) {
    // Ignition OFF -> speed remains locked at 0 km/h
    speedKmh = Math.max(0, speedKmh - 45 * dt);
    return;
  }

  // Speed dynamics when ON
  if (g === "BOTH_FISTS") {
    speedKmh = Math.max(0, speedKmh - 54 * dt);
  } else if (g === "BOTH_OPEN") {
    if (speedKmh < maxSpeed) {
      speedKmh = Math.min(maxSpeed, speedKmh + 28 * dt);
    } else {
      speedKmh = Math.max(maxSpeed, speedKmh - 30 * dt);
    }
  } else if (g === "LEFT_FIST" || g === "RIGHT_FIST") {
    if (speedKmh > maxSpeed) {
      speedKmh = Math.max(maxSpeed, speedKmh - 26 * dt);
    } else {
      speedKmh = Math.max(0, speedKmh - 4 * dt);
    }
  } else {
    if (speedKmh > maxSpeed) {
      speedKmh = Math.max(maxSpeed, speedKmh - 26 * dt);
    } else {
      speedKmh = Math.max(0, speedKmh - 7 * dt);
    }
  }
}

function updateAirCursor(dt) {
  cursorCooldownSec = Math.max(0, cursorCooldownSec - dt);
  const useClientCursor = clientHandsActive && clientHandsState.cursor_active;
  const cActive = useClientCursor || Boolean(telemetry.cursor_active);
  const rawCx = useClientCursor ? clientHandsState.cursor_x : (telemetry.cursor_x ?? 0.5);
  const rawCy = useClientCursor ? clientHandsState.cursor_y : (telemetry.cursor_y ?? 0.5);

  indexCursor.classList.toggle("hidden", !cActive);

  let hoveringIgnition = false;
  if (cActive) {
    const targetX = rawCx * window.innerWidth;
    const targetY = rawCy * window.innerHeight;
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
    activeGesture: getEffectiveGesture(),
    timeSec: nowMs / 1000,
  });

  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
