import base64
import gc
import math
import os
import sys
import threading
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch

from train_model import (
    ensure_linear_classifier,
    load_linear_classifier,
    predict_linear_probs,
)

cv2.setNumThreads(1)
try:
    torch.set_num_threads(1)
except Exception:
    pass


class VisionTracker:
    """Real-time OpenCV + MediaPipe webcam tracker for helmet safety and 2-hand riding gestures."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self.camera_available = False
        self.camera_label = "Initializing..."

        # Latest encoded JPEG frame (square-cropped for circular PiP)
        self._latest_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._init_placeholder_frame()

        # Simulation / system state
        self.system_on = False
        self.helmet_override: str = "auto"  # "auto" | "helmet" | "special_case" | "no_helmet"
        self.demo_state: Dict[str, object] = {
            "helmet": "no_helmet",
            "left_hand": "NONE",
            "right_hand": "NONE",
        }

        # Telemetry state
        self.helmet_detected = False
        self.helmet_score = 0.0
        self.yolo_class = "none"
        self.yolo_conf = 0.0
        self.compliance_status = "NO_HEAD"
        self.compliance_label = "NO HELMET DETECTED"
        self.mid_ride_buffering = False
        self.no_helmet_sustained_sec = 0.0
        self.face_detected = False
        self.left_hand_state = "NONE"
        self.right_hand_state = "NONE"
        self.active_gesture = "NONE"
        self.index_progress = 0.0
        self.index_hand = "NONE"
        self.cursor_active = False
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.head_box_norm: Optional[Dict[str, object]] = None
        self.hands_landmarks: list = []
        self.frame_aspect: float = 4.0 / 3.0
        self.fps = 0.0

        # Internal temporal smoothing & gesture timers
        self._helmet_ema = 0.0
        self._no_helmet_timer = 0.0
        self._cursor_x_ema = 0.5
        self._cursor_y_ema = 0.5
        self._index_hold_sec = 0.0
        self._toggle_cooldown = 0.0
        self._index_hold_threshold = 0.45
        self._last_probe_time = 0.0
        self._browser_frame: Optional[np.ndarray] = None
        self._browser_jpeg_raw: Optional[bytes] = None
        self._browser_seq = 0
        self._processed_browser_seq = 0
        self._browser_frame_ts = 0.0
        self._browser_event = threading.Event()
        self._processed_event = threading.Event()
        self._last_yolo_ts = 0.0
        self._last_yolo_res: Tuple[str, float] = ("none", 0.0)
        self._last_demo_key: Optional[Tuple[object, ...]] = None

        # Frozen EfficientNet-B0 feature extractor + balanced linear classifier (auto-trains if missing)
        weights_path = ensure_linear_classifier()
        self._eff_backbone, self._eff_head, self._cls_names = load_linear_classifier(weights_path)
        print(f"[VisionTracker] Loaded frozen-backbone linear classifier: {weights_path}")

        # MediaPipe Hands (model_complexity=0 keeps RAM & CPU low for cloud/edge) & OpenCV Haar
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )
        self._pose = None
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Warm up classifier & MediaPipe once at boot so first live frame is fast
        try:
            dummy_rgb = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            predict_linear_probs(self._eff_backbone, self._eff_head, [dummy_rgb])
            self._hands.process(np.zeros((224, 224, 3), dtype=np.uint8))
        except Exception:
            pass
        gc.collect()

    def _init_placeholder_frame(self) -> None:
        img = np.zeros((360, 360, 3), dtype=np.uint8)
        img[:] = (30, 24, 20)
        cv2.putText(
            img,
            "READY - CLICK IGNITE",
            (52, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 225, 80),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self._latest_jpeg = buf.tobytes()
            self._frame_seq = 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.camera_label = "SNAPSHOT MODE (IGNITE / 1-MIN)"

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._hands.close()
        if self._pose is not None:
            self._pose.close()

    def _open_camera(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._last_probe_time = time.monotonic()
        # Headless cloud Linux (e.g. Render) has no /dev/video0; skip slow V4L2 probing
        if sys.platform != "win32" and not os.path.exists(f"/dev/video{self.camera_index}"):
            self.camera_available = False
            self.camera_label = "BROWSER / CLOUD CAM"
            print("[VisionTracker] No /dev/video device; awaiting browser webcam frames")
            return False

        # Prefer CAP_DSHOW first on Windows (fast & reliable); fallback to CAP_MSMF / CAP_ANY
        candidates = [
            (self.camera_index, cv2.CAP_DSHOW),
            (self.camera_index, cv2.CAP_MSMF),
            (self.camera_index, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),
            (2, cv2.CAP_DSHOW),
        ]
        for idx, backend in candidates:
            if not self._running:
                return False
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap.isOpened():
                # Configure resolution before reading warmup frames; reject pure-black dummy/IR feeds
                if backend == cv2.CAP_DSHOW:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, frame = False, None
                non_black = False
                for _ in range(15):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        if float(frame.mean()) >= 2.5 or float(frame.std()) >= 2.5:
                            non_black = True
                            break
                    time.sleep(0.05)

                if ok and non_black and frame is not None:
                    self._cap = cap
                    self.camera_available = True
                    self.camera_label = f"LIVE CAM #{idx}"
                    print(f"[VisionTracker] Opened webcam index={idx} backend={backend} shape={frame.shape}")
                    return True
            cap.release()

        self.camera_available = False
        self.camera_label = "BROWSER / CLOUD CAM"
        print("[VisionTracker] No non-black OpenCV webcam found; awaiting browser webcam or Demo Mode")
        return False

    def _capture_single_opencv_frame(self) -> Optional[np.ndarray]:
        """Open camera briefly (if available), grab 1 fresh non-black frame, and release."""
        if sys.platform != "win32" and not os.path.exists(f"/dev/video{self.camera_index}"):
            return None
        candidates = [
            (self.camera_index, cv2.CAP_DSHOW),
            (self.camera_index, cv2.CAP_MSMF),
            (self.camera_index, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW),
        ]
        for idx, backend in candidates:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap.isOpened():
                if backend == cv2.CAP_DSHOW:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, frame = False, None
                non_black = False
                for _ in range(10):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        if float(frame.mean()) >= 2.5 or float(frame.std()) >= 2.5:
                            non_black = True
                            break
                    time.sleep(0.03)
                cap.release()
                if ok and non_black and frame is not None:
                    self.camera_available = True
                    self.camera_label = f"SNAPSHOT CAM #{idx}"
                    return frame
            else:
                cap.release()
        return None

    def analyze_snapshot(
        self, jpeg_bytes: Optional[bytes] = None, check_mid_ride: bool = False
    ) -> Dict[str, object]:
        """Run OpenCV + YOLO ('best.pt') + MediaPipe Hands on a single on-demand snapshot."""
        raw_frame = None
        source_label = "DEMO SNAPSHOT"

        if jpeg_bytes:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                raw_frame = decoded
                source_label = "SNAPSHOT (BROWSER CAM)"
        else:
            raw_frame = self._capture_single_opencv_frame()
            if raw_frame is not None:
                source_label = self.camera_label

        if raw_frame is not None:
            self._process_single_frame(
                raw_frame, source_label=source_label, check_mid_ride=check_mid_ride
            )
        else:
            # Demo mode fallback when no camera frame is available
            display_frame, left_state, right_state, demo_cls = self._render_demo_frame(
                time.perf_counter()
            )
            with self._lock:
                ov = self.helmet_override
            if ov in ("helmet", "special_case", "no_helmet"):
                demo_cls = ov
            yolo_cls = demo_cls
            yolo_conf = 0.95 if demo_cls != "no_helmet" else 0.92
            if demo_cls == "helmet":
                helmet_now = True
                comp_status, comp_label = "HELMET_VERIFIED", "HELMET DETECTED"
            elif demo_cls == "special_case":
                helmet_now = True
                comp_status, comp_label = "TURBAN_EXEMPT", "TURBAN / PAGRI EXEMPT"
            else:
                helmet_now = False
                comp_status, comp_label = "NO_HELMET", "NO HELMET DETECTED"
            ok_enc, buf = cv2.imencode(
                ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            with self._lock:
                self.face_detected = True
                self.helmet_detected = helmet_now
                self.helmet_score = yolo_conf
                self.yolo_class = yolo_cls
                self.yolo_conf = yolo_conf
                self.compliance_status = comp_status
                self.compliance_label = comp_label
                self.left_hand_state = left_state
                self.right_hand_state = right_state
                self.head_box_norm = {
                    "x1": 0.33,
                    "y1": 0.23,
                    "x2": 0.67,
                    "y2": 0.60,
                    "label": f"{yolo_cls.upper()} ({yolo_conf:.2f})",
                    "compliant": helmet_now,
                }
                self.hands_landmarks = []
                if check_mid_ride and self.system_on:
                    self.mid_ride_buffering = not helmet_now
                elif helmet_now:
                    self.mid_ride_buffering = False
                if ok_enc and buf is not None:
                    self._latest_jpeg = buf.tobytes()
                    self._frame_seq += 1

        return self.get_telemetry()

    def _process_single_frame(
        self, raw_frame: np.ndarray, source_label: str, check_mid_ride: bool = False
    ) -> None:
        frame = cv2.flip(raw_frame, 1)
        fh, fw = frame.shape[:2]
        frame_aspect = round(fw / max(1, fh), 4)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_found = False
        helmet_now = False
        yolo_cls = "none"
        yolo_conf = 0.0
        comp_status = "NO_HEAD"
        comp_label = "NO RIDER HEAD DETECTED"
        head_box_norm: Optional[Dict[str, object]] = None
        hands_lms_payload: list = []
        left_state = "NONE"
        right_state = "NONE"

        # 1. OpenCV Haar Face -> Head Box & Local YOLO Classification
        head_box, head_pts = self._detect_head_box(frame, None)
        if head_box is not None:
            face_found = True
            yolo_cls, yolo_conf = self._classify_head_yolo(frame, head_box)

            with self._lock:
                ov = self.helmet_override
            if ov == "helmet":
                yolo_cls, yolo_conf = "helmet", 0.96
            elif ov == "special_case":
                yolo_cls, yolo_conf = "special_case", 0.95
            elif ov == "no_helmet":
                yolo_cls, yolo_conf = "no_helmet", 0.95

            if yolo_cls == "helmet" and yolo_conf >= 0.60:
                helmet_now = True
                comp_status, comp_label = "HELMET_VERIFIED", "HELMET DETECTED"
                box_col, box_lbl = (80, 245, 140), f"HELMET ({yolo_conf:.2f})"
            elif yolo_cls == "special_case" and yolo_conf >= 0.60:
                helmet_now = True
                comp_status, comp_label = "TURBAN_EXEMPT", "TURBAN / PAGRI EXEMPT"
                box_col, box_lbl = (80, 245, 140), f"TURBAN ({yolo_conf:.2f})"
            else:
                helmet_now = False
                comp_status, comp_label = "NO_HELMET", "NO HELMET DETECTED"
                box_col, box_lbl = (65, 85, 255), f"NO HELMET ({yolo_conf:.2f})"

            head_box_norm = {
                "x1": round(head_box[0] / max(1, fw), 4),
                "y1": round(head_box[1] / max(1, fh), 4),
                "x2": round(head_box[2] / max(1, fw), 4),
                "y2": round(head_box[3] / max(1, fh), 4),
                "label": box_lbl,
                "compliant": helmet_now,
            }
            self._draw_corner_box(frame, head_box, box_col, box_lbl)
            for px, py in head_pts:
                cv2.circle(frame, (px, py), 3, (0, 235, 255), -1)

        # 2. MediaPipe Hands (21-point landmarks on snapshot)
        hand_res = self._hands.process(rgb)
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            detected = []
            for lms, handed in zip(
                hand_res.multi_hand_landmarks, hand_res.multi_handedness
            ):
                label = handed.classification[0].label.upper()
                wrist_x = lms.landmark[0].x
                pose_lbl = self._classify_hand_pose(lms)
                detected.append((wrist_x, label, lms, pose_lbl))

            if len(detected) == 2:
                detected.sort(key=lambda item: item[0])
                detected[0] = (detected[0][0], "LEFT", detected[0][2], detected[0][3])
                detected[1] = (detected[1][0], "RIGHT", detected[1][2], detected[1][3])

            for _, side, lms, pose_lbl in detected:
                if side == "LEFT":
                    left_state = pose_lbl
                else:
                    right_state = pose_lbl
                hands_lms_payload.append(
                    {
                        "side": side,
                        "pose": pose_lbl,
                        "points": [
                            [round(pt.x, 3), round(pt.y, 3)] for pt in lms.landmark
                        ],
                    }
                )
                self._draw_hand_skeleton(frame, lms, side, pose_lbl)

        display_frame = self._center_square_crop(frame, 360)
        ok_enc, buf = cv2.imencode(
            ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        with self._lock:
            self.camera_label = source_label
            self.face_detected = face_found
            self.helmet_detected = helmet_now
            self.helmet_score = yolo_conf if helmet_now else (1.0 - yolo_conf)
            self.yolo_class = yolo_cls
            self.yolo_conf = yolo_conf
            self.compliance_status = comp_status
            self.compliance_label = comp_label
            self.head_box_norm = head_box_norm
            self.hands_landmarks = hands_lms_payload
            self.frame_aspect = frame_aspect
            self.left_hand_state = left_state
            self.right_hand_state = right_state
            if check_mid_ride and self.system_on:
                self.mid_ride_buffering = not helmet_now
                self.no_helmet_sustained_sec = 2.5 if not helmet_now else 0.0
            elif helmet_now:
                self.mid_ride_buffering = False
                self.no_helmet_sustained_sec = 0.0
            if ok_enc and buf is not None:
                self._latest_jpeg = buf.tobytes()
                self._frame_seq += 1

    def push_browser_frame(self, jpeg_bytes: bytes) -> bool:
        if not jpeg_bytes:
            return False
        self.analyze_snapshot(jpeg_bytes, check_mid_ride=False)
        return True

    def push_browser_frame_and_wait(self, jpeg_bytes: bytes, timeout: float = 0.35) -> bool:
        if not jpeg_bytes:
            return False
        self.analyze_snapshot(jpeg_bytes, check_mid_ride=False)
        return True

    def retry_camera(self) -> bool:
        with self._lock:
            return self._open_camera()

    def attempt_ignite(self, target_state: Optional[bool] = None) -> Dict[str, object]:
        """Enforce Pre-Ride Ignition Gate: require verified helmet or special_case (turban) >= 0.60."""
        with self._lock:
            want_on = (not self.system_on) if target_state is None else bool(target_state)
            if not want_on:
                self.system_on = False
                self._no_helmet_timer = 0.0
                self.mid_ride_buffering = False
                self.no_helmet_sustained_sec = 0.0
                return {
                    "system_on": False,
                    "blocked": False,
                    "compliance_status": self.compliance_status,
                    "yolo_class": self.yolo_class,
                    "yolo_conf": round(self.yolo_conf, 3),
                    "message": "Ignition turned OFF.",
                }

            if self.helmet_detected:
                self.system_on = True
                self._no_helmet_timer = 0.0
                self.mid_ride_buffering = False
                self.no_helmet_sustained_sec = 0.0
                msg = (
                    "IGNITION ON - TURBAN EXEMPT"
                    if self.compliance_status == "TURBAN_EXEMPT"
                    else "IGNITION ON - HELMET VERIFIED"
                )
                return {
                    "system_on": True,
                    "blocked": False,
                    "compliance_status": self.compliance_status,
                    "yolo_class": self.yolo_class,
                    "yolo_conf": round(self.yolo_conf, 3),
                    "message": msg,
                }

            self.system_on = False
            return {
                "system_on": False,
                "blocked": True,
                "compliance_status": self.compliance_status,
                "yolo_class": self.yolo_class,
                "yolo_conf": round(self.yolo_conf, 3),
                "message": "Helmet not detected! Vehicle ignition prohibited!",
            }

    def toggle_system(self) -> bool:
        return bool(self.attempt_ignite(None)["system_on"])

    def set_system(self, state: bool) -> bool:
        return bool(self.attempt_ignite(bool(state))["system_on"])

    def set_helmet_override(self, mode: str) -> str:
        with self._lock:
            if mode in ("auto", "helmet", "special_case", "no_helmet"):
                self.helmet_override = mode
            return self.helmet_override

    def set_demo_state(
        self,
        helmet: Optional[Union[bool, str]] = None,
        left_hand: Optional[str] = None,
        right_hand: Optional[str] = None,
    ) -> Dict[str, object]:
        with self._lock:
            if helmet is not None:
                if isinstance(helmet, bool):
                    self.demo_state["helmet"] = "helmet" if helmet else "no_helmet"
                elif str(helmet) in ("helmet", "special_case", "no_helmet"):
                    self.demo_state["helmet"] = str(helmet)
            if left_hand in ("NONE", "OPEN", "FIST", "INDEX_ONLY"):
                self.demo_state["left_hand"] = left_hand
            if right_hand in ("NONE", "OPEN", "FIST", "INDEX_ONLY"):
                self.demo_state["right_hand"] = right_hand
            return dict(self.demo_state)

    def get_telemetry(self) -> Dict[str, object]:
        with self._lock:
            browser_live = (
                self._browser_seq > 0 or self._browser_frame is not None
            ) and (time.monotonic() - self._browser_frame_ts) < 3.0
            return {
                "system_on": self.system_on,
                "helmet_detected": self.helmet_detected,
                "helmet_score": round(self.helmet_score, 3),
                "yolo_class": self.yolo_class,
                "yolo_conf": round(self.yolo_conf, 3),
                "compliance_status": self.compliance_status,
                "compliance_label": self.compliance_label,
                "mid_ride_buffering": self.mid_ride_buffering,
                "no_helmet_sustained_sec": round(self.no_helmet_sustained_sec, 2),
                "helmet_override": self.helmet_override,
                "face_detected": self.face_detected,
                "left_hand": self.left_hand_state,
                "right_hand": self.right_hand_state,
                "active_gesture": self.active_gesture,
                "index_progress": round(self.index_progress, 3),
                "index_hand": self.index_hand,
                "cursor_active": self.cursor_active,
                "cursor_x": round(self.cursor_x, 4),
                "cursor_y": round(self.cursor_y, 4),
                "head_box": self.head_box_norm,
                "hands_landmarks": self.hands_landmarks,
                "frame_aspect": self.frame_aspect,
                "camera_available": self.camera_available or browser_live,
                "camera_label": self.camera_label,
                "fps": round(self.fps, 1),
                "demo_state": dict(self.demo_state),
            }

    def get_jpeg_frame(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._latest_jpeg, self._frame_seq

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _classify_hand_pose(self, hand_landmarks) -> str:
        """Classify a single 21-point MediaPipe hand as FIST, OPEN, INDEX_ONLY, or NEUTRAL."""
        lm = hand_landmarks.landmark
        wrist = lm[0]

        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            d_wrist_tip = self._dist(wrist, lm[tip_idx])
            d_wrist_pip = self._dist(wrist, lm[pip_idx])
            d_mcp_tip = self._dist(lm[mcp_idx], lm[tip_idx])
            d_mcp_pip = self._dist(lm[mcp_idx], lm[pip_idx])
            return (d_wrist_tip > d_wrist_pip * 1.12) and (d_mcp_tip > d_mcp_pip * 1.08)

        def is_curled(tip_idx: int, pip_idx: int) -> bool:
            return self._dist(wrist, lm[tip_idx]) < self._dist(wrist, lm[pip_idx]) * 1.07

        index_up = is_extended(8, 6, 5)
        middle_up = is_extended(12, 10, 9)
        ring_up = is_extended(16, 14, 13)
        pinky_up = is_extended(20, 18, 17)

        index_curled = is_curled(8, 6)
        middle_curled = is_curled(12, 10)
        ring_curled = is_curled(16, 14)
        pinky_curled = is_curled(20, 18)

        up_count = int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)
        curled_count = (
            int(index_curled) + int(middle_curled) + int(ring_curled) + int(pinky_curled)
        )

        # Dedicated Index-Finger ON/OFF toggle pose (either hand)
        if index_up and middle_curled and ring_curled and pinky_curled:
            return "INDEX_ONLY"

        # Closed Fist: all four fingers curled
        if up_count == 0 and curled_count >= 3:
            return "FIST"

        # Open Palm: at least 3 fingers clearly extended
        if up_count >= 3:
            return "OPEN"

        return "NEUTRAL"

    def _detect_head_box(
        self, frame: np.ndarray, pose_results
    ) -> Tuple[Optional[Tuple[int, int, int, int]], list]:
        """Compute head bounding box (x1, y1, x2, y2) and key head points from Pose + Haar."""
        h, w = frame.shape[:2]
        head_points = []
        pose_box = None

        if pose_results and pose_results.pose_landmarks:
            lms = pose_results.pose_landmarks.landmark
            face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            pts = []
            for idx in face_indices:
                lm = lms[idx]
                if lm.visibility > 0.35 and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pts.append((px, py))
                    if idx in (0, 2, 5, 7, 8):
                        head_points.append((px, py))

            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                span_x = max(max_x - min_x, int(w * 0.14))
                span_y = max(max_y - min_y, int(h * 0.10))
                cx = (min_x + max_x) // 2
                cy = (min_y + max_y) // 2
                half_w = int(max(span_x * 0.85, span_y * 1.15))
                top_y = max(0, cy - int(half_w * 1.35))
                bot_y = min(h - 1, cy + int(half_w * 0.95))
                x1 = max(0, cx - half_w)
                x2 = min(w - 1, cx + half_w)
                pose_box = (x1, top_y, x2, bot_y)

        # Refine/fallback with OpenCV Haar frontal face cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_face = max(24, int(min(w, h) * 0.10))
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.10, minNeighbors=4, minSize=(min_face, min_face)
        )
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            hx1 = max(0, int(fx - fw * 0.22))
            hy1 = max(0, int(fy - fh * 0.55))
            hx2 = min(w - 1, int(fx + fw * 1.22))
            hy2 = min(h - 1, int(fy + fh * 1.10))
            haar_box = (hx1, hy1, hx2, hy2)
            if pose_box is None:
                return haar_box, head_points
            # Blend Pose + Haar for stable crown + chin bounds
            bx1 = min(pose_box[0], haar_box[0])
            by1 = min(pose_box[1], haar_box[1])
            bx2 = max(pose_box[2], haar_box[2])
            by2 = max(pose_box[3], haar_box[3])
            return (bx1, by1, bx2, by2), head_points

        # Fallback: if face landmarks/Haar missed (e.g. dark full-face visor), use Pose shoulders
        if pose_box is None and pose_results and pose_results.pose_landmarks:
            lms = pose_results.pose_landmarks.landmark
            ls, rs = lms[11], lms[12]
            if ls.visibility > 0.40 and rs.visibility > 0.40:
                cx = int(((ls.x + rs.x) * 0.5) * w)
                sy = int(((ls.y + rs.y) * 0.5) * h)
                sh_w = max(int(abs(ls.x - rs.x) * w), int(w * 0.22))
                half_w = int(sh_w * 0.34)
                bot_y = max(24, sy - int(sh_w * 0.04))
                top_y = max(0, bot_y - int(half_w * 2.25))
                x1 = max(0, cx - half_w)
                x2 = min(w - 1, cx + half_w)
                if x2 > x1 + 30 and bot_y > top_y + 30:
                    pose_box = (x1, top_y, x2, bot_y)

        # Fallback: if Haar missed (common for full-face helmets / visors / turbans),
        # inspect upper-center head region if scene has sufficient structure (non-blank wall).
        if pose_box is None:
            ux1, uy1 = int(w * 0.24), int(h * 0.12)
            ux2, uy2 = int(w * 0.76), int(h * 0.82)
            roi = gray[uy1:uy2, ux1:ux2]
            if roi.size > 0 and float(np.std(roi)) > 16.0:
                pose_box = (ux1, uy1, ux2, uy2)

        return pose_box, head_points

    def _classify_head_yolo(
        self, frame: np.ndarray, head_box: Tuple[int, int, int, int]
    ) -> Tuple[str, float]:
        """Run frozen EfficientNet-B0 + balanced linear classifier on full-frame + padded head crop."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = head_box
        bw = x2 - x1
        bh = y2 - y1
        if bw < 28 or bh < 28:
            return "none", 0.0

        # Pad generously upward/sideways to capture full front-view helmet shell & straps
        cx1 = max(0, x1 - int(bw * 0.28))
        cy1 = max(0, y1 - int(bh * 0.32))
        cx2 = min(w, x2 + int(bw * 0.28))
        cy2 = min(h, y2 + int(bh * 0.16))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return "none", 0.0

        pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        probs_batch = predict_linear_probs(
            self._eff_backbone, self._eff_head, [pil_full, pil_crop]
        )
        probs = 0.55 * probs_batch[0] + 0.45 * probs_batch[1]
        pred_idx = int(probs.argmax().item())
        pred_cls = str(self._cls_names[pred_idx])
        conf = float(probs[pred_idx].item())
        return pred_cls, conf

    def _estimate_helmet_score(
        self, frame: np.ndarray, head_box: Tuple[int, int, int, int]
    ) -> float:
        """Analyze crown/forehead skin absence, shell smoothness, and dome edges to score helmet presence."""
        x1, y1, x2, y2 = head_box
        bw = x2 - x1
        bh = y2 - y1
        if bw < 32 or bh < 32:
            return 0.0

        # Upper crown & forehead region (top 46% of head box)
        crown_y2 = y1 + max(12, int(bh * 0.46))
        crown = frame[y1:crown_y2, x1:x2]
        if crown.size == 0:
            return 0.0

        # Lower face region for baseline skin comparison
        face_y1 = y1 + int(bh * 0.42)
        face_y2 = y1 + int(bh * 0.85)
        face_roi = frame[face_y1:face_y2, x1 + int(bw * 0.15) : x2 - int(bw * 0.15)]

        # 1. YCrCb skin mask on crown vs lower face
        ycrcb_crown = cv2.cvtColor(crown, cv2.COLOR_BGR2YCrCb)
        skin_crown = cv2.inRange(ycrcb_crown, (0, 133, 77), (255, 173, 127))
        crown_skin_ratio = float(np.count_nonzero(skin_crown)) / float(skin_crown.size)

        # Bare forehead typically shows > 18-25% skin in crown center; helmet visor/shell < 8%
        mid_c1 = int(crown.shape[1] * 0.22)
        mid_c2 = int(crown.shape[1] * 0.78)
        forehead_strip = skin_crown[int(crown.shape[0] * 0.35) :, mid_c1:mid_c2]
        forehead_skin = (
            float(np.count_nonzero(forehead_strip)) / float(max(1, forehead_strip.size))
        )

        # 2. Hair vs smooth shell texture (Laplacian variance on upper dome)
        gray_crown = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_crown, cv2.CV_64F).var())
        # Smooth shell tends to have lower fine noise OR bright uniform highlight; hair has dense high-freq texture
        smoothness_score = max(0.0, min(1.0, (240.0 - lap_var) / 180.0))

        # 3. Artificial helmet color / brightness contrast vs skin/dark hair
        hsv_crown = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv_crown[:, :, 1]))
        val_mean = float(np.mean(hsv_crown[:, :, 2]))
        # Saturated helmet (red/blue/yellow/green) or bright white/silver helmet
        shell_color_score = 0.0
        if sat_mean > 75 and forehead_skin < 0.16:
            shell_color_score = 0.85
        elif val_mean > 155 and crown_skin_ratio < 0.12:
            shell_color_score = 0.80

        # 4. Dome arc edge symmetry in upper crown
        edges = cv2.Canny(cv2.GaussianBlur(gray_crown, (5, 5), 0), 50, 140)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        dome_score = 1.0 if 0.025 <= edge_density <= 0.14 else 0.35

        # Face skin presence boosts confidence we are looking at a real person
        has_face_skin = True
        if face_roi.size > 0:
            ycrcb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            face_skin = cv2.inRange(ycrcb_face, (0, 133, 77), (255, 173, 127))
            face_skin_ratio = float(np.count_nonzero(face_skin)) / float(face_skin.size)
            has_face_skin = face_skin_ratio > 0.12

        # Forehead occlusion score (low skin in forehead strip -> covered by helmet/visor)
        occlusion_score = max(0.0, min(1.0, (0.20 - forehead_skin) / 0.16))

        score = (
            0.46 * occlusion_score
            + 0.24 * smoothness_score
            + 0.20 * shell_color_score
            + 0.10 * dome_score
        )
        if not has_face_skin and shell_color_score < 0.5:
            # Full-face dark helmet or low lighting: still allow high occlusion + smoothness
            score *= 0.90

        return max(0.0, min(1.0, score))

    def _draw_hand_skeleton(
        self, frame: np.ndarray, hand_landmarks, hand_side: str, pose_label: str
    ) -> None:
        h, w = frame.shape[:2]
        if pose_label == "FIST":
            line_col = (255, 165, 30)
            dot_col = (0, 215, 255)
        elif pose_label == "OPEN":
            line_col = (80, 240, 140)
            dot_col = (255, 255, 255)
        elif pose_label == "INDEX_ONLY":
            line_col = (255, 220, 40)
            dot_col = (255, 80, 240)
        else:
            line_col = (190, 210, 230)
            dot_col = (140, 220, 255)

        self._mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=dot_col, thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=line_col, thickness=2, circle_radius=2),
        )

        # Wrist badge ("L: FIST" / "R: OPEN" / "L: INDEX")
        wrist = hand_landmarks.landmark[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        tag = f"{hand_side[0]}:{pose_label.replace('_ONLY', '')}"
        tx = max(8, min(w - 92, wx - 35))
        ty = max(22, min(h - 10, wy + 24))
        cv2.rectangle(frame, (tx - 4, ty - 15), (tx + 82, ty + 5), (15, 22, 32), -1)
        cv2.rectangle(frame, (tx - 4, ty - 15), (tx + 82, ty + 5), line_col, 1)
        cv2.putText(
            frame, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_col, 1, cv2.LINE_AA
        )

    @staticmethod
    def _draw_corner_box(
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        x1, y1, x2, y2 = box
        corner = max(14, min(28, (x2 - x1) // 5))
        t = 2
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, t, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, t, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, t, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, t, cv2.LINE_AA)

        # Label pill above box
        ly = max(22, y1 - 8)
        pill_w = max(124, len(label) * 8 + 14)
        cv2.rectangle(frame, (x1, ly - 17), (x1 + pill_w, ly + 4), (15, 22, 32), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

    def _render_demo_frame(self, t: float) -> Tuple[np.ndarray, str, str, str]:
        """Render a clean synthetic selfie frame when no physical webcam is attached."""
        w, h = 480, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient backdrop
        for row in range(h):
            val = int(28 + 20 * (row / h))
            frame[row, :] = (val + 14, val + 6, val)

        with self._lock:
            raw_demo = self.demo_state["helmet"]
            if isinstance(raw_demo, bool):
                demo_cls = "helmet" if raw_demo else "no_helmet"
            else:
                demo_cls = str(raw_demo)
            demo_l = str(self.demo_state["left_hand"])
            demo_r = str(self.demo_state["right_hand"])

        cx, cy = w // 2, int(h * 0.43)
        # Shoulders & jacket
        cv2.ellipse(frame, (cx, h + 15), (150, 115), 0, 180, 360, (48, 52, 62), -1)
        cv2.ellipse(frame, (cx, h + 15), (150, 115), 0, 180, 360, (80, 95, 115), 2)

        # Neck & Face
        cv2.rectangle(frame, (cx - 22, cy + 45), (cx + 22, cy + 95), (120, 150, 190), -1)
        cv2.ellipse(frame, (cx, cy + 8), (52, 66), 0, 0, 360, (138, 172, 212), -1)
        # Eyes & mask/smile
        cv2.circle(frame, (cx - 18, cy - 4), 5, (35, 35, 40), -1)
        cv2.circle(frame, (cx + 18, cy - 4), 5, (35, 35, 40), -1)
        cv2.ellipse(frame, (cx, cy + 32), (26, 20), 0, 0, 180, (40, 44, 52), -1)

        if demo_cls == "helmet":
            # Sleek matte-cyan & dark-graphite motorcycle helmet dome
            cv2.ellipse(frame, (cx, cy - 8), (74, 78), 0, 175, 365, (45, 50, 58), -1)
            cv2.ellipse(frame, (cx, cy - 8), (74, 78), 0, 175, 365, (0, 225, 255), 3)
            # Visor rim
            cv2.ellipse(frame, (cx, cy - 6), (64, 34), 0, 185, 355, (25, 30, 38), -1)
            cv2.line(frame, (cx - 64, cy - 8), (cx + 64, cy - 8), (0, 230, 255), 2)
        elif demo_cls == "special_case":
            # Saffron / royal-blue turban / pagri wrap
            cv2.ellipse(frame, (cx, cy - 30), (72, 54), 0, 175, 365, (30, 130, 235), -1)
            cv2.ellipse(frame, (cx, cy - 34), (66, 42), -12, 180, 360, (25, 165, 250), 2)
            cv2.ellipse(frame, (cx, cy - 34), (66, 42), 12, 180, 360, (25, 165, 250), 2)
        else:
            # Hair
            cv2.ellipse(frame, (cx, cy - 44), (54, 32), 0, 180, 360, (32, 32, 38), -1)

        # Head box overlay
        box = (cx - 82, cy - 94, cx + 82, cy + 84)
        if demo_cls == "helmet":
            col, lbl = (80, 245, 140), "HELMET DETECTED (0.95)"
        elif demo_cls == "special_case":
            col, lbl = (80, 245, 140), "TURBAN EXEMPT (0.94)"
        else:
            col, lbl = (70, 90, 255), "NO HELMET (0.92)"
        self._draw_corner_box(frame, box, col, lbl)

        # Draw simulated 21-point hand skeletons for left/right hands
        for side, state, hx in (("LEFT", demo_l, 95), ("RIGHT", demo_r, w - 95)):
            if state == "NONE":
                continue
            hy = int(h * 0.72 + math.sin(t * 3.0) * 4)
            self._draw_synthetic_hand(frame, hx, hy, side, state)

        cv2.putText(
            frame,
            "DEMO CAM (NO WEBCAM FOUND)",
            (118, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (170, 200, 220),
            1,
            cv2.LINE_AA,
        )
        return frame, demo_l, demo_r, demo_cls

    @staticmethod
    def _draw_synthetic_hand(
        frame: np.ndarray, cx: int, cy: int, side: str, state: str
    ) -> None:
        col = (255, 175, 40) if state == "FIST" else (80, 240, 145)
        if state == "INDEX_ONLY":
            col = (255, 220, 50)
        wrist = (cx, cy + 32)
        cv2.circle(frame, wrist, 5, (255, 255, 255), -1)

        offsets = [-24, -8, 8, 22]
        for idx, ox in enumerate(offsets):
            mcp = (cx + ox, cy + 2)
            cv2.line(frame, wrist, mcp, col, 2, cv2.LINE_AA)
            cv2.circle(frame, mcp, 3, (255, 255, 255), -1)
            extended = state == "OPEN" or (state == "INDEX_ONLY" and idx == 1)
            tip = (cx + ox, cy - (44 if extended else -8))
            mid = (cx + ox, cy - (20 if extended else -4))
            cv2.line(frame, mcp, mid, col, 2, cv2.LINE_AA)
            cv2.line(frame, mid, tip, col, 2, cv2.LINE_AA)
            cv2.circle(frame, mid, 3, col, -1)
            cv2.circle(frame, tip, 4, (0, 230, 255), -1)

        cv2.putText(
            frame,
            f"{side[0]}:{state.replace('_ONLY', '')}",
            (cx - 28, cy + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            col,
            1,
            cv2.LINE_AA,
        )

    def _center_square_crop(self, frame: np.ndarray, size: int = 380) -> np.ndarray:
        h, w = frame.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        cropped = frame[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    def _loop(self) -> None:
        self._open_camera()
        last_t = time.perf_counter()
        fps_smooth = 15.0
        last_browser_seq = -1
        black_frames = 0

        while self._running:
            if not self.camera_available:
                self._browser_event.wait(timeout=0.25)
                self._browser_event.clear()

            now = time.perf_counter()

            left_state = "NONE"
            right_state = "NONE"
            face_found = False
            helmet_now = False
            yolo_cls = "none"
            yolo_conf = 0.0
            comp_status = "NO_HEAD"
            comp_label = "NO HELMET DETECTED"
            cursor_active = False
            raw_cursor_xy: Optional[Tuple[float, float]] = None
            head_box_norm: Optional[Dict[str, object]] = None
            hands_lms_payload: list = []
            frame_aspect = 4.0 / 3.0
            processed_seq_now: Optional[int] = None

            raw_frame = None
            using_live = False
            is_browser_cam = False

            if self.camera_available and self._cap is not None:
                ok, cam_frame = self._cap.read()
                if ok and cam_frame is not None:
                    if float(cam_frame.mean()) < 2.0 and float(cam_frame.std()) < 2.0:
                        black_frames += 1
                        if black_frames >= 12:
                            print("[VisionTracker] OpenCV camera returning black frames; releasing for Browser Cam fallback")
                            self._cap.release()
                            self._cap = None
                            self.camera_available = False
                            self.camera_label = "BROWSER / CLOUD CAM"
                            black_frames = 0
                    else:
                        black_frames = 0
                        raw_frame = cam_frame
                        using_live = True
                else:
                    self.camera_available = False
                    self.camera_label = "BROWSER / CLOUD CAM"
            else:
                with self._lock:
                    b_seq = self._browser_seq
                    b_raw = self._browser_jpeg_raw
                    b_ts = self._browser_frame_ts
                    b_img = self._browser_frame
                if (time.monotonic() - b_ts) < 3.0 and (b_raw is not None or b_img is not None):
                    if b_seq == last_browser_seq:
                        # No new browser frame yet; skip redundant AI inference on Render 0.1 CPU
                        continue
                    last_browser_seq = b_seq
                    if b_raw is not None:
                        arr = np.frombuffer(b_raw, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            raw_frame = decoded
                    elif b_img is not None:
                        raw_frame = b_img.copy()
                    if raw_frame is not None:
                        using_live = True
                        is_browser_cam = True
                        processed_seq_now = b_seq
                        self.camera_label = "LIVE BROWSER CAM"

            dt = max(0.001, min(1.5, now - last_t))
            last_t = now
            fps_smooth = 0.85 * fps_smooth + 0.15 * min(60.0, 1.0 / dt)

            if using_live and raw_frame is not None:
                # Mirror selfie frame for intuitive PiP & steering
                frame = cv2.flip(raw_frame, 1)
                fh, fw = frame.shape[:2]
                frame_aspect = round(fw / max(1, fh), 4)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. OpenCV Haar Face -> Head Box & Local YOLO Classification
                head_box, head_pts = self._detect_head_box(frame, None)
                if head_box is not None:
                    face_found = True
                    mono_now = time.monotonic()
                    if (mono_now - self._last_yolo_ts) >= 0.40 or self._last_yolo_res[0] == "none":
                        self._last_yolo_res = self._classify_head_yolo(frame, head_box)
                        self._last_yolo_ts = mono_now
                    yolo_cls, yolo_conf = self._last_yolo_res

                    with self._lock:
                        ov = self.helmet_override
                    if ov == "helmet":
                        yolo_cls, yolo_conf = "helmet", 0.96
                    elif ov == "special_case":
                        yolo_cls, yolo_conf = "special_case", 0.95
                    elif ov == "no_helmet":
                        yolo_cls, yolo_conf = "no_helmet", 0.95

                    if yolo_cls == "helmet" and yolo_conf >= 0.60:
                        helmet_now = True
                        comp_status = "HELMET_VERIFIED"
                        comp_label = "HELMET DETECTED"
                        box_col = (80, 245, 140)
                        box_lbl = f"HELMET ({yolo_conf:.2f})"
                    elif yolo_cls == "special_case" and yolo_conf >= 0.60:
                        helmet_now = True
                        comp_status = "TURBAN_EXEMPT"
                        comp_label = "TURBAN / PAGRI EXEMPT"
                        box_col = (80, 245, 140)
                        box_lbl = f"TURBAN ({yolo_conf:.2f})"
                    else:
                        helmet_now = False
                        comp_status = "NO_HELMET"
                        comp_label = "NO HELMET DETECTED"
                        box_col = (65, 85, 255)
                        box_lbl = f"NO HELMET ({yolo_conf:.2f})"

                    self._helmet_ema = 0.75 * self._helmet_ema + 0.25 * (
                        yolo_conf if helmet_now else (1.0 - yolo_conf)
                    )
                    head_box_norm = {
                        "x1": round(head_box[0] / max(1, fw), 4),
                        "y1": round(head_box[1] / max(1, fh), 4),
                        "x2": round(head_box[2] / max(1, fw), 4),
                        "y2": round(head_box[3] / max(1, fh), 4),
                        "label": box_lbl,
                        "compliant": helmet_now,
                    }
                    self._draw_corner_box(frame, head_box, box_col, box_lbl)
                    for px, py in head_pts:
                        cv2.circle(frame, (px, py), 3, (0, 235, 255), -1)
                else:
                    self._helmet_ema *= 0.90
                    with self._lock:
                        ov = self.helmet_override
                    if ov == "helmet":
                        helmet_now = True
                        yolo_cls, yolo_conf = "helmet", 0.96
                        comp_status, comp_label = "HELMET_VERIFIED", "HELMET DETECTED"
                    elif ov == "special_case":
                        helmet_now = True
                        yolo_cls, yolo_conf = "special_case", 0.95
                        comp_status, comp_label = "TURBAN_EXEMPT", "TURBAN / PAGRI EXEMPT"
                    else:
                        helmet_now = False
                        yolo_cls, yolo_conf = "none", 0.0
                        comp_status, comp_label = "NO_HEAD", "NO RIDER HEAD DETECTED"

                # 2. MediaPipe Two-Hand Tracking (21-point landmarks)
                hand_res = self._hands.process(rgb)
                if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
                    detected = []
                    for lms, handed in zip(
                        hand_res.multi_hand_landmarks, hand_res.multi_handedness
                    ):
                        label = handed.classification[0].label.upper()  # LEFT or RIGHT
                        wrist_x = lms.landmark[0].x
                        pose_lbl = self._classify_hand_pose(lms)
                        detected.append((wrist_x, label, lms, pose_lbl))

                    # If 2 hands detected, disambiguate cleanly by screen X in mirrored selfie
                    if len(detected) == 2:
                        detected.sort(key=lambda item: item[0])
                        detected[0] = (detected[0][0], "LEFT", detected[0][2], detected[0][3])
                        detected[1] = (detected[1][0], "RIGHT", detected[1][2], detected[1][3])

                    for _, side, lms, pose_lbl in detected:
                        if side == "LEFT":
                            left_state = pose_lbl
                        else:
                            right_state = pose_lbl
                        hands_lms_payload.append(
                            {
                                "side": side,
                                "pose": pose_lbl,
                                "points": [
                                    [round(pt.x, 3), round(pt.y, 3)]
                                    for pt in lms.landmark
                                ],
                            }
                        )
                        self._draw_hand_skeleton(frame, lms, side, pose_lbl)
                        if pose_lbl == "INDEX_ONLY" and raw_cursor_xy is None:
                            tip = lms.landmark[8]
                            # Ergonomic edge amplification so user easily reaches screen corners
                            nx = max(0.0, min(1.0, (tip.x - 0.14) / 0.72))
                            ny = max(0.0, min(1.0, (tip.y - 0.14) / 0.72))
                            raw_cursor_xy = (nx, ny)
                            fh, fw = frame.shape[:2]
                            cv2.circle(
                                frame,
                                (int(tip.x * fw), int(tip.y * fh)),
                                10,
                                (255, 230, 40),
                                2,
                                cv2.LINE_AA,
                            )

                display_frame = None if is_browser_cam else self._center_square_crop(frame, 360)
            else:
                # Synthetic demo camera frame when no webcam is attached
                display_frame, left_state, right_state, demo_cls = self._render_demo_frame(
                    now
                )
                face_found = True
                with self._lock:
                    ov = self.helmet_override
                if ov in ("helmet", "special_case", "no_helmet"):
                    demo_cls = ov
                yolo_cls = demo_cls
                yolo_conf = 0.95 if demo_cls != "no_helmet" else 0.92
                if demo_cls == "helmet":
                    helmet_now = True
                    comp_status, comp_label = "HELMET_VERIFIED", "HELMET DETECTED"
                elif demo_cls == "special_case":
                    helmet_now = True
                    comp_status, comp_label = "TURBAN_EXEMPT", "TURBAN / PAGRI EXEMPT"
                else:
                    helmet_now = False
                    comp_status, comp_label = "NO_HELMET", "NO HELMET DETECTED"
                self._helmet_ema = yolo_conf
                if left_state == "INDEX_ONLY" or right_state == "INDEX_ONLY":
                    raw_cursor_xy = (0.075, 0.075)

            # Mid-ride sustained non-compliance tracker (> 2.0s triggers progressive speed buffering)
            with self._lock:
                is_on = self.system_on
            if is_on:
                if not helmet_now:
                    self._no_helmet_timer += dt
                else:
                    self._no_helmet_timer = 0.0
                mid_buffering = self._no_helmet_timer > 2.0
            else:
                self._no_helmet_timer = 0.0
                mid_buffering = False

            # 3. Index-Finger Cursor Tracking (frontend triggers ON/OFF only when cursor is on ignition button)
            idx_hand = "NONE"
            if left_state == "INDEX_ONLY" and right_state != "FIST":
                idx_hand = "LEFT"
            elif right_state == "INDEX_ONLY" and left_state != "FIST":
                idx_hand = "RIGHT"

            if idx_hand != "NONE" and raw_cursor_xy is not None:
                cursor_active = True
                self._cursor_x_ema = 0.62 * self._cursor_x_ema + 0.38 * raw_cursor_xy[0]
                self._cursor_y_ema = 0.62 * self._cursor_y_ema + 0.38 * raw_cursor_xy[1]

            idx_prog = 0.0

            # 4. Combined Two-Hand Riding Gesture
            if idx_hand != "NONE":
                gesture = "INDEX_TOGGLE"
            elif left_state == "FIST" and right_state == "FIST":
                gesture = "BOTH_FISTS"
            elif left_state == "FIST":
                gesture = "LEFT_FIST"
            elif right_state == "FIST":
                gesture = "RIGHT_FIST"
            elif left_state == "OPEN" and right_state == "OPEN":
                gesture = "BOTH_OPEN"
            else:
                gesture = "NONE"

            ok_enc = False
            buf = None
            if display_frame is not None:
                ok_enc, buf = cv2.imencode(
                    ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )

            with self._lock:
                self.face_detected = face_found
                self.helmet_detected = helmet_now
                self.helmet_score = self._helmet_ema
                self.yolo_class = yolo_cls
                self.yolo_conf = yolo_conf
                self.compliance_status = comp_status
                self.compliance_label = comp_label
                self.mid_ride_buffering = mid_buffering
                self.no_helmet_sustained_sec = self._no_helmet_timer
                self.left_hand_state = left_state
                self.right_hand_state = right_state
                self.active_gesture = gesture
                self.index_progress = idx_prog
                self.index_hand = idx_hand
                self.cursor_active = cursor_active
                self.cursor_x = self._cursor_x_ema
                self.cursor_y = self._cursor_y_ema
                self.head_box_norm = head_box_norm
                self.hands_landmarks = hands_lms_payload
                self.frame_aspect = frame_aspect
                self.fps = fps_smooth
                if processed_seq_now is not None:
                    self._processed_browser_seq = processed_seq_now
                if ok_enc and buf is not None:
                    self._latest_jpeg = buf.tobytes()
                    self._frame_seq += 1

            if processed_seq_now is not None:
                self._processed_event.set()

            if self.camera_available:
                time.sleep(0.02)
