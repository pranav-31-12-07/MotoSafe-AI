# MotoSafe AI — Helmet-Safety & Gesture-Controlled Motorcycle Simulator

**Developed by:** Pranav Shah

MotoSafe AI is a real-time computer vision and gesture-controlled motorcycle safety simulator built with **FastAPI**, **OpenCV**, **MediaPipe**, and a custom-trained **Ultralytics YOLO11n-cls** model. It demonstrates an intelligent **Pre-Ride Ignition Gate**, culturally inclusive headgear compliance (supporting both helmets and turbans/pagris), progressive mid-ride speed buffering, and hands-free gesture riding simulation.

---

## Problem Statement

Two-wheeler riders account for a large share of road traffic fatalities, largely due to riding without a helmet or removing a helmet mid-journey after starting the vehicle. At the same time:
- Standard helmet checks often lack real-time, tamper-resistant enforcement before ignition.
- Abruptly cutting engine power mid-ride when a violation occurs poses a severe traffic hazard.
- Automated enforcement systems must also respect cultural and religious exemptions (such as riders wearing a **turban / pagri**) without triggering false non-compliance blocks.

---

## Proposed Solution

MotoSafe AI enforces two-wheeler safety across the full riding lifecycle using purely local, on-device vision models (zero third-party APIs):

1. **Pre-Ride Ignition Gate (`IGNITE`)**
   - When the rider attempts to start the vehicle (via mouse click or hovering the **Index-Finger Air Cursor** over the **IGNITE** button), the backend checks the live YOLO classification on the cropped head region.
   - **Compliant (`helmet` or `special_case` turban/pagri, confidence $\ge 0.60$):** Ignition unlocks (`IGNITION ON - HELMET VERIFIED` / `IGNITION ON - TURBAN EXEMPT`) and full throttle up to `85 km/h` is enabled.
   - **Non-Compliant (`no_helmet` or no head verified):** Ignition remains locked at `0 km/h`, triggers a Web Audio alert beep, and displays **`"Helmet not detected! Vehicle ignition prohibited!"`**.

2. **Mid-Ride Progressive Speed Buffering**
   - Continuous YOLO monitoring runs while riding. If the helmet is removed mid-ride (`no_helmet` sustained for $> 2.0\text{ s}$), the system never kills the engine abruptly—instead, it triggers **`HELMET REMOVED - PROGRESSIVE SPEED BUFFERING`** and smoothly caps/decelerates speed to a safe **`20 km/h`** until compliant headgear is detected again.

3. **Periodic Auto-Capture Compliance Check (5-Min Real-World / 1-Min Testing)**
   - **Real-Life Scenario:** In real-world deployment, the onboard camera automatically captures and verifies a rider compliance image every **5 minutes**.
   - **Testing & Demo Mode:** For rapid testing and live demonstrations, the periodic compliance scan triggers every **1 minute (`01:00`)**, and includes a **`DEMO 10x SPEED`** toggle button on the HUD to trigger scans every **10 seconds**.

4. **Interactive POV Riding Simulator & Two-Hand Gesture Control**
   - Real-time circular Picture-in-Picture (PiP) selfie feed in the top-right corner with MediaPipe head/face bounding box and 21-point hand skeleton overlays.
   - **Gestures:**
     - `LEFT FIST` $\rightarrow$ Steer / lane-shift left
     - `RIGHT FIST` $\rightarrow$ Steer / lane-shift right
     - `BOTH FISTS` $\rightarrow$ Brake smoothly toward `0 km/h`
     - `BOTH OPEN` $\rightarrow$ Accelerate smoothly up to cruising speed
     - `INDEX FINGER` (left or right hand) $\rightarrow$ Moves an on-screen air cursor to toggle **IGNITE** ON/OFF
   - Urban Indian 3-lane road canvas with animated lane markings, moving traffic (auto-rickshaws, hatchbacks, SUVs, sedans), tilting motorcycle handlebar POV, and a glowing cyan digital speedometer.

---

## Tech Stack Used

- **Backend & Streaming:** Python 3.12, FastAPI, Uvicorn, Pydantic, Server-Sent Events (`/stream`), MJPEG (`/video_feed`)
- **Computer Vision & Deep Learning:**
  - **Ultralytics YOLO (`yolo11n-cls.pt` $\rightarrow$ `best.pt`)** & PyTorch for local 3-class headgear classification
  - **MediaPipe** (`Hands` 21-point two-hand landmark tracking + `Pose` head/shoulder tracking)
  - **OpenCV** (`VideoCapture`, Haar Frontal Face Cascade, image cropping, skeleton & HUD overlays)
- **Frontend UI & Simulation:** HTML5 Canvas, Vanilla JavaScript (ES Modules), CSS3, Web Audio API (alert beeps), `getUserMedia` browser webcam fallback

---

## Training Information

The custom classification model is trained locally via `train_model.py` (and automatically trained on startup by `vision_tracker.py` if `best.pt` is missing):

- **Base Model:** `YOLO("yolo11n-cls.pt")` (86 layers, 1,534,947 parameters, 3.3 GFLOPs)
- **Hyperparameters:** `epochs=35`, `imgsz=224`, `batch=16`, optimizer `AdamW`
- **Dataset (`datasets/train` & `datasets/val`, 3 classes):**
  - `helmet`: 15 training images / 5 validation images
  - `no_helmet`: 31 training images / 17 validation images
  - `special_case` (Turban / Pagri): 7 training images / 4 validation images
- **Validation Performance (`datasets/val`, 26 images):**
  - `helmet`: **5/5 (100.00%)**
  - `no_helmet`: **17/17 (100.00%)**
  - `special_case`: **1/4 (25.00%)** *(can be improved further by adding more turban/pagri samples to `datasets/train/special_case`)*
  - **Overall Top-1 Accuracy:** **23/26 (88.46%)**
  - **Inference Speed:** ~2.5 ms per image on CPU

---

## Feasibility & Viability

- **Technical Feasibility (Edge-Ready & Offline):** The trained `best.pt` weights are only ~3.2 MB and run at ~2.5 ms per crop on a standard CPU without cloud APIs or GPU dependency, making deployment viable on low-cost edge hardware (e.g., Raspberry Pi, Jetson Nano, or smart dashcam modules).
- **Safety & Operational Viability:** Enforcing the safety lock **before ignition** prevents unsafe rides from starting, while **progressive speed buffering (`20 km/h` cap)** instead of sudden mid-ride engine cutoff avoids dangerous sudden stalls in live traffic.
- **Cultural & Regulatory Viability:** Explicitly modeling `special_case` (turban/pagri exemption) aligns with regional traffic laws and ensures fair, inclusive adoption across Indian roadways.
- **Commercial Scalability:** Ideal for OEM smart two-wheelers, EV scooter digital clusters, fleet/delivery rider compliance (e-commerce & food delivery), and two-wheeler rental platforms.

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py           # Starts FastAPI server at http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in your browser and allow camera access if prompted.
