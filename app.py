import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vision_tracker import VisionTracker

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

tracker = VisionTracker(camera_index=0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    tracker.start()
    try:
        yield
    finally:
        tracker.stop()


app = FastAPI(title="MotoSafe AI - Helmet & Gesture Riding Simulator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ToggleRequest(BaseModel):
    state: Optional[bool] = None
    image_b64: Optional[str] = None


class ComplianceScanRequest(BaseModel):
    image_b64: Optional[str] = None


class HelmetOverrideRequest(BaseModel):
    mode: str = "auto"


class DemoStateRequest(BaseModel):
    helmet: Optional[Union[bool, str]] = None
    left_hand: Optional[str] = None
    right_hand: Optional[str] = None


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.get("/")
async def index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/api/state")
async def get_state():
    return JSONResponse(tracker.get_telemetry())


def _decode_b64_image(image_b64: Optional[str]) -> Optional[bytes]:
    if not image_b64:
        return None
    raw = image_b64.split(",", 1)[-1]
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


@app.post("/api/toggle")
async def toggle_system(body: Optional[ToggleRequest] = None):
    target = body.state if body is not None else None
    want_on = (not tracker.system_on) if target is None else bool(target)
    if want_on:
        jpg_bytes = _decode_b64_image(body.image_b64 if body is not None else None)
        await run_in_threadpool(tracker.analyze_snapshot, jpg_bytes, False)
    result = tracker.attempt_ignite(target)
    result["telemetry"] = tracker.get_telemetry()
    return JSONResponse(result)


@app.post("/api/compliance_scan")
async def compliance_scan(body: Optional[ComplianceScanRequest] = None):
    jpg_bytes = _decode_b64_image(body.image_b64 if body is not None else None)
    telemetry = await run_in_threadpool(tracker.analyze_snapshot, jpg_bytes, True)
    return JSONResponse({"ok": True, "telemetry": telemetry})


@app.post("/api/helmet_override")
async def helmet_override(body: HelmetOverrideRequest):
    mode = tracker.set_helmet_override(body.mode)
    return JSONResponse({"helmet_override": mode})


@app.post("/api/demo")
async def update_demo(body: DemoStateRequest):
    demo = tracker.set_demo_state(
        helmet=body.helmet,
        left_hand=body.left_hand,
        right_hand=body.right_hand,
    )
    return JSONResponse({"demo_state": demo})


@app.post("/api/retry_camera")
async def retry_camera():
    ok = tracker.retry_camera()
    return JSONResponse({"camera_available": ok, "camera_label": tracker.camera_label})


@app.post("/api/upload_frame")
async def upload_frame(request: Request):
    data = await request.body()
    ok = await run_in_threadpool(tracker.push_browser_frame_and_wait, data, 1.2)
    return JSONResponse({"ok": ok, "telemetry": tracker.get_telemetry()})


@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        last_seq = -1
        while True:
            jpeg, seq = tracker.get_jpeg_frame()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream")
async def telemetry_stream():
    async def event_generator():
        while True:
            payload = json.dumps(tracker.get_telemetry())
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.10)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("app:app", host=host, port=port, reload=False)
