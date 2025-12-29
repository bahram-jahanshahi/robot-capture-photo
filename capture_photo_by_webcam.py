import time
from io import BytesIO

import cv2
from fastapi import FastAPI, HTTPException, Response, Query

app = FastAPI(title="Webcam Capture API", version="1.0.0")


def capture_frame(device_index: int = 0, width: int | None = None, height: int | None = None):
    # CAP_AVFOUNDATION works best on macOS; if it fails, OpenCV will fallback if you remove it.
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail=f"Cannot open camera device_index={device_index}")

    # Warm up camera a bit so exposure/auto-focus stabilizes
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")

    return frame


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/capture", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
def capture(
    device_index: int = Query(0, ge=0),
    width: int | None = Query(None, ge=1),
    height: int | None = Query(None, ge=1),
    quality: int = Query(90, ge=1, le=100),
):
    frame = capture_frame(device_index=device_index, width=width, height=height)

    # Encode to JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode JPEG")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
