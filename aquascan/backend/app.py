"""AquaScan AI backend: detect -> shadow-aware rescoring -> geotag."""
import io, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AquaScan AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
_yolo = None
def get_yolo():
    global _yolo
    if _yolo is None and os.path.exists(MODEL_PATH):
        try:
            from ultralytics import YOLO
            _yolo = YOLO(MODEL_PATH)
        except Exception as e:
            print("YOLO unavailable, using heuristic detector:", e)
            _yolo = False
    return _yolo or None

def sigmoid(x): return 1 / (1 + math.exp(-x))

def shadow_features(g, x0, y0, x1, y1):
    """Shadow lies on the side of the object away from nadir (image centre)."""
    H, W = g.shape
    w = max(x1 - x0, 4)
    on_right = (x0 + x1) / 2 >= W / 2
    sx0, sx1 = (x1, min(W, x1 + w)) if on_right else (max(0, x0 - w), x0)
    if sx1 - sx0 < 2: return 0.0, 0.0
    strip = g[y0:y1, sx0:sx1]
    ring = g[max(0, y0 - 10):min(H, y1 + 10), max(0, sx0 - 30):min(W, sx1 + 30)]
    darkness = float(np.clip(1 - strip.mean() / (np.median(ring) + 1e-6), 0, 1))
    gx = np.abs(np.diff(strip.mean(axis=0))) if strip.shape[1] > 1 else np.zeros(1)
    sharp = float(np.clip(gx.max() / (np.median(ring) + 1e-6) * 3, 0, 1))
    return darkness, sharp

def heuristic_detect(g):
    """Fallback detector: bright echo blobs. Replace by YOLO weights for real use."""
    hi = g > np.percentile(g, 97)
    hi = ndi.binary_closing(ndi.binary_opening(hi, iterations=1), iterations=2)
    lab, n = ndi.label(hi)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        y, x = sl
        if (lab[sl] == i).sum() < 40: continue
        out.append((x.start, y.start, x.stop, y.stop, 0.5, "object"))
    return out

def detect(img):
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    H, W = g.shape
    model = get_yolo()
    if model:
        r = model.predict(img.convert("RGB"), conf=0.1, verbose=False)[0]
        raw = [(*map(int, b.xyxy[0].tolist()), float(b.conf[0]), r.names[int(b.cls[0])]) for b in r.boxes]
        source = "yolo"
    else:
        raw, source = heuristic_detect(g), "heuristic"
    dets = []
    for x0, y0, x1, y1, base, cls in raw:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0: continue
        dark, sharp = shadow_features(g, x0, y0, x1, y1)
        bw, bh = x1 - x0, y1 - y0
        elong = max(bw, bh) / max(1, min(bw, bh))
        score = -2.0 + 2.0 * dark + 2.0 * sharp + 0.5 * min(elong, 4) + (2.0 * (base - 0.5) if source == "yolo" else 0)
        conf = sigmoid(score)
        reason = f"shadow darkness {dark:.2f}, edge sharpness {sharp:.2f}, elongation {elong:.1f}"
        label = "debris" if conf >= 0.5 else "natural (likely rock)"
        dets.append(dict(box=[x0, y0, x1, y1], confidence=round(conf, 3), label=label,
                         model_class=cls, reason=reason, source=source))
    return dets, (W, H)

def geotag(px, py, W, H, lat0, lon0, heading, swath_m, length_m):
    across = (px - W / 2) * (swath_m / W)      # + = starboard
    along = py * (length_m / H)
    th = math.radians(heading)
    dE = along * math.sin(th) + across * math.cos(th)
    dN = along * math.cos(th) - across * math.sin(th)
    lat = lat0 + dN / 111_320
    lon = lon0 + dE / (111_320 * math.cos(math.radians(lat0)))
    return round(lat, 6), round(lon, 6)

@app.get("/api/health")
def health(): return {"ok": True, "model": bool(get_yolo())}

@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...), lat0: float = Form(...), lon0: float = Form(...),
                     heading: float = Form(0), swath_m: float = Form(100), length_m: float = Form(200),
                     min_conf: float = Form(0.5)):
    if not (-90 <= lat0 <= 90 and -180 <= lon0 <= 180): raise HTTPException(422, "Invalid lat/lon")
    try: img = Image.open(io.BytesIO(await file.read()))
    except Exception: raise HTTPException(400, "Could not read image. Upload PNG/JPG (XTF/JSF: convert first).")
    dets, (W, H) = detect(img)
    res = []
    for d in dets:
        if d["confidence"] < min_conf: continue
        x0, y0, x1, y1 = d["box"]
        d["lat"], d["lon"] = geotag((x0 + x1) / 2, (y0 + y1) / 2, W, H, lat0, lon0, heading, swath_m, length_m)
        res.append(d)
    res.sort(key=lambda d: -d["confidence"])
    return {"width": W, "height": H, "count": len(res), "detections": res}

if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
