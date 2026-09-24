import io, sys, numpy as np
from PIL import Image
sys.path.insert(0, "backend")
from fastapi.testclient import TestClient
from app import app, geotag
rng = np.random.default_rng(0)
g = rng.normal(0.35, 0.05, (400, 400)).clip(0, 1)
g[100:106, 300:360] = 0.95; g[106:112, 360:420 - 20] = 0.02     # bright net-like bar + sharp shadow
g[250:270, 60:80] = 0.9;    g[250:270, 40:60] = g[250:270, 40:60] * 0.9   # rock, weak shadow
buf = io.BytesIO(); Image.fromarray((g * 255).astype("uint8")).save(buf, "PNG")
c = TestClient(app)
assert c.get("/api/health").json()["ok"]
r = c.post("/api/detect", files={"file": ("s.png", buf.getvalue(), "image/png")},
           data=dict(lat0=20.3, lon0=86.7, heading=45, min_conf=0.0)); print(r.status_code, r.json()["count"])
for d in r.json()["detections"]: print(d["box"], d["confidence"], d["label"], d["lat"], d["lon"])
assert c.post("/api/detect", files={"file": ("x.png", b"junk", "image/png")}, data=dict(lat0=1, lon0=1)).status_code == 400
assert c.post("/api/detect", files={"file": ("s.png", buf.getvalue(), "image/png")}, data=dict(lat0=99, lon0=1)).status_code == 422
lat, lon = geotag(200, 0, 400, 400, 10, 10, 90, 100, 200); assert abs(lat - 10) < 1e-6 and abs(lon - 10) < 1e-6
lat, lon = geotag(200, 400, 400, 400, 10, 10, 0, 100, 200); assert abs(lat - (10 + 200 / 111320)) < 1e-6
print("ALL TESTS PASSED")
