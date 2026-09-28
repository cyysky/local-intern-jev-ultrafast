"""Smoke test for internlm/Intern-Decision-4B using the checkpoint's own inference.py.

Usage:  python smoke_test.py [cuda:0|cuda:1|cpu]

Runs three checks:
  1. text-only structured decision (model-card example)
  2. multimodal decision on a generated image
  3. red-vs-green A/B contrast proving the vision path drives the output
"""
import json
import sys
import time
from pathlib import Path

MODEL_DIR = Path(r"D:\models\Intern-Decision-4B")
sys.path.insert(0, str(MODEL_DIR))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from inference import DecisionEngine  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

t0 = time.perf_counter()
engine = DecisionEngine(checkpoint=MODEL_DIR, device=DEVICE)
print(f"[load] {time.perf_counter() - t0:.1f}s on {DEVICE}")
print(f"[mem]  {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated")

# ---------- 1. text-only ----------
text_request = {
    "state": "The customer was charged twice and asks for the extra payment back.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments and refunds", "delivery": "Shipping and delivery"},
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate the priority.",
            "criteria": ["Low", "Medium", "High"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
        },
    },
}
print("\n=== 1. TEXT-ONLY ===")
print(json.dumps(engine.predict(text_request), indent=2))

# ---------- 2. multimodal ----------
scene = ASSETS / "scene.png"
img = Image.new("RGB", (448, 448), (235, 238, 242))
d = ImageDraw.Draw(img)
d.rectangle([40, 260, 408, 320], fill=(60, 90, 160))
d.ellipse([120, 150, 200, 230], fill=(220, 60, 60))
d.rectangle([250, 120, 330, 230], fill=(250, 200, 40))
img.save(scene)

print("\n=== 2. MULTIMODAL ===")
print(json.dumps(engine.predict({
    "state": "A camera frame from an intersection approach.",
    "images": [str(scene)],
    "questions": {
        "hazard": {"type": "noul", "instructions": "Is there a pedestrian on the road?"},
        "action": {
            "type": "choice",
            "instructions": "What should the vehicle do?",
            "criteria": {
                "stop": "Come to a full stop",
                "yield": "Slow down and yield",
                "proceed": "Continue at current speed",
            },
        },
    },
}), indent=2))

# ---------- 3. vision A/B ----------
def lamp(color, path):
    im = Image.new("RGB", (448, 448), (30, 30, 30))
    dr = ImageDraw.Draw(im)
    dr.rectangle([150, 60, 300, 400], fill=(45, 45, 45))
    dr.ellipse([175, 90, 275, 190], fill=color)
    im.save(path)
    return str(path)

red = lamp((230, 40, 40), ASSETS / "light_red.png")
green = lamp((40, 210, 70), ASSETS / "light_green.png")

def signal_request(p):
    return {
        "state": "A traffic signal photographed from a driver's viewpoint.",
        "images": [p],
        "questions": {
            "signal": {
                "type": "choice",
                "instructions": "What colour is the illuminated lamp?",
                "criteria": {"red": "The lamp is red", "green": "The lamp is green", "yellow": "The lamp is yellow"},
            },
        },
    }

print("\n=== 3. VISION A/B ===")
for label, path in (("RED image", red), ("GREEN image", green)):
    ans = engine.predict(signal_request(path))["answers"]["signal"]
    probs = {k: round(v, 3) for k, v in ans["probabilities"].items()}
    print(f"  {label:12s} -> {ans['decision']:<6} {probs}")
