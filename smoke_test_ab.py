"""A/B test: does the vision path actually drive the decision?"""
import sys
from pathlib import Path

MODEL_DIR = Path(r"D:\models\Intern-Decision-4B")
sys.path.insert(0, str(MODEL_DIR))

from PIL import Image, ImageDraw  # noqa: E402
from inference import DecisionEngine  # noqa: E402

def light(color, path):
    img = Image.new("RGB", (448, 448), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([150, 60, 300, 400], fill=(45, 45, 45))
    d.ellipse([175, 90, 275, 190], fill=color)
    img.save(path)
    return str(path)

red = light((230, 40, 40), MODEL_DIR / "_ab_red.png")
green = light((40, 210, 70), MODEL_DIR / "_ab_green.png")

engine = DecisionEngine(checkpoint=MODEL_DIR, device="cuda:1")

req = lambda p: {
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

for label, p in (("RED image", red), ("GREEN image", green)):
    a = engine.predict(req(p))["answers"]["signal"]
    print(f"{label:12s} -> decision={a['decision']:<6} probs={ {k: round(v, 3) for k, v in a['probabilities'].items()} }")
