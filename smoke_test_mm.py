"""Multimodal smoke test for internlm/Intern-Decision-4B."""
import json
import sys
from pathlib import Path

MODEL_DIR = Path(r"D:\models\Intern-Decision-4B")
sys.path.insert(0, str(MODEL_DIR))

from PIL import Image, ImageDraw  # noqa: E402
from inference import DecisionEngine  # noqa: E402

img_path = MODEL_DIR / "_smoke_scene.png"
img = Image.new("RGB", (448, 448), (235, 238, 242))
d = ImageDraw.Draw(img)
d.rectangle([40, 260, 408, 320], fill=(60, 90, 160))
d.ellipse([120, 150, 200, 230], fill=(220, 60, 60))
d.rectangle([250, 120, 330, 230], fill=(250, 200, 40))
img.save(img_path)

engine = DecisionEngine(checkpoint=MODEL_DIR, device="cuda:1")
mm_request = {
    "state": "A camera frame from an intersection approach.",
    "images": [str(img_path)],
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
}
print(json.dumps(engine.predict(mm_request), indent=2))
