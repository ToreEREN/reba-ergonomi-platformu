"""Manual end-to-end smoke check using the bundled example image."""

from pathlib import Path
import sys

import cv2

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from app import analyze_frame, load_all_models

frame = cv2.imread(str(root / "Output" / "reba_sonuc.png"))
if frame is None:
    raise RuntimeError("Bundled smoke image could not be read")

yolo, angle_model, scaler, meta = load_all_models()
result, keypoints = analyze_frame(frame, yolo, angle_model, scaler, meta)
print({
    "runtime": meta.get("runtime"),
    "person_detected": result is not None,
    "reba": result.get("final_reba") if result else None,
    "keypoints": len(keypoints) if keypoints else 0,
})
