"""Create a reproducibility and integrity manifest for production artifacts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.FeatureSchema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FEATURE_SEMANTICS
from Utils.ModelReliability import sha256_file

ARTIFACTS = {
    "angle_model": ROOT / "ModelExperiments" / "extra_trees.joblib",
    "input_scaler": ROOT / "BestModel" / "ft_transformer_scaler.pkl",
    "pose_model": ROOT / "yolo11n-pose.pt",
    "archived_predictions_all": ROOT / "BestModel" / "advanced_ft_transformer_predictions_all.csv",
    "archived_predictions_test": ROOT / "BestModel" / "advanced_ft_transformer_predictions_test.csv",
}


def main():
    model = joblib.load(ARTIFACTS["angle_model"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_semantics": FEATURE_SEMANTICS,
        "model_class": type(model).__name__,
        "model_parameters": model.get_params(),
        "artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in ARTIFACTS.items()
        },
        "known_limitations": [
            "Archived training features are standardized and anonymous.",
            "Runtime *_z inputs are YOLO confidence values, not physical depth.",
            "Person/sequence/camera group identifiers are not available in the archived dataset.",
        ],
    }
    output = ROOT / "ModelExperiments" / "artifact_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
