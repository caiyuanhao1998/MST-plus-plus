"""Download the eleven official RGB-to-HSI checkpoints locally."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download

from demo.inference import MODEL_REPO, MODEL_REVISION, MODELS, ROOT, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "pretrained_weights")
    args = parser.parse_args()
    snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[f"{k}.pth" for k in MODELS],
        local_dir=args.weights_dir,
        max_workers=3,
    )
    manifest = dict(
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        source_revision=SOURCE_REVISION,
        sha256={},
    )
    for key in MODELS:
        path = args.weights_dir / f"{key}.pth"
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = ROOT / "docs/model_assets.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(MODELS)} checkpoints to {args.weights_dir}")


if __name__ == "__main__":
    main()
