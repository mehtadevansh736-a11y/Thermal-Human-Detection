"""Test thermal human YOLOv8 model - no training needed."""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL = BASE / "model.pt"

def main(image: str | None = None):
    model = YOLO(str(MODEL))
    source = image or str(BASE / "demo_detections")
    print(f"Model: {MODEL}")
    print(f"Source: {source}")
    results = model.predict(
        source=source,
        conf=0.4,
        save=True,
        project=str(BASE),
        name="outputs",
        exist_ok=True,
    )
    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        print(f"{r.path} -> {n} HUMAN(s)")
        if r.boxes is not None:
            for b in r.boxes:
                print(f"  conf={float(b.conf):.2f} box={b.xyxy.tolist()}")
    print(f"\nSaved to: {BASE / 'outputs'}")

if __name__ == "__main__":
    import sys
    # usage: python test_model.py [optional_image_path]
    main(sys.argv[1] if len(sys.argv) > 1 else None)
