"""Test thermal human model on video file or webcam - no thermal camera needed."""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL = BASE / "model.pt"

def run(source: str, fake_thermal: bool = False):
    model = YOLO(str(MODEL))
    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        print("Put an .mp4 in this folder and run: python test_video.py myvideo.mp4")
        return
    out_path = BASE / "outputs" / "video_output.mp4"
    out_path.parent.mkdir(exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"Reading: {source} ({w}x{h} @ {fps:.1f}fps) fake_thermal={fake_thermal}")
    print("Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        infer_frame = frame
        if fake_thermal:
            # mimic thermal palette for normal RGB video/webcam
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            infer_frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        res = model.predict(infer_frame, conf=0.4, verbose=False)[0]
        annotated = res.plot()
        writer.write(annotated)
        cv2.imshow("thermal-human (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    # usage:
    #   python test_video.py myvideo.mp4
    #   python test_video.py 0                  -> webcam
    #   python test_video.py myvideo.mp4 fake   -> RGB video converted to fake-thermal
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    fake = len(sys.argv) > 2 and sys.argv[2].lower().startswith("fake")
    run(src, fake_thermal=fake)
