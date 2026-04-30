import cv2
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
video_path = Path("calibration_video.MP4")
output_dir = Path("all_frames")
output_dir.mkdir(exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
jpg_quality = 95

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video:", video_path)
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    out_path = output_dir / f"frame_{frame_idx:06d}.jpg"

    ok = cv2.imwrite(
        str(out_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    )

    if ok:
        saved_count += 1
    else:
        print("Failed to save:", out_path)

    frame_idx += 1

    if frame_idx % 500 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

cap.release()

print("\nDone extracting all frames.")
print("Total video frames read:", frame_idx)
print("Frames saved:", saved_count)
print("Saved to:", output_dir)