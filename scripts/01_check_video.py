import cv2
from pathlib import Path

video_path = Path("calibration_video.MP4")

if not video_path.exists():
    print("Video not found:", video_path)
    exit()

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

print("Video opened successfully!")
print("Frames:", frame_count)
print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Duration seconds:", duration)

cap.release()