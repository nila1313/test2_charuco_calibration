import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
video_path = Path("calibration_video.MP4")
calib_path = Path("calibration_results/charuco_omnidir_selected_frames_round7.npz")
output_path = Path("calibration_results/undistorted_video_round7_scale25.mp4")

# -----------------------------
# Settings
# -----------------------------
scale = 2.5
rectify_mode = cv2.omnidir.RECTIFY_PERSPECTIVE

# -----------------------------
# Load calibration
# -----------------------------
if not calib_path.exists():
    print("Calibration file not found:", calib_path)
    exit()

data = np.load(calib_path, allow_pickle=True)

K = data["camera_matrix"].astype(np.float64)
xi = np.asarray(data["xi"], dtype=np.float64).reshape(1, 1)
D = data["dist_coeffs"].astype(np.float64)
rms = data["rms"]

print("Loaded calibration:")
print("RMS:", rms)
print("K:\n", K)
print("xi:\n", xi)
print("D:\n", D)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video:", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nInput video:")
print("Frames:", frame_count)
print("FPS:", fps)
print("Size:", w, "x", h)

image_size = (w, h)

# -----------------------------
# New camera matrix
# -----------------------------
new_K = K.copy()
new_K[0, 0] = K[0, 0] / scale
new_K[1, 1] = K[1, 1] / scale
new_K[0, 2] = w / 2
new_K[1, 2] = h / 2

print("\nUndistortion:")
print("Scale:", scale)
print("New K:\n", new_K)

# -----------------------------
# Precompute undistortion maps
# -----------------------------
map1, map2 = cv2.omnidir.initUndistortRectifyMap(
    K,
    D,
    xi,
    np.eye(3, dtype=np.float64),
    new_K.astype(np.float64),
    image_size,
    cv2.CV_16SC2,
    rectify_mode
)

# -----------------------------
# Output video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    image_size
)

if not writer.isOpened():
    print("Could not create output video:", output_path)
    cap.release()
    exit()

# -----------------------------
# Process video
# -----------------------------
frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    undistorted = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    writer.write(undistorted)

    frame_idx += 1

    if frame_idx % 300 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

cap.release()
writer.release()

print("\nDone.")
print("Frames processed:", frame_idx)
print("Saved undistorted video to:", output_path)