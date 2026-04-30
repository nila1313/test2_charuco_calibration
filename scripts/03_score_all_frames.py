import cv2
import numpy as np
import csv
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
input_dir = Path("all_frames")
debug_dir = Path("debug_detections")
results_dir = Path("calibration_results")

debug_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "frame_scores.csv"

# -----------------------------
# ChArUco board settings
# -----------------------------
squares_x = 7
squares_y = 7
square_length = 1.0
marker_length = 0.8

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    dictionary
)

# -----------------------------
# Detector parameters
# -----------------------------
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 80
params.cornerRefinementMinAccuracy = 0.0005

detector = cv2.aruco.ArucoDetector(dictionary, params)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_gamma(img, gamma=1.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(gray, table)


def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_stats(gray):
    mean_brightness = float(np.mean(gray))
    overexposed_ratio = float(np.mean(gray > 245))
    underexposed_ratio = float(np.mean(gray < 10))
    return mean_brightness, overexposed_ratio, underexposed_ratio


def coverage_stats(corners, width, height):
    """
    Estimate where detected ChArUco corners lie in the image.
    This helps us later choose frames covering center/edges/corners.
    """
    if corners is None or len(corners) == 0:
        return None

    pts = corners.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]

    center_x = float(np.mean(xs) / width)
    center_y = float(np.mean(ys) / height)

    min_x = float(np.min(xs) / width)
    max_x = float(np.max(xs) / width)
    min_y = float(np.min(ys) / height)
    max_y = float(np.max(ys) / height)

    area = float((max_x - min_x) * (max_y - min_y))

    return center_x, center_y, min_x, max_x, min_y, max_y, area


# -----------------------------
# Process frames
# -----------------------------
frame_paths = sorted(input_dir.glob("*.jpg"))

print("Frames found:", len(frame_paths))
print("Scoring frames...")

rows = []

for idx, path in enumerate(frame_paths):
    img = cv2.imread(str(path))

    if img is None:
        continue

    height, width = img.shape[:2]

    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_gamma(img, gamma=1.8)

    blur = blur_score(gray_raw)
    mean_brightness, over_ratio, under_ratio = brightness_stats(gray_raw)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    marker_count = 0 if marker_ids is None else len(marker_ids)
    charuco_count = 0

    cx = cy = min_x = max_x = min_y = max_y = area = -1.0

    detected = False

    if marker_ids is not None and len(marker_ids) >= 4:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )

        if charuco_ids is not None:
            charuco_count = len(charuco_ids)

            cov = coverage_stats(charuco_corners, width, height)

            if cov is not None:
                cx, cy, min_x, max_x, min_y, max_y, area = cov

            if charuco_count >= 4:
                detected = True

                # Save debug image for every 30th detected frame only,
                # so the debug folder does not become too huge.
                if idx % 30 == 0:
                    debug = img.copy()
                    cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
                    cv2.aruco.drawDetectedCornersCharuco(
                        debug,
                        charuco_corners,
                        charuco_ids,
                        (0, 0, 255)
                    )
                    debug_path = debug_dir / f"debug_{path.stem}.jpg"
                    cv2.imwrite(str(debug_path), debug)

    rows.append({
        "filename": path.name,
        "frame_index": int(path.stem.replace("frame_", "")),
        "marker_count": marker_count,
        "charuco_count": charuco_count,
        "blur": blur,
        "mean_brightness": mean_brightness,
        "overexposed_ratio": over_ratio,
        "underexposed_ratio": under_ratio,
        "center_x": cx,
        "center_y": cy,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "coverage_area": area,
        "detected": int(detected)
    })

    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(frame_paths)} frames...")


# -----------------------------
# Save CSV
# -----------------------------
fieldnames = [
    "filename",
    "frame_index",
    "marker_count",
    "charuco_count",
    "blur",
    "mean_brightness",
    "overexposed_ratio",
    "underexposed_ratio",
    "center_x",
    "center_y",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "coverage_area",
    "detected"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

detected_count = sum(r["detected"] for r in rows)

print("\nDone.")
print("Total frames scored:", len(rows))
print("Frames with ChArUco detected:", detected_count)
print("CSV saved to:", csv_path)
print("Debug images saved to:", debug_dir)