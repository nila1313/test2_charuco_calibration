import cv2
import numpy as np
from pathlib import Path
import glob

# -----------------------------
# Paths
# -----------------------------
calib_path = Path("calibration_results/charuco_omnidir_selected_frames_round7.npz")
input_folder = Path("good_charuco_frames")
output_folder = Path("calibration_results/undistortion_test")
output_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load calibration
# -----------------------------
if not calib_path.exists():
    print("Calibration file not found:", calib_path)
    exit()

data = np.load(calib_path, allow_pickle=True)

K = data["camera_matrix"]
xi = data["xi"]
D = data["dist_coeffs"]
rms = data["rms"]

xi_scalar = float(np.asarray(xi).reshape(-1)[0])

print("Loaded calibration:")
print("RMS:", rms)
print("K:\n", K)
print("xi:", xi_scalar)
print("D:\n", D)

# -----------------------------
# Select test images
# -----------------------------
image_paths = sorted(glob.glob(str(input_folder / "*.jpg")))

if len(image_paths) == 0:
    print("No images found in:", input_folder)
    exit()

# Pick some images across the sequence
num_tests = 12
indices = np.linspace(0, len(image_paths) - 1, num_tests).astype(int)
test_paths = [image_paths[i] for i in indices]

print("Testing images:", len(test_paths))

# -----------------------------
# Undistortion settings
# -----------------------------
# balance/scale idea:
# smaller scale = more zoomed/cropped
# larger scale = wider view but may show black borders
scale = 2.0

for idx, path in enumerate(test_paths):
    img = cv2.imread(path)

    if img is None:
        print("Could not read:", path)
        continue

    h, w = img.shape[:2]
    image_size = (w, h)

    # New camera matrix for undistorted output
    new_K = K.copy()
    new_K[0, 0] = K[0, 0] / scale
    new_K[1, 1] = K[1, 1] / scale
    new_K[0, 2] = w / 2
    new_K[1, 2] = h / 2

    xi_map = np.asarray(xi, dtype=np.float64).reshape(1, 1)

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K.astype(np.float64),
        D.astype(np.float64),
        xi_map,
        np.eye(3, dtype=np.float64),
        new_K.astype(np.float64),
        image_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    

    undistorted = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Side-by-side comparison
    comparison = np.hstack([img, undistorted])

    name = Path(path).stem
    out_path = output_folder / f"undistort_test_{idx:02d}_{name}.jpg"

    cv2.imwrite(str(out_path), comparison)

    print("Saved:", out_path)

print("\nDone.")
print("Open this folder to inspect results:")
print(output_folder)