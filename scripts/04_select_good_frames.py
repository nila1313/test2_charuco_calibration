import csv
import shutil
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
scores_path = Path("calibration_results/frame_scores.csv")
all_frames_dir = Path("all_frames")
good_dir = Path("good_charuco_frames")
results_dir = Path("calibration_results")

good_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

selected_csv_path = results_dir / "selected_good_frames.csv"

# -----------------------------
# Selection settings
# -----------------------------
MIN_CHARUCO = 8
MIN_BLUR = 70.0
MAX_OVEREXPOSED = 0.15
MIN_COVERAGE_AREA = 0.04

# To avoid selecting almost identical neighboring video frames
MIN_FRAME_GAP = 5

# Image coverage grid
# 5 columns x 3 rows = 15 image regions
GRID_X = 5
GRID_Y = 3

# Maximum selected frames per grid cell
MAX_PER_CELL = 80

# Overall max selected frames
MAX_TOTAL_FRAMES = 500

# If True, remove old files from good_charuco_frames before copying
CLEAR_OUTPUT_FOLDER = True


# -----------------------------
# Helper functions
# -----------------------------
def to_float(row, key, default=-1.0):
    try:
        return float(row[key])
    except Exception:
        return default


def to_int(row, key, default=0):
    try:
        return int(row[key])
    except Exception:
        return default


def compute_quality_score(row):
    """
    Higher score = better frame.

    We reward:
    - more ChArUco corners
    - sharper images
    - larger board coverage

    We penalize:
    - overexposure
    """

    charuco = to_float(row, "charuco_count")
    blur = to_float(row, "blur")
    over = to_float(row, "overexposed_ratio")
    area = to_float(row, "coverage_area")

    score = (
        charuco * 10.0
        + blur * 0.5
        + area * 100.0
        - over * 100.0
    )

    return score


def grid_cell(row):
    """
    Put the frame into a spatial grid based on where the board center is.

    center_x and center_y are normalized:
    0.0 = left/top
    1.0 = right/bottom
    """
    cx = to_float(row, "center_x")
    cy = to_float(row, "center_y")

    gx = int(cx * GRID_X)
    gy = int(cy * GRID_Y)

    gx = max(0, min(GRID_X - 1, gx))
    gy = max(0, min(GRID_Y - 1, gy))

    return gx, gy


def far_enough_from_selected(frame_index, selected_indices):
    """
    Avoid selecting frames that are too close in time.
    """
    for idx in selected_indices:
        if abs(frame_index - idx) < MIN_FRAME_GAP:
            return False
    return True


# -----------------------------
# Clear output folder if needed
# -----------------------------
if CLEAR_OUTPUT_FOLDER:
    for old_file in good_dir.glob("*.jpg"):
        old_file.unlink()

# -----------------------------
# Read scores
# -----------------------------
if not scores_path.exists():
    print("Could not find:", scores_path)
    print("Run scripts/03_score_all_frames.py first.")
    exit()

rows = []

with open(scores_path, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        detected = to_int(row, "detected")
        charuco = to_int(row, "charuco_count")
        blur = to_float(row, "blur")
        over = to_float(row, "overexposed_ratio")
        area = to_float(row, "coverage_area")
        cx = to_float(row, "center_x")
        cy = to_float(row, "center_y")

        # Basic quality filters
        if detected != 1:
            continue

        if charuco < MIN_CHARUCO:
            continue

        if blur < MIN_BLUR:
            continue

        if over > MAX_OVEREXPOSED:
            continue

        if area < MIN_COVERAGE_AREA:
            continue

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue

        row["quality_score"] = compute_quality_score(row)
        row["grid_x"], row["grid_y"] = grid_cell(row)

        rows.append(row)

print("Frames passing basic filters:", len(rows))

if len(rows) == 0:
    print("No frames passed the filters. Try relaxing thresholds.")
    exit()

# -----------------------------
# Group by image location grid
# -----------------------------
cells = {}

for row in rows:
    cell = (row["grid_x"], row["grid_y"])
    cells.setdefault(cell, []).append(row)

# Sort each cell by quality score, best first
for cell in cells:
    cells[cell].sort(key=lambda r: r["quality_score"], reverse=True)

# -----------------------------
# Select balanced frames from each cell
# -----------------------------
selected = []
selected_indices = []

# First pass: take good frames from each region
for gy in range(GRID_Y):
    for gx in range(GRID_X):
        cell = (gx, gy)
        candidates = cells.get(cell, [])

        count_in_cell = 0

        for row in candidates:
            frame_index = to_int(row, "frame_index")

            if not far_enough_from_selected(frame_index, selected_indices):
                continue

            selected.append(row)
            selected_indices.append(frame_index)
            count_in_cell += 1

            if count_in_cell >= MAX_PER_CELL:
                break

            if len(selected) >= MAX_TOTAL_FRAMES:
                break

        if len(selected) >= MAX_TOTAL_FRAMES:
            break

    if len(selected) >= MAX_TOTAL_FRAMES:
        break

# If still fewer than max, fill from remaining best frames globally
if len(selected) < MAX_TOTAL_FRAMES:
    already_selected_files = set(r["filename"] for r in selected)
    remaining = [
        r for r in rows
        if r["filename"] not in already_selected_files
    ]
    remaining.sort(key=lambda r: r["quality_score"], reverse=True)

    for row in remaining:
        frame_index = to_int(row, "frame_index")

        if not far_enough_from_selected(frame_index, selected_indices):
            continue

        selected.append(row)
        selected_indices.append(frame_index)

        if len(selected) >= MAX_TOTAL_FRAMES:
            break

# Sort selected frames by frame index for easier viewing
selected.sort(key=lambda r: to_int(r, "frame_index"))

# -----------------------------
# Copy selected frames
# -----------------------------
copied = 0

for row in selected:
    src = all_frames_dir / row["filename"]
    dst = good_dir / row["filename"]

    if src.exists():
        shutil.copy2(src, dst)
        copied += 1
    else:
        print("Missing source frame:", src)

# -----------------------------
# Save selected CSV
# -----------------------------
fieldnames = list(selected[0].keys()) if selected else []

with open(selected_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected)

# -----------------------------
# Print summary
# -----------------------------
print("\nSelection complete.")
print("Selected frames:", len(selected))
print("Copied frames:", copied)
print("Saved selected frames to:", good_dir)
print("Saved selected CSV to:", selected_csv_path)

print("\nGrid distribution:")
for gy in range(GRID_Y):
    line = []
    for gx in range(GRID_X):
        count = sum(
            1 for r in selected
            if r["grid_x"] == gx and r["grid_y"] == gy
        )
        line.append(f"{count:3d}")
    print(" ".join(line))