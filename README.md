# ChArUco Omnidirectional Camera Calibration from Video

This project is a full camera calibration pipeline built from a raw calibration video.  
The video was challenging: it was overexposed, blurry in some parts, and captured with a strongly distorted wide-angle/omnidirectional camera.

The goal was to get a better calibration result by carefully selecting useful ChArUco frames instead of using random or manually chosen frames.

---

## What this project does

The pipeline starts with a raw video and ends with a calibrated camera model and an undistorted video.

In short, it does this:

```text
raw video
→ extract frames
→ detect ChArUco board
→ score frame quality
→ select good frames
→ calibrate camera
→ remove bad frames using reprojection error
→ test undistortion
→ undistort the full video
