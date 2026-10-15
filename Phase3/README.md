# Phase3 — Classical Computer Vision

Five independent projects: camera calibration, GrabCut segmentation, perspective rectification, image undistortion, and watershed segmentation. Each script is a separate executable; no project imports another.

## Install and run

```powershell
python -m pip install -r requirements.txt
python camera_calibration.py
python grabcut_segmentation.py
python perspective_transform.py
python undistort_image.py
python watershed_segmentation.py
```

Each script writes its result under `outputs/`. Add `--show` to view the output for image-based demos.

## Sample asset guide

- `test_images/grabcut_scene.png` — synthetic foreground/background scene.
- `test_images/document_scene.png` — perspective-distorted synthetic page.
- `test_images/lens_grid.png` — visual grid for the undistortion pipeline.
- `test_images/watershed_scene.png` — synthetic touching objects.
- `test_images/checkerboard_to_print.png` — 10 × 7 squares (9 × 6 inner corners).
- `test_images/calibration_samples/` — generated views to smoke-test the calibration pipeline.

**Camera calibration warning:** the generated board views only check whether the script can detect corners and write an NPZ. They are not real camera captures and should not be used to claim accurate calibration. The bundled default output is named `demo_only_calibration.npz`; for your own real photographs, pass `--output outputs/camera_calibration.npz`. For meaningful results, print the checkerboard, keep its square size known, and take 10–20 real photos at varied positions, rotations, and distances; then run `python camera_calibration.py --images "C:\path\to\your_calibration_photos" --pattern 9x6 --square-size 25` (if each square is 25 mm).

## Specifics

- `grabcut_segmentation.py` accepts `--rect X Y W H` to set the initial foreground rectangle.
- `perspective_transform.py` detects the largest contour quadrilateral; if auto-detection fails, use `--points x1 y1 x2 y2 x3 y3 x4 y4` in top-left, top-right, bottom-right, bottom-left order.
- `undistort_image.py` uses fabricated distortion values if `--calibration` is omitted. Use `--calibration outputs/camera_calibration.npz` only after calibrating against real images.
