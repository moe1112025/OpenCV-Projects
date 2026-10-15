# Object Measurement

A contour-based educational measurement demo. It detects high-contrast objects and reports bounding-box width, height, and contour area.

## Run

```powershell
python -m pip install -r requirements.txt
python object_measurement.py
```

The provided `test_images/object_scene.png` produces a visual result and a threshold mask in `outputs/`. Webcam mode:

```powershell
python object_measurement.py --source 0
```

**Important measurement limitation:** the default is pixels and pixel-squared, not centimetres. Physical units are only meaningful after calibrating the camera/setup and passing a known scale in **pixels per unit**, for example `--pixels-per-unit 100` means 100 pixels represent one chosen unit. Perspective, lens distortion, object tilt, shadows, and inaccurate calibration affect measurements. This is not metrology-grade software.
