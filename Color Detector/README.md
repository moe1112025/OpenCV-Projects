# Color Detector

HSV-based red, green, and blue detector. Red is correctly split into two hue ranges because red wraps around the HSV hue axis. Morphological opening and closing reduce isolated noise. The application supports a sample still image and a live webcam/video source.

## Run

```powershell
python -m pip install -r requirements.txt
python color_detector.py
```

The included `test_images/color_scene.png` is processed and saved to `outputs/color_detector_output.png`. Add `--no-display` to save without opening a GUI window.

Webcam mode:

```powershell
python color_detector.py --source 0
```

Press **Q** in the video window to stop. Change sensitivity with `--min-area 1000`. Thresholds are practical defaults, not universal constants: lighting, camera white balance, and object color affect detections.
