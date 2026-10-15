# OpenCV Professional Project Suite

**Independent Computer Vision projects for PyCharm + Python on Windows.** This suite organizes your current OpenCV practice into separate, runnable folders. Each project has its own source file(s), `requirements.txt`, README, sample inputs and output directory. No project imports or launches another project.

> **Important for your existing `OpenCV` workspace:** extract this archive to a separate folder first, then copy only the project folders you want into your current workspace. Do **not** replace your existing `.venv`, `pyproject.toml`, or `uv.lock`. They belong to your current environment. The suite's root `pyproject.toml` is only for optional QA tooling.

## Folder map

```text
OpenCV_Professional_Project_Suite/
├── Beginner Projects CLI/
│   ├── contours.py
│   ├── threshold.py
│   └── test_images/ + outputs/
├── Beginner Projects Vision/
│   ├── centroid_trajectory.py
│   ├── dense_optical_flow.py
│   ├── motion_detection.py
│   ├── object_tracker.py
│   ├── sparse_optical_flow.py
│   └── test_images/sample_motion.avi
├── Color Detector/
│   └── color_detector.py
├── Drowsiness Detector/
│   └── drowsiness_detector.py
├── Face and Eye Detection/
│   ├── face_eye_detection.py
│   ├── haarcascade_eye.xml
│   └── haarcascade_frontalface_default.xml
├── Object Measurement/
│   └── object_measurement.py
├── QR Code Scanner/
│   └── qr_scanner.py
├── Hand Gesture Volume Control/
│   └── hand_gesture_volume_control.py
├── Phase3/
│   ├── camera_calibration.py
│   ├── grabcut_segmentation.py
│   ├── perspective_transform.py
│   ├── undistort_image.py
│   └── watershed_segmentation.py
├── Phase4/
│   ├── hog_svm_classifier.py
│   └── image_classification.py
├── tests/
├── pyproject.toml
└── README.md
```

The incorrect spelling `countors.py` / `contors.py` from the working notes is normalized to the standard name **`contours.py`**. The code you called the root `main.py` is kept as an independent **Hand Gesture Volume Control** project so it cannot accidentally become an entry point for other projects.

## 1. Recommended environment

You are already using a project-local virtual environment in PyCharm. Python **3.11** is a reasonable target for the optional MediaPipe projects; use the Python interpreter that is selected for your existing project. These commands assume Windows PowerShell:

```powershell
# From your existing OpenCV project directory
python --version
python -m pip --version

# If OpenCV and NumPy are not installed in that selected interpreter:
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
```

Don't install every optional library just to run the basic projects. Each folder declares its own dependencies. In a PowerShell terminal, `cd` into the folder you want, then install that folder's requirements. For example:

```powershell
cd "C:\Users\YOUR_USER\OneDrive\Desktop\OpenCV\Color Detector"
python -m pip install -r requirements.txt
python color_detector.py
```

Replace the example path with your actual folder path. The screenshots indicate that your existing workspace is under OneDrive; this suite does not need that exact location.

## 2. Run every project independently

### Beginner Projects CLI

```powershell
cd "Beginner Projects CLI"
python -m pip install -r requirements.txt
python threshold.py
python contours.py
```

### Beginner Projects Vision

```powershell
cd "Beginner Projects Vision"
python -m pip install -r requirements.txt
python centroid_trajectory.py
python dense_optical_flow.py
python sparse_optical_flow.py
python motion_detection.py
python object_tracker.py
```

These default to the packaged synthetic clip. `object_tracker.py` opens an ROI selector: drag around the moving object and press Enter. If CSRT is unavailable in your OpenCV build, try `python object_tracker.py --tracker MIL`.

### Color Detector

```powershell
cd "Color Detector"
python -m pip install -r requirements.txt
python color_detector.py
python color_detector.py --source 0
```

The first command processes its included still image and saves an annotated output. The second opens the default webcam.

### Drowsiness Detector

```powershell
cd "Drowsiness Detector"
python -m pip install -r requirements.txt
python drowsiness_detector.py
```

This one requires MediaPipe. If pip reports no compatible wheel, check the Python version and the selected PyCharm interpreter before changing code. A still image can be used only for landmark/EAR smoke testing; drowsiness is a temporal signal and cannot be inferred from one photograph. This demo is not a certified safety device.

### Face and Eye Detection

```powershell
cd "Face and Eye Detection"
python -m pip install -r requirements.txt
python face_eye_detection.py
python face_eye_detection.py --source 0
```

Cascade XML files are bundled next to the script, so you do not need to fetch those files separately.

### Object Measurement

```powershell
cd "Object Measurement"
python -m pip install -r requirements.txt
python object_measurement.py
python object_measurement.py --source 0
```

By default, the result is in pixels and pixel-squared. To report physical units, calibrate the setup and supply a correct `--pixels-per-unit` value; don't assume the example scale text is your camera calibration.

### QR Code Scanner

```powershell
cd "QR Code Scanner"
python -m pip install -r requirements.txt
python qr_scanner.py
python qr_scanner.py --source 0
```

The scanner does not open QR links automatically. Use `--open-url` only if you intentionally want the program to ask you to confirm opening a decoded HTTP(S) URL.

### Hand Gesture Volume Control (your original `main.py` concept)

```powershell
cd "Hand Gesture Volume Control"
python -m pip install -r requirements.txt
python hand_gesture_volume_control.py --disable-controls
```

First use `--disable-controls` to validate webcam tracking. Remove that flag when ready to send media-key commands. The original implementation pressed volume up/down on every frame, which could repeatedly change the volume. The updated version sends one key press only when a gesture state changes. Press **Q** or **Esc** to quit.

### Phase3

```powershell
cd Phase3
python -m pip install -r requirements.txt
python camera_calibration.py
python grabcut_segmentation.py
python perspective_transform.py
python undistort_image.py
python watershed_segmentation.py
```

Outputs are written under `Phase3/outputs/`. Synthetic calibration images are included only as a pipeline smoke test. Real camera calibration requires multiple actual photos of the printed checkerboard.

### Phase4

```powershell
cd Phase4
python -m pip install -r requirements.txt
python hog_svm_classifier.py --mode train
python image_classification.py --mode train
python hog_svm_classifier.py --mode predict --image test_images\sample_shape.png
python image_classification.py --mode predict --image test_images\sample_shape.png
```

The packaged train/test dataset contains simple geometric shapes. This validates a classical machine-learning pipeline, not recognition of arbitrary objects or natural scenes.

## 3. PyCharm setup

1. Open your existing **OpenCV** project folder in PyCharm.
2. Copy the project folder(s) from this suite into your existing root directory. Do not merge the contents of separate projects into one Python file.
3. Expand the project folder in the left pane and open its own `.py` file.
4. Open **Settings → Project → Python Interpreter** and confirm the interpreter points to your intended `.venv`.
5. Open the Terminal, `cd` into that project folder, and run `python -m pip install -r requirements.txt`.
6. Right-click the file and select **Run**, or use the terminal command in that project's README.
7. Generated results go into that folder's `outputs/` directory. Original sample images stay inside `test_images/`.

For projects using webcam input, close other apps that may be using the camera. On Windows, check Settings → Privacy & security → Camera if access fails. A webcam index other than `0` can be tried by using `--source 1`.

## 4. Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named cv2` | Dependency installed into another interpreter | In the PyCharm terminal, run `python -m pip install -r requirements.txt` from that project folder. |
| `Could not open camera` | Camera permission, busy device, or wrong index | Close other camera apps, check Windows camera permissions, try `--source 1`. |
| `Could not read image` | Path typo or working-directory assumptions | Use the included defaults, or quote an absolute path with `--image`. |
| MediaPipe install fails | Incompatible Python/platform wheel | Confirm Python version with `python --version` and interpreter path with `python -c "import sys; print(sys.executable)"`. |
| No eye/face boxes | Lighting, pose, resolution, detector limitation | Test the sample first, face the camera, add light, reduce occlusion. Detection is not guaranteed. |
| Measurement units look wrong | Scale factor was assumed rather than calibrated | Use pixel measurements or obtain a real pixels-per-unit calibration. |
| QR link looks suspicious | QR payload is untrusted data | Don't confirm opening it. The default is not to open anything. |
| Tracker unavailable | OpenCV build lacks that tracker | Try `--tracker MIL`; if needed install `opencv-contrib-python` instead of simultaneously installing both OpenCV packages. |
| Camera calibration error | Not enough valid checkerboard images | Collect 10–20 real photos with the printed board at varied angles and distances; use correct inner-corner pattern. |

## 5. Test images, videos and outputs

Every project has its own `test_images/` and `outputs/`. Sample media includes synthetic shapes, a color detector scene, a motion AVI, a QR code, a checkerboard, a perspective-warped paper, a segmentation scene, and a compact labelled shape dataset. Most image projects write their outputs the first time they run; the output folders are intentionally separate and aren't shared between projects.

The portrait supplied with the face/eye demos is a bundled sample fixture for visual smoke testing, not a detector accuracy claim. Synthetic images are intended to check that scripts run and create understandable results; they do not replace testing on your own camera and target scenes.

## 6. Offline QA tests

From the suite root, install the root dev tools if desired and run:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

These tests validate selected image-processing functions, paths, and QR URL policy without opening a camera. They do **not** prove the webcam works on your computer or that algorithms are production-certified.

## Safety and engineering notes

- All projects are local educational CV demos. They have no hidden background launcher and do not connect to each other.
- QR content is untrusted input; URLs are never opened by default.
- Hand gesture media-key control is state-transition-based; test with `--disable-controls` first.
- Drowsiness detection is a heuristic demo, not medical advice or certified driver-monitoring technology.
- Object measurement requires calibration before reporting physical units.
- Calibration needs real camera imagery for meaningful intrinsic/distortion coefficients.
- `joblib` model files are pickle-based. Load only models created by your own trusted training runs.
