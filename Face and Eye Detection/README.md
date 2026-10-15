# Face and Eye Detection

Standalone Haar-cascade detector for faces and eye regions. The two XML files are included in the folder, and the script validates the cascade files before starting. A bundled portrait is available for a still-image smoke test.

## Run

```powershell
python -m pip install -r requirements.txt
python face_eye_detection.py
python face_eye_detection.py --source 0
```

The first command reads `test_images/face_reference.png`, writes `outputs/face_eye_output.png`, and opens a window. Use `--no-display` to write without a GUI. Webcam mode exits with **Q**.

Haar cascades are lightweight but can miss faces/eyes or return false positives depending on pose, occlusion, light, and image quality. Eye rectangles are regions proposed by a classifier, not a biometric identity result.
