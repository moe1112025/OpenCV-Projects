# Drowsiness Detector (Experimental)

Uses MediaPipe Face Mesh eye landmarks and the Eye Aspect Ratio (EAR) heuristic. A temporal duration and alert cooldown reduce single-frame triggers and repeated alert spam. This replaces the original external `shape_predictor_68_face_landmarks1.dat` requirement, so there is no large predictor file to find manually.

## Install and run

```powershell
python -m pip install -r requirements.txt
python drowsiness_detector.py
```

The default source is webcam index `0`. To run a single-frame visual smoke test:

```powershell
python drowsiness_detector.py --image test_images/face_reference.png
```

A single image can report landmarks/EAR, but it cannot establish whether a person is becoming drowsy over time. The bundled portrait is a **visual smoke-test fixture**, not a validated benchmark.

Options: `--threshold 0.21`, `--closed-seconds 1.2`, `--cooldown 3.0`, `--source 0`, `--no-display`, or `--voice-alert` for optional non-blocking speech alerts (requires the platform TTS engine).

**Safety limitation:** EAR is sensitive to camera angle, glasses, lighting, face geometry, and model errors. This is an educational prototype, not a medical device, a driver-monitoring certification, or a substitute for rest. Do not use it as the sole basis for safety-critical decisions.
