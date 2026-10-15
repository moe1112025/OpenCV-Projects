# Hand Gesture Volume Control

This is the standalone project corresponding to your original `main.py`; it does **not** call or connect to any of the other OpenCV projects. The important logic is fixed: the old code compared every frame and pressed volume-up or volume-down repeatedly, which could change volume uncontrollably. This version normalizes thumb-index distance by palm width and issues one media-key press only when gesture state transitions into `UP` or `DOWN`. The original broken condition `if key == 27 and 'q':` is replaced with explicit Q/Esc key handling.

## Install and run (Windows)

```powershell
python -m pip install -r requirements.txt
python hand_gesture_volume_control.py
```

- Open hand / spread thumb-index distance: one volume-up key press when entering `UP`.
- Pinch thumb-index distance: one volume-down key press when entering `DOWN`.
- Middle range / no hand: no volume command.
- Press **Q** or **Esc** to exit.

Safe test without sending media keys:

```powershell
python hand_gesture_volume_control.py --disable-controls
```

Thresholds can be tuned through the `classify_gesture()` function. Lighting, camera framing, individual hand geometry, and occlusion affect landmark quality. Windows media keys are sent via PyAutoGUI; if the library cannot initialize on your OS/session, check its permissions and platform setup. The included `test_images/gesture_landmark_reference.png` is a concept illustration only, not a valid MediaPipe detector input.
