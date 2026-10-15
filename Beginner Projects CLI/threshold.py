"""Thresholding demo with an included offline sample image."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "test_images" / "threshold_scene.png"

def process(image, mode="binary", threshold=127, max_value=255, blur=3):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        k = blur if blur % 2 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    modes = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
        "otsu": cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    }
    if mode not in modes:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    _, result = cv2.threshold(gray, 0 if mode == "otsu" else threshold, max_value, modes[mode])
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    ap.add_argument("--mode", choices=["binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu"], default="otsu")
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--output", type=Path, default=HERE / "outputs" / "threshold_output.png")
    ap.add_argument("--show", action="store_true", help="Display the result window")
    args = ap.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"ERROR: Could not read image: {args.image}")
    result = process(image, args.mode, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise SystemExit(f"ERROR: Could not save output: {args.output}")
    print(f"[OK] Mode: {args.mode} | Output: {args.output.resolve()}")
    if args.show:
        cv2.imshow("Thresholding Output", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
