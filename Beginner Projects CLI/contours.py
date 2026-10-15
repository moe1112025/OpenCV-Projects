"""Detect and annotate external contours in a still image."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "test_images" / "geometric_scene.png"

def process(image, min_area=500):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(output, [contour], -1, (0, 190, 255), 2)
        cv2.rectangle(output, (x, y), (x+w, y+h), (40, 220, 80), 2)
        cv2.putText(output, f"A={area:.0f}px2", (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .48, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(output, f"A={area:.0f}px2", (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
        kept += 1
    cv2.putText(output, f"Contours: {kept}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,160,255), 2, cv2.LINE_AA)
    return output, kept, mask

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--min-area", type=float, default=600)
    ap.add_argument("--output", type=Path, default=HERE / "outputs" / "contours_output.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"ERROR: Could not read image: {args.image}")
    result, count, mask = process(image, args.min_area)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)
    cv2.imwrite(str(args.output.with_name(args.output.stem + "_mask.png")), mask)
    print(f"[OK] {count} contours retained | Output: {args.output.resolve()}")
    if args.show:
        cv2.imshow("Contours", result); cv2.imshow("Binary mask", mask)
        cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
