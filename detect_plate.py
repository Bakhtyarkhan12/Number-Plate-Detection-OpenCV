"""
Number Plate Detection Using Python & OpenCV
============================================
Detects vehicle number plates from images or live webcam feed.

HOW TO USE:
    python detect_plate.py                        (runs on the included car.jpg)
    python detect_plate.py --image your_car.jpg   (run on your own image)
    python detect_plate.py --webcam               (live webcam detection)
"""

import cv2
import numpy as np
import argparse
import os


# ─────────────────────────────────────────────
# STEP 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────

def preprocess(img):
    """
    Convert to grayscale, apply bilateral filter to reduce noise
    while keeping edges sharp, then run Canny edge detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filter — smooths noise but preserves edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Canny edge detection
    edges = cv2.Canny(filtered, 30, 200)

    return gray, filtered, edges


# ─────────────────────────────────────────────
# STEP 2 — FIND PLATE CANDIDATES
# ─────────────────────────────────────────────

def find_plates(edges):
    """
    Find contours and filter by shape.
    Number plates are rectangular with aspect ratio between 2 and 6.
    """
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area, keep top 30 candidates only
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    candidates = []

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * perimeter, True)

        # Plates have 4 corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Plates are wide rectangles — filter by ratio and minimum size
            if 2.0 <= aspect_ratio <= 6.5 and w > 80 and h > 20:
                candidates.append((x, y, w, h))

    return candidates


# ─────────────────────────────────────────────
# STEP 3 — REMOVE DUPLICATE BOXES
# ─────────────────────────────────────────────

def deduplicate(regions, threshold=0.5):
    """Remove overlapping detections using IoU overlap check."""
    kept = []
    for box in regions:
        x1, y1, w1, h1 = box
        duplicate = False
        for kx, ky, kw, kh in kept:
            ix = max(x1, kx)
            iy = max(y1, ky)
            ix2 = min(x1+w1, kx+kw)
            iy2 = min(y1+h1, ky+kh)
            inter = max(0, ix2-ix) * max(0, iy2-iy)
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter/union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


# ─────────────────────────────────────────────
# STEP 4 — DRAW RESULTS
# ─────────────────────────────────────────────

def draw_results(img, plates):
    """Draw bounding boxes and labels on the image."""
    result = img.copy()
    for i, (x, y, w, h) in enumerate(plates):
        # Green bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Label background
        label = f"Number Plate #{i+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result, (x, y - th - 12), (x + tw + 8, y), (0, 255, 0), -1)
        cv2.putText(result, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Summary text
    status = f"Plates Detected: {len(plates)}" if plates else "No Plates Detected"
    color = (0, 255, 0) if plates else (0, 0, 255)
    cv2.rectangle(result, (0, 0), (320, 45), (0, 0, 0), -1)
    cv2.putText(result, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return result


# ─────────────────────────────────────────────
# STEP 5 — CROP THE PLATE
# ─────────────────────────────────────────────

def crop_plates(gray, plates, output_dir="cropped_plates"):
    """Save each detected plate as a separate cropped image."""
    if not plates:
        return
    os.makedirs(output_dir, exist_ok=True)
    for i, (x, y, w, h) in enumerate(plates):
        crop = gray[y:y+h, x:x+w]
        # Upscale for better readability
        crop = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        path = os.path.join(output_dir, f"plate_{i+1}.jpg")
        cv2.imwrite(path, crop)
        print(f"   💾 Plate #{i+1} cropped and saved: {path}")


# ─────────────────────────────────────────────
# STEP 6 — SAVE PIPELINE OUTPUT
# ─────────────────────────────────────────────

def save_pipeline(original, gray, filtered, edges, result, output_path):
    """Save a 4-panel image showing every step of the pipeline."""

    def label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(out, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        return out

    # Resize all to same dimensions
    h, w = original.shape[:2]
    target = (w, h)

    gray_bgr    = cv2.cvtColor(cv2.resize(gray,     target), cv2.COLOR_GRAY2BGR)
    filter_bgr  = cv2.cvtColor(cv2.resize(filtered, target), cv2.COLOR_GRAY2BGR)
    edges_bgr   = cv2.cvtColor(cv2.resize(edges,    target), cv2.COLOR_GRAY2BGR)
    result_r    = cv2.resize(result, target)
    original_r  = cv2.resize(original, target)

    # Top row: Original + Grayscale
    row1 = np.hstack([
        label(original_r, "1. Original Image"),
        label(gray_bgr,   "2. Grayscale")
    ])

    # Bottom row: Edges + Final Detection
    row2 = np.hstack([
        label(edges_bgr,  "3. Edge Detection (Canny)"),
        label(result_r,   "4. Plate Detected")
    ])

    panel = np.vstack([row1, row2])

    # Scale down if too large
    max_w = 1600
    if panel.shape[1] > max_w:
        scale = max_w / panel.shape[1]
        panel = cv2.resize(panel, (max_w, int(panel.shape[0] * scale)))

    cv2.imwrite(output_path, panel)
    print(f"✅ Pipeline output saved: {output_path}")


# ─────────────────────────────────────────────
# WEBCAM MODE
# ─────────────────────────────────────────────

def run_webcam():
    """Live webcam — press S to save, Q or ESC to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📷 Webcam running. Press S to save | Q or ESC to quit.\n")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, _, edges = preprocess(frame)
        plates = deduplicate(find_plates(edges))
        result = draw_results(frame, plates)

        cv2.imshow("Number Plate Detection  |  S=save  Q=quit", result)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            fname = f"webcam_capture_{count}.jpg"
            cv2.imwrite(fname, result)
            print(f"📸 Saved: {fname}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam closed.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Number Plate Detection using OpenCV")
    parser.add_argument("--image",  type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Live webcam mode")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output file name")
    args = parser.parse_args()

    if args.webcam:
        run_webcam()
        return

    # Default to included car.jpg if no image given
    image_path = args.image if args.image else "car.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Usage: python detect_plate.py --image your_car.jpg")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load: {image_path}")
        return

    print(f"\n📷 Image: {image_path}  ({img.shape[1]}x{img.shape[0]} px)")
    print("🔍 Running detection...\n")

    # Run pipeline
    gray, filtered, edges = preprocess(img)
    plates = deduplicate(find_plates(edges))
    result = draw_results(img, plates)

    # Print results
    print("=" * 50)
    if plates:
        print(f"✅ {len(plates)} number plate(s) detected!")
        for i, (x, y, w, h) in enumerate(plates):
            print(f"   Plate #{i+1}:  position=({x},{y})  size={w}x{h}px  ratio={w/h:.2f}")
        print()
        crop_plates(gray, plates)
    else:
        print("⚠️  No plates detected.")
        print("   Tips: use a clear front/rear facing image with good lighting.")
    print("=" * 50 + "\n")

    # Save pipeline visualization
    save_pipeline(img, gray, filtered, edges, result, args.output)
    print(f"\n✅ Done! Open '{args.output}' to see the full pipeline.\n")


if __name__ == "__main__":
    main()
