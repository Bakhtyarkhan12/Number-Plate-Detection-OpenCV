# Number Plate Detection Using Python & OpenCV

Detects vehicle number plates from images or a live webcam using edge detection and contour filtering. Tested on real CCTV footage.

---

## How it works

The detection runs in 4 steps:

1. **Grayscale + Bilateral Filter** — converts to grayscale and reduces noise while keeping edges intact
2. **Canny Edge Detection** — finds all edges in the image
3. **Contour Filtering** — finds rectangular contours with an aspect ratio between 2 and 6 (the shape of a number plate)
4. **Deduplication** — removes overlapping boxes using IoU overlap check

The final output is saved as a 4-panel image showing every step of the pipeline.

---

## Install

```bash
pip install -r requirements.txt
```

---

## How to Run

### Default — runs on the included car.jpg
```bash
python detect_plate.py
```

### Your own image
```bash
python detect_plate.py --image your_car.jpg
```

### Live webcam
```bash
python detect_plate.py --webcam
```
Press **S** to save a screenshot, **Q** or **ESC** to quit.

### Custom output filename
```bash
python detect_plate.py --image car.jpg --output result.jpg
```

---

## Output

Running the script produces:

- `output.jpg` — 4-panel pipeline visualization (Original → Grayscale → Edges → Detected)
- `cropped_plates/plate_1.jpg` — each detected plate cropped and upscaled 3x for readability

---

## Sample Result

The included `car.jpg` is a real CCTV image of a Citroën C2. The script successfully detects and crops the number plate **CZ 399DL**.

---

## Notes

- Works best on front or rear facing cars with clear plates
- Good lighting improves accuracy significantly
- You can tune the Canny thresholds (line 47) or aspect ratio range (line 73) in the script for different plate sizes
