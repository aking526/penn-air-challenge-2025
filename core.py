import cv2 
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found at {path}")
    return img

def show_image(img: np.ndarray) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Part 1: Shape Detection on Static Image:
def part_1() -> None:
    # Use this image (PennAir 2024 App Static.png) that features solid shapes on a grassy background.
    # Implement an algorithm (we recommend OpenCV - cv2) to detect the shapes.
    # Trace the outlines of the detected shapes.
    # Locate and mark the centers of the shapes.

    img = load_image("PennAir 2024 App Static.png")

    # 1) Build a robust grass mask using multiple color models
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    hsv_green = cv2.inRange(hsv, lower_green, upper_green)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    # Otsu on 'a' tends to split greenish (lower a) from non-green; invert to make green=255
    _, lab_green = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    b, g, r = cv2.split(img)
    exg = (2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16))
    exg_min = int(exg.min())
    exg_max = int(exg.max())
    range_val = max(1, exg_max - exg_min)
    exg = ((exg - exg_min) * 255 / range_val).astype(np.uint8)
    _, exg_green = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    grass_mask = cv2.bitwise_or(hsv_green, lab_green)
    grass_mask = cv2.bitwise_or(grass_mask, exg_green)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel_open)
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2) Shapes mask is the inverse of grass
    shape_mask = cv2.bitwise_not(grass_mask)

    # 2b) Background-agnostic foreground extraction via large-kernel background subtraction
    # This highlights regions that differ from the locally smoothed background, regardless of background color
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=21, sigmaY=21)
    fg = cv2.absdiff(img, bg)
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    fg_gray = cv2.GaussianBlur(fg_gray, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Union with inverse grass to improve recall while remaining background-agnostic
    shape_mask = cv2.bitwise_or(shape_mask, fg_mask)

    # 3) Remove tiny blobs via connected components
    h, w = shape_mask.shape[:2]
    min_area = max(150, int(0.0005 * w * h))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(shape_mask, connectivity=8)
    cleaned_mask = np.zeros_like(shape_mask)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255
            kept += 1
    logging.info(f"Connected components kept (area >= {min_area}): {kept}")

    # 4) Optional edge enhance, then contours
    blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    composite = cv2.bitwise_or(cleaned_mask, edges)

    contours, _ = cv2.findContours(composite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"Total contours found: {len(contours)}")

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Filter out very ragged grass fragments via solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        if solidity < 0.75:
            continue

        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            centers.append((cx, cy))

    logging.info(f"Number of shapes detected: {len(centers)}")
    for i, (x, y) in enumerate(centers):
        logging.info(f"Shape {i+1} center: ({x}, {y})")

    show_image(img)


def part_2() -> None:
    pass

def part_3() -> None:
    pass

def part_4() -> None:
    pass

def part_5() -> None:
    pass

def part_6() -> None:
    pass