import cv2 
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, cast

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ========== Basic Functions  ==========

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

# ========== Core Class ==========

class Processor:
    def __init__(self, img: np.ndarray):
        self.img = img

    def _to_bgr(self, img, is_bgr=True):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), True
        if is_bgr:
            return img, True
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), True
    
    def _auto_nonbg_mask(self, img, is_bgr=True, border_frac=0.05, return_extras=False):
        bgr, _ = self._to_bgr(img, is_bgr=is_bgr)
        H, W = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        t = max(2, int(min(H, W) * border_frac))
        # Sample from corners, which are more robustly background
        corners = np.concatenate([
            lab[:t, :t].reshape(-1, 3),       # Top-left
            lab[:t, W - t:].reshape(-1, 3),  # Top-right
            lab[H - t:, :t].reshape(-1, 3),   # Bottom-left
            lab[H - t:, W - t:].reshape(-1, 3), # Bottom-right
        ], axis=0)
        bg_lab = np.median(corners, axis=0)

        dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
        # Manual normalization to avoid type issues with cv2.normalize and None dst
        dmin = float(dist.min())
        dptp = float(dist.max() - dmin)
        if dptp < 1e-6:
            dist_u8 = np.zeros_like(dist, dtype=np.uint8)
        else:
            dist_u8 = (((dist - dmin) / dptp) * 255.0).astype(np.uint8)
        otsu_t, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if return_extras:
            return mask, dist_u8, float(otsu_t)
        return mask
    
    def _find_contours(self, mask, include_holes=False):
        mode = cv2.RETR_TREE if include_holes else cv2.RETR_EXTERNAL
        # Use SIMPLE to drop redundant points early
        method = cv2.CHAIN_APPROX_SIMPLE
        out = cv2.findContours(mask, mode, method)
        contours = out[0] if len(out) == 2 else out[1]
        hierarchy = out[1] if len(out) == 2 else out[2]
        return contours, hierarchy
    
    def _to_xy_arrays(self, contours):
        polys = []
        for c in contours:
            c = c.reshape(-1, 2)
            polys.append(c.astype(np.float32))
        return polys

    def _draw_overlays(self, img, contours, centers=None, is_bgr=True, thickness=2):
        bgr, _ = self._to_bgr(img, is_bgr=is_bgr)
        overlay = bgr.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness)
        
        if centers:
            for cX, cY in centers:
                # Draw the center of the shape on the image
                cv2.circle(overlay, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(overlay, f"({cX}, {cY})", (cX - 40, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return overlay if is_bgr else cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    def _smooth_mask(self, mask, smooth_px=5, close_px=5, open_px=3, upsample=1):
        """
        Heavily denoise/smooth the binary mask to kill stair-steps and burrs.
        `smooth_px` is median blur size; `close_px`/`open_px` are morph kernel sizes.
        `upsample`>1: process at higher res for subpixel-quality borders, then downsample.
        """
        m = mask.copy()
        if upsample > 1:
            m = cv2.resize(m, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_NEAREST)

        if close_px and close_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

        if open_px and open_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_px, open_px))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

        if smooth_px and smooth_px > 1 and smooth_px % 2 == 1:
            m = cv2.medianBlur(m, smooth_px)

        if upsample > 1:
            m = cv2.resize(m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m
    
    def _remove_small_components(self, mask, *, min_area_px=0.0, min_area_frac=None):
        """
        Remove connected components smaller than the area threshold.
        If min_area_frac is provided, threshold is max(min_area_px, min_area_frac * H * W).
        """
        if mask is None:
            return mask
        H, W = mask.shape[:2]
        area_thresh = float(min_area_px) if min_area_px else 0.0
        if min_area_frac is not None:
            area_thresh = max(area_thresh, float(min_area_frac) * float(H * W))

        if area_thresh <= 1.0:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= area_thresh:
                keep[labels == label] = 255
        return keep
    
    def _approx_rdp(self, cnt, frac=0.02):
        eps = frac * cv2.arcLength(cnt, True)
        return cv2.approxPolyDP(cnt, eps, True)

    def _merge_collinear_vertices(self, poly_xy, angle_tol_deg=6.0, dist_tol=1.0):
        """
        Remove vertices where the turn angle is ~180° (nearly straight) or
        where consecutive vertices are extremely close.
        Works on an (N,2) float array; returns an (M,2) float array.
        """
        pts = poly_xy.astype(np.float32)
        N = len(pts)
        if N <= 3:
            return pts

        keep = []
        for i in range(N):
            p_prev = pts[(i - 1) % N]
            p      = pts[i]
            p_next = pts[(i + 1) % N]

            # drop near-duplicate points
            if np.linalg.norm(p - p_prev) < dist_tol:
                continue

            v1 = p - p_prev
            v2 = p_next - p
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                keep.append(p); continue

            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))  # 0..180
            # If angle ~ 180°, it's almost straight -> drop it
            if abs(180.0 - ang) < angle_tol_deg:
                continue
            keep.append(p)

        keep = np.array(keep, dtype=np.float32)
        # If we were too aggressive, fall back
        if len(keep) < 3:
            return pts
        return keep

    def process_frame(
        self,
        *,
        is_bgr=True,
        manual_hsv_ranges=None,  # (kept for parity; not used in this quick fix path)
        include_holes=False,
        simplify=True,
        approx_epsilon_frac=0.02,   # <-- more aggressive default
        min_area=150.0,             # <-- base absolute area threshold for contours
        min_area_frac=None,         # <-- optional fractional area threshold (of frame area)
        return_edge_mask=False,
        # NEW knobs:
        smooth_px=5,                # median blur on mask (odd int; try 5–9)
        close_px=7,                 # closing kernel (try 5–9 for straight edges)
        open_px=3,                  # opening kernel (small to preserve corners)
        upsample=2,                 # process at 2× for smoother borders, then downsample
        merge_collinear=True,       # merge almost-straight vertices after RDP
        angle_tol_deg=6.0,
        dist_tol=1.0,
        # Quality filters (robust across videos; conservative defaults)
        use_quality_filters=True,
        min_solidity=0.55,          # area / convexHull(area)
        min_circularity=0.0,        # 4πA / P^2; set >0 to enable
        contrast_margin_u8=8,       # require mean LAB-distance ≥ otsu+margin inside contour
    ):
        # 1) Build initial mask (auto background separation)
        mask_extras = self._auto_nonbg_mask(self.img, is_bgr=is_bgr, return_extras=True)
        mask, dist_u8, otsu_t = cast(Tuple[np.ndarray, np.ndarray, float], mask_extras)

        # 2) Strong smoothing to remove zippering along straight edges
        mask = self._smooth_mask(mask, smooth_px=smooth_px, close_px=close_px, open_px=open_px, upsample=upsample)

        # 2b) Remove tiny connected components to suppress speckle noise
        mask = self._remove_small_components(mask, min_area_px=min_area, min_area_frac=min_area_frac)

        # 3) Optional thin edge for visualization
        edge_mask = None
        if return_edge_mask:
            kernel = np.ones((3, 3), np.uint8)
            edge_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        # 4) Contours (SIMPLE reduces redundant points)
        contours, hierarchy = self._find_contours(mask, include_holes=include_holes)

        # 5) Filter, simplify, and straighten polygons
        cleaned = []
        centers = []
        # Compute effective min area for contour filtering (same as used for CC removal)
        H, W = mask.shape[:2]
        effective_min_area = float(min_area)
        if min_area_frac is not None:
            effective_min_area = max(effective_min_area, float(min_area_frac) * float(H * W))

        for c in contours:
            area_c = float(cv2.contourArea(c))
            if area_c < effective_min_area:
                continue

            # Optional quality filters to reject non-shapes
            if use_quality_filters:
                # Solidity
                hull = cv2.convexHull(c)
                hull_area = float(cv2.contourArea(hull))
                solidity = (area_c / hull_area) if hull_area > 1e-6 else 0.0
                if solidity < float(min_solidity):
                    continue

                # Circularity (disabled by default)
                if min_circularity and float(min_circularity) > 0.0:
                    perim = float(cv2.arcLength(c, True))
                    circ = (4.0 * np.pi * area_c / (perim * perim)) if perim > 1e-6 else 0.0
                    if circ < float(min_circularity):
                        continue

                # Low-contrast rejector relative to background distance
                if contrast_margin_u8 and float(contrast_margin_u8) > 0.0:
                    contour_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)
                    masked_vals = dist_u8[contour_mask > 0]
                    mean_dist = float(masked_vals.mean()) if masked_vals.size > 0 else 0.0
                    if mean_dist < (float(otsu_t) + float(contrast_margin_u8)):
                        continue

            # Calculate the center of the contour
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))

            c_use = c
            if simplify:
                c_use = self._approx_rdp(c_use, approx_epsilon_frac)
            poly = c_use.reshape(-1, 2).astype(np.float32)
            if merge_collinear and len(poly) >= 4:
                poly = self._merge_collinear_vertices(poly, angle_tol_deg=angle_tol_deg, dist_tol=dist_tol)
            cleaned.append(poly.reshape(-1, 1, 2).astype(np.int32))  # for drawing

        outlines = self._to_xy_arrays(cleaned)
        overlay = self._draw_overlays(self.img, cleaned, centers=centers, is_bgr=is_bgr, thickness=2)
        debug = {"mask": mask, "overlay": overlay, "dist_u8": dist_u8, "otsu_t": otsu_t}
        if return_edge_mask:
            debug["edge_mask"] = edge_mask
        return outlines, debug
    
# ========== Part 1: Shape Detection on Static Image ==========
# Use this image (PennAir 2024 App Static.png) that features solid shapes on a grassy background.
# Implement an algorithm (we recommend OpenCV - cv2) to detect the shapes.
# Trace the outlines of the detected shapes.
# Locate and mark the centers of the shapes.

def part_1() -> None:
    # Run the tracer
    img = load_image("PennAir 2024 App Static.png")
    processor = Processor(img)
    outlines, debug = processor.process_frame(
        is_bgr=False,
        simplify=True,
        approx_epsilon_frac=0.01,
        return_edge_mask=True,
        smooth_px=7,
    )

    # Show overlay (convert to RGB if it’s BGR)
    overlay = debug["overlay"]
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_rgb)
    plt.title("Contours Overlay")
    plt.axis("off")
    plt.show()

# ========== Part 2: Shape Detection on Dynamic Video ==========
# - Apply the algorithm to the video file ( PennAir 2024 App Dynamic.mp4).
# - Ensure the algorithm consistently detects, traces, and locates the centers of the shapes throughout the video.
# - Please treat the video file as a streamed input - this part should involve applying your prior algorithm to each frame. Just like how our aircraft doesn’t have the entire video journey ahead of time, we want to feed the algorithm each frame one at the time.

def part_2(save: bool = False, output_path: str | None = None) -> None:
    video_path = "PennAir 2024 App Dynamic.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error opening video stream or file: {video_path}")
        return

    # Determine FPS (fallback if unavailable)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0

    writer = None  # Initialized lazily after first frame to get exact size

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processor = Processor(frame)
            outlines, debug = processor.process_frame(
                is_bgr=True,
                simplify=True,
                approx_epsilon_frac=0.01,
                return_edge_mask=False,
                smooth_px=7,
                # Suppress small speckles that flicker by using both absolute and fractional area
                min_area=300.0,
                min_area_frac=0.001,  # 0.1% of frame area
            )
            
            overlay = debug["overlay"]
            cv2.imshow('Frame', overlay)

            # Lazily create writer with correct size once we have the first frame
            if save and writer is None:
                H, W = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
                out_path = output_path or "PennAir 2024 App Dynamic.overlay.mp4"
                tmp_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                if not tmp_writer.isOpened():
                    logging.error(f"Failed to open VideoWriter: {out_path}")
                else:
                    writer = tmp_writer
                    logging.info(f"Writing output video to: {out_path} at {fps:.2f} fps, size {W}x{H}")

            if writer is not None:
                writer.write(overlay)

            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

def part_3(save: bool = False, output_path: str | None = None) -> None:
    video_path = "PennAir 2024 App Dynamic Hard.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error opening video stream or file: {video_path}")
        return

    # Determine FPS (fallback if unavailable)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0

    writer = None  # Initialized lazily after first frame to get exact size

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processor = Processor(frame)
            outlines, debug = processor.process_frame(
                is_bgr=True,
                simplify=True,
                approx_epsilon_frac=0.01,
                return_edge_mask=False,
                smooth_px=7,
                # Suppress small speckles that flicker by using both absolute and fractional area
                min_area=300.0,
                min_area_frac=0.001,  # 0.1% of frame area
            )
            
            overlay = debug["overlay"]
            cv2.imshow('Frame', overlay)

            # Lazily create writer with correct size once we have the first frame
            if save and writer is None:
                H, W = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
                out_path = output_path or "PennAir 2024 App Dynamic Hard.overlay.mp4"
                tmp_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                if not tmp_writer.isOpened():
                    logging.error(f"Failed to open VideoWriter: {out_path}")
                else:
                    writer = tmp_writer
                    logging.info(f"Writing output video to: {out_path} at {fps:.2f} fps, size {W}x{H}")

            if writer is not None:
                writer.write(overlay)

            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

def part_4() -> None:
    pass

def part_5() -> None:
    pass

def part_6() -> None:
    pass