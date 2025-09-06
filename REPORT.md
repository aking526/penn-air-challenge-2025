## PennAir 2024 Challenge — Parts 1 & 2 Write‑Up

This document explains the shape detection solution implemented in `core.py` for Part 1 (static image) and Part 2 (streaming video). It details the end‑to‑end pipeline, each function in the `Processor` class, and the relevant math/theory underpinning the approach.

### High‑level overview

- **Goal**: Detect solid colored shapes on natural backgrounds, trace their outlines, and mark their centers.
- **Approach**: Automatic background/foreground separation in CIELAB space via contrast to border color, mask smoothing and cleanup, contour extraction, quality filtering, polygon simplification and straightening, and centroid estimation.
- **Why it works**: Shapes are chromatically distinct from the grass/background. Working in CIELAB approximates perceptual differences; Otsu thresholding on a LAB‑distance image separates shapes robustly. Morphology and connected‑component filtering stabilize results across frames.

## End‑to‑end pipeline

1. Convert the input image to CIELAB and compute a per‑pixel distance to a background color sampled from corners.
2. Normalize the distance image to 8‑bit and run Otsu thresholding to produce a binary foreground mask.
3. Denoise and regularize the mask via median blur, morphological closing/opening, optionally at a higher resolution (upsampling) for smoother boundaries.
4. Remove small connected components (by absolute and/or fractional area) to suppress speckles.
5. Extract contours, filter them by geometric/contrast quality, and compute centroids (moments).
6. Simplify polygonal chains (Ramer–Douglas–Peucker) and merge nearly collinear vertices to straighten edges.
7. Draw overlays with contour strokes and center markers; optionally, compute a thin edge mask for debugging.

## Processor class — function‑by‑function

Below, parentheses indicate important parameters and defaults.

### `_to_bgr(img, is_bgr=True)`
- Ensures a 3‑channel BGR image regardless of input; converts from grayscale or RGB as needed.
- Returns `(bgr, True)` for downstream consistency.

### `_auto_nonbg_mask(img, is_bgr=True, border_frac=0.05, return_extras=False)`
- Core background separation in LAB space.
- Samples the four corner blocks of size `t = max(2, int(min(H,W) * border_frac))` and computes the median LAB vector of these samples as background color `b`.
- For each pixel with LAB value `x`, computes a perceptual distance to background:
  - $ d(x) = \lVert x - b \rVert_2 = \sqrt{(L-L_b)^2 + (a-a_b)^2 + (b-b_b)^2} $
- Linearly normalizes distances to 8‑bit: $ d_{u8} = 255 \cdot \frac{d - d_{\min}}{d_{\max} - d_{\min}} $ with a guard for degenerate ranges.
- Applies Otsu thresholding to obtain a binary foreground mask. Otsu selects $ t^* $ maximizing between‑class variance
  - $ \sigma_b^2(t) = \omega_0(t)\,\omega_1(t) \,[\mu_0(t) - \mu_1(t)]^2 $
  which is equivalent to minimizing within‑class variance.
- Returns `mask` or `(mask, dist_u8, otsu_t)` when `return_extras=True`.

### `_find_contours(mask, include_holes=False)`
- Extracts contours using `cv2.findContours` with retrieval mode `RETR_EXTERNAL` (or `RETR_TREE` if `include_holes=True`) and `CHAIN_APPROX_SIMPLE` to discard redundant points.
- Returns `(contours, hierarchy)`.

### `_to_xy_arrays(contours)`
- Converts OpenCV contour format `(N,1,2)` to float32 `(N,2)` arrays for easier geometric processing.

### `_draw_overlays(img, contours, centers=None, is_bgr=True, thickness=2)`
- Draws contour polylines in red and optional center points with white markers and coordinate labels.
- Returns an image (BGR or RGB depending on `is_bgr`).

### `_smooth_mask(mask, smooth_px=5, close_px=5, open_px=3, upsample=1)`
- Regularizes the binary mask with three steps:
  - Optional upsample by integer `upsample` to reduce aliasing and enable subpixel‑quality shaping when downsampled back.
  - Morphological closing (`close_px`) to fill small gaps along edges, followed by opening (`open_px`) to remove small protrusions while preserving corners.
  - Median blur (`smooth_px`, odd) to suppress salt‑and‑pepper noise without shifting edges.
- Morphology uses rectangular structuring elements of the given sizes.

### `_remove_small_components(mask, min_area_px=0.0, min_area_frac=None)`
- Removes connected components whose area is below a threshold $ A_{\min} = \max(\text{min\_area\_px}, \text{min\_area\_frac} \cdot H\,W) $.
- This suppresses flickering speckles and stabilizes tracking across frames.

### `_approx_rdp(cnt, frac=0.02)`
- Applies Ramer–Douglas–Peucker (RDP) polygonal simplification with epsilon set as a fraction of the contour perimeter:
  - $ \varepsilon = f \cdot P $, where $ P = \text{arcLength}(\text{cnt}) $.
- Reduces vertex count while retaining salient corners.

### `_merge_collinear_vertices(poly_xy, angle_tol_deg=6.0, dist_tol=1.0)`
- Straightens polygon chains by removing nearly collinear or near‑duplicate vertices.
- For consecutive vectors $ v_1, v_2 $, compute
  - $ \cos\theta = \frac{v_1 \cdot v_2}{\lVert v_1 \rVert\, \lVert v_2 \rVert} $, $ \theta \in [0, 180^\circ] $
  - Drop a vertex if $ |180^\circ - \theta| < \text{angle\_tol\_deg} $ or if adjacent vertices are within `dist_tol`.

### `process_frame(...)`
- Orchestrates the full pipeline and returns `(outlines, debug)` where `outlines` are simplified polygon vertices and `debug` includes the final `mask`, `overlay`, and intermediate `dist_u8`, `otsu_t` (and optional `edge_mask`).
- Key parameters:
  - `simplify`, `approx_epsilon_frac`, `merge_collinear`, `angle_tol_deg`, `dist_tol`: control polygon simplification/straightening.
  - `smooth_px`, `close_px`, `open_px`, `upsample`: control mask regularization quality.
  - `min_area`, `min_area_frac`: reject small shapes via absolute or frame‑relative thresholds.
  - `use_quality_filters`, `min_solidity`, `min_circularity`, `contrast_margin_u8`: robustly reject non‑shapes.
- Quality filters:
  - **Solidity**: $ \text{solidity} = \frac{A}{A_{\text{hull}}} $. Rejects highly concave/noisy blobs.
  - **Circularity**: $ C = \frac{4\pi A}{P^2} $. Enables exclusion of overly round or irregular blobs (disabled by default; set a positive threshold to use).
  - **Contrast margin (LAB distance)**: Let $ \bar d $ be the mean of `dist_u8` inside the contour and $ t $ be Otsu’s threshold. Require $ \bar d \ge t + m $ where `m = contrast_margin_u8`.
- Centroid estimation via image moments:
  - $ c_x = m_{10}/m_{00},\quad c_y = m_{01}/m_{00} $

## Part 1 — Static image

Entry point: `part_1()`

- Loads `PennAir 2024 App Static.png`, constructs a `Processor`, and runs `process_frame` with:
  - `is_bgr=False` (display convenience; overlay converted to RGB for Matplotlib)
  - `simplify=True`, `approx_epsilon_frac=0.01` (preserve corners with modest simplification)
  - `return_edge_mask=True` (debugging), `smooth_px=7` (stronger median denoising)
- Displays the contours overlay with shape centers.

Expected behavior: clean separation of solid shapes from grass, well‑shaped contours with straight edges, and clearly marked centroids.

## Part 2 — Streaming video

Entry point: `part_2(save: bool = False, output_path: str | None = None)`

- Opens `PennAir 2024 App Dynamic.mp4` as a stream; obtains FPS (falls back to 30 fps if unavailable).
- For each frame:
  - Runs the same processing pipeline with parameters tuned for stability:
    - `simplify=True`, `approx_epsilon_frac=0.01`
    - `smooth_px=7`
    - `min_area=300.0` and `min_area_frac=0.001` (0.1% of frame area) to suppress temporal speckle.
  - Shows the overlay in a window; optionally writes frames to `PennAir 2024 App Dynamic.overlay.mp4` when `save=True`.
- Uses lazy `VideoWriter` initialization after first processed frame to match the exact frame size.

Notes on streaming robustness:

- Combining absolute and fractional area thresholds keeps detections stable across zoom/scale changes and brief noise.
- Morphology and median blur reduce frame‑to‑frame jitter along straight edges.
- The LAB‑distance/Otsu combination adapts to global illumination changes better than fixed thresholds in RGB.

## Additional theory and implementation details

- **CIELAB color space**: Designed to be perceptually (more) uniform; Euclidean distances correlate better with perceived differences than in RGB. This is particularly effective when background color is coherent (e.g., grass) and shapes are chromatically distinct.
- **Corner background sampling**: Using the four corners reduces bias from shapes located centrally and avoids relying on global statistics that can be skewed by large foreground areas.
- **Morphological operators**: Closing followed by opening is a common sequence to fill small gaps then remove small artifacts, preserving larger geometric features. Rectangular kernels favor straightening of rectilinear edges.
- **Ramer–Douglas–Peucker**: Reduces vertex count based on maximum deviation from the polyline; using an epsilon tied to perimeter makes the simplification scale‑aware across shapes and frames.
- **Collinearity merging**: A simple angle‑based pruning further regularizes polygons into straight‑edged shapes, improving readability and reducing label jitter.
- **Image moments**: The centroid from raw spatial moments is stable and inexpensive to compute for each contour.

## Parameters and tuning tips

- **`approx_epsilon_frac`**: Increase to simplify more aggressively; decrease to preserve fine details.
- **`smooth_px`, `close_px`, `open_px`**: Larger values produce smoother, straighter borders but can round corners; keep `smooth_px` odd.
- **`upsample`**: Values 2–3 can yield noticeably smoother edges; costs additional CPU per frame.
- **`min_area` / `min_area_frac`**: Use higher thresholds to reject small, noisy detections; pair fractional threshold with changing resolutions.
- **`min_solidity`**: Raise to reject more concave/noisy blobs; typical 0.5–0.9.
- **`min_circularity`**: Enable and tune when you need to reject near‑circular blobs.
- **`contrast_margin_u8`**: Increase to require stronger foreground‑background separation.

## Limitations and potential improvements

- Corner/background sampling assumes corners are background. If shapes touch corners, the background model can be biased; mitigate by shrinking `border_frac` or using robust clustering of LAB space.
- Illumination changes that alter background strongly may shift Otsu threshold; adaptive mixtures (e.g., Gaussian Mixture Models) or temporal background modeling can help for more challenging videos.
- For highly textured backgrounds, adding texture/edge cues (e.g., gradient magnitude) to the decision function can improve separability.

## File and function references

- Code: `core.py`
- Entry points:
  - Part 1: `part_1()` — static image visualization
  - Part 2: `part_2(save=False, output_path=None)` — streaming video processing and optional saving


