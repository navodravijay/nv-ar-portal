"""NV. AR Portal V4 cinematic renderer: purely local NumPy/OpenCV image ops.

There is no camera, network, disk output or data collection in this module.
The '3D' effect is simulated with a perspective-transformed 2D plane.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

NAVY = (27, 13, 8)          # OpenCV BGR: brand navy #080d1b
BLUE = (255, 141, 77)       # brand blue #4d8dff
CYAN = (255, 224, 77)       # brand cyan #4de0ff
WHITE = (248, 247, 241)

EFFECT_NAMES = (
    "RGB GLITCH", "NEON EDGES", "THERMAL", "NEGATIVE",
    "INK SKETCH", "PIXEL SHIFT", "HOLOGRAM", "DATA GHOST",
    "AURORA FIELD", "SPECTRAL SPLIT",
)
FILTER_SIZE = (480, 270)
ECO_FILTER_SIZE = (320, 180)


def _alpha_blend(base: np.ndarray, over: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - alpha, over, alpha, 0)


def _draw_text(img: np.ndarray, message: str, point: tuple[int, int],
               scale: float = .45, color: tuple[int, int, int] = WHITE,
               thickness: int = 1) -> None:
    cv2.putText(img, message, point, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def effect_filter(src: np.ndarray, index: int, original_six_filter) -> np.ndarray:
    """Four stylized holographic effects and six original filters; input unchanged."""
    if not (0 <= index < len(EFFECT_NAMES)):
        raise ValueError("Effect index must be 0..9")
    if index < 6:
        return original_six_filter(src.copy(), index)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if index == 6:
        # A monochrome cyan/blue hologram with edges + faint camera texture.
        lum = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(lum, 50, 140)
        out = np.zeros_like(src)
        out[:, :, 0] = np.clip(lum.astype(np.float32) * .70 + 32, 0, 255).astype(np.uint8)
        out[:, :, 1] = np.clip(lum.astype(np.float32) * .62 + 12, 0, 255).astype(np.uint8)
        out[:, :, 2] = (lum.astype(np.float32) * .16).astype(np.uint8)
        out[edges > 0] = CYAN
        return out

    if index == 8:
        base = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
        lines = cv2.Canny(gray, 50, 135)
        base[lines > 0] = CYAN
        return cv2.addWeighted(base, .78, np.full_like(src, NAVY), .22, 0)
    if index == 9:
        b, g, r = cv2.split(src)
        result = cv2.merge((np.roll(b, -11, axis=1), g, np.roll(r, 12, axis=1)))
        lines = cv2.Canny(gray, 50, 120)
        result[lines > 0] = WHITE
        return cv2.addWeighted(result, .76, np.full_like(src, NAVY), .24, 0)

    # Glowing cyan contour map over dark navy with slight double-image offset.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 115)
    shifted = np.roll(edges, 5, axis=1)
    out = np.empty_like(src)
    out[:] = NAVY
    out[shifted > 0] = BLUE
    out[edges > 0] = CYAN
    return out


def _tech_grid(texture: np.ndarray, phase: float) -> None:
    """World-space UI lines on texture; the perspective warp skews them."""
    h, w = texture.shape[:2]
    overlay = texture.copy()
    shift = int((phase * 16.0) % 42)
    for x in range(shift, w, 42):
        cv2.line(overlay, (x, 0), (x, h - 1), BLUE, 1, cv2.LINE_AA)
    for y in range(0, h, 42):
        cv2.line(overlay, (0, y), (w - 1, y), CYAN, 1, cv2.LINE_AA)
    texture[:] = _alpha_blend(texture, overlay, .22)


def _scan_beam(texture: np.ndarray, phase: float) -> None:
    h, w = texture.shape[:2]
    center_y = int((phase * 74) % max(1, h))
    beam = texture.copy()
    # A small vertical falloff, not a huge full-image blur.
    for distance, strength in [(10, .08), (6, .13), (3, .2), (1, .5)]:
        y = center_y + distance // 2
        if 0 <= y < h:
            cv2.line(beam, (0, y), (w - 1, y), CYAN, max(1, distance // 2), cv2.LINE_AA)
    texture[:] = _alpha_blend(texture, beam, .29)


def _scanlines(texture: np.ndarray) -> None:
    texture[::5, :, :] = (texture[::5, :, :] * .79).astype(np.uint8)


def _quad_good(quad: np.ndarray | None) -> bool:
    if quad is None or np.asarray(quad).shape != (4, 2):
        return False
    if not np.isfinite(quad).all():
        return False
    pts = np.rint(quad).astype(np.int32)
    return bool(cv2.isContourConvex(pts) and cv2.contourArea(pts) > 2500)


def _orbital_energy(texture: np.ndarray, phase: float, strength: float) -> None:
    """Deterministic light particles and orbital paths drawn in portal texture.

    This stays in the perspective-mapped ROI; no full-frame particle processing.
    """
    h, w = texture.shape[:2]
    cx, cy = w // 2, h // 2
    radius_x, radius_y = int(w * .36), int(h * .33)
    glow = texture.copy()
    for ring in (0, 1):
        cv2.ellipse(glow, (cx, cy), (radius_x - ring*18, radius_y-ring*9),
                    0, int((phase*30+ring*110) % 360),
                    int((phase*30+ring*110) % 360)+205, BLUE if ring else CYAN,
                    1, cv2.LINE_AA)
    # Phase-based positions are reproducible and do not use camera/user data.
    for i in range(26):
        angle = phase * (.38 if i % 2 else -.51) + i * 2.3999632297
        depth = .19 + .75 * ((i*0.6180339887) % 1)
        px = int(cx + radius_x * depth * math.cos(angle))
        py = int(cy + radius_y * depth * math.sin(angle))
        radius = 1 if i % 4 else 2
        color = CYAN if i % 3 else WHITE
        cv2.circle(glow, (px, py), radius, color, -1, cv2.LINE_AA)
    texture[:] = _alpha_blend(texture, glow, min(.55, .17 + strength * .33))


def draw_holographic_portal(
        frame: np.ndarray,
        corners: np.ndarray,
        index: int,
        original_six_filter,
        elapsed: float = 0.0,
        perspective: bool = True,
        scanlines: bool = True,
        grid: bool = True,
        bloom: bool = True,
        energy: float = .65,
        orbitals: bool = True,
        eco: bool = False,
) -> np.ndarray:
    """Apply a live camera effect inside the hand-defined quadrilateral.

    Filters run at a predictable texture resolution, preventing CPU cost from
    growing with portal size. All overlays use a tight ROI, not a full-screen blur.
    """
    result = frame.copy()
    if not _quad_good(corners):
        return result

    h, w = frame.shape[:2]
    # Sufficient margin for local edge glow; keep the ROI bounded by frame.
    x0 = max(0, int(np.floor(corners[:, 0].min())) - 32)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - 32)
    x1 = min(w, int(np.ceil(corners[:, 0].max())) + 33)
    y1 = min(h, int(np.ceil(corners[:, 1].max())) + 33)
    if x1 <= x0 or y1 <= y0:
        return result

    base = frame[y0:y1, x0:x1].copy()
    region_h, region_w = base.shape[:2]
    local_pts = corners - np.array([x0, y0], dtype=np.float32)
    poly = np.rint(local_pts).astype(np.int32)
    mask = np.zeros((region_h, region_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_AA)

    if perspective:
        texture_size = ECO_FILTER_SIZE if eco else FILTER_SIZE
        tex = cv2.resize(base, texture_size, interpolation=cv2.INTER_AREA)
    else:
        tex = base.copy()
    tex = effect_filter(tex, index, original_six_filter)
    if grid:
        _tech_grid(tex, elapsed)
    if orbitals:
        _orbital_energy(tex, elapsed, float(np.clip(energy, .3, 1)))
    _scan_beam(tex, elapsed)
    if scanlines:
        _scanlines(tex)

    if perspective:
        tex_w, tex_h = texture_size
        src = np.float32([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]])
        matrix = cv2.getPerspectiveTransform(src, local_pts.astype(np.float32))
        tex = cv2.warpPerspective(tex, matrix, (region_w, region_h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    feather = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0 * .97
    region = np.uint8(np.clip(base * (1 - feather) + tex * feather, 0, 255))

    # Pulsing double stroke + local bloom. No full-frame glow buffer.
    pulse = (.58 + .22 * math.sin(elapsed * 2.8)) * (.75 + .50 * float(np.clip(energy, .3, 1)))
    if bloom:
        light = np.zeros_like(region)
        cv2.polylines(light, [poly], True, CYAN, 9, cv2.LINE_AA)
        light = cv2.GaussianBlur(light, (0, 0), 7.5)
        region = cv2.addWeighted(region, 1.0, light, .88 * pulse, 0)
        cv2.polylines(region, [poly], True, BLUE, 5, cv2.LINE_AA)
    cv2.polylines(region, [poly], True, CYAN, 2, cv2.LINE_AA)

    # Label anchored to the top edge of the screen; clip safely in tight ROIs.
    anchor_x = int(np.clip(min(local_pts[0, 0], local_pts[1, 0]) + 15, 0, max(0, region_w-1)))
    anchor_y = int(np.clip(min(local_pts[0, 1], local_pts[1, 1]) + 23, 14, max(14, region_h-8)))
    _draw_text(region, "NV. / QUANTUM LINK", (anchor_x, anchor_y), .32, CYAN, 1)
    result[y0:y1, x0:x1] = region
    return result


def draw_anchors(frame: np.ndarray, corners: np.ndarray, elapsed: float) -> None:
    """Small futuristic fasteners drawn above hand-occlusion compositing."""
    if not _quad_good(corners):
        return
    pts = np.rint(corners).astype(np.int32)
    for idx, (x, y) in enumerate(pts):
        x, y = int(x), int(y)
        radius = 8 + int(2 * math.sin(elapsed * 2.2 + idx * 1.5))
        cv2.circle(frame, (x, y), radius + 4, BLUE, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(3, radius - 5), CYAN, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, WHITE, -1, cv2.LINE_AA)
        d = 18
        for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(frame, (x + sx * (d - 5), y + sy * (d - 5)),
                     (x + sx * d, y + sy * d), CYAN, 1, cv2.LINE_AA)


def vignette_factor(height: int, width: int) -> np.ndarray:
    """Cached per camera resolution for efficient cinematic edge lighting."""
    ys, xs = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
    distance = np.clip((xs / 1.15) ** 2 + (ys / 1.2) ** 2, 0, 1.75)
    return (1 - .35 * distance).astype(np.float32)[:, :, None]


def apply_vignette(frame: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def render_hud(frame: np.ndarray, effect_index: int, fps: float,
               tracking: bool, automatic: bool, perspective: bool,
               gesture_status: str, gesture_progress: float, gestures: bool,
               eco: bool, orbitals: bool, energy: float, debug: bool,
               show_help: bool, elapsed: float) -> None:
    """Compact NV. V4 HUD; status is derived from locally detected hand landmarks."""
    h, w = frame.shape[:2]
    head = min(102, max(70, int(h*.19)))
    foot = min(80, max(56, int(h*.15)))
    layer = frame.copy()
    cv2.rectangle(layer, (0, 0), (w, head), NAVY, -1)
    cv2.rectangle(layer, (0, h-foot), (w, h), NAVY, -1)
    frame[:] = cv2.addWeighted(layer, .90, frame, .10, 0)
    cv2.line(frame, (0, head-1), (w, head-1), BLUE, 1, cv2.LINE_AA)
    cv2.line(frame, (0, h-foot), (w, h-foot), BLUE, 1, cv2.LINE_AA)
    shrink = min(1.0, w / 850.)
    _draw_text(frame, "NV.", (15, 43), 1.19*shrink, CYAN, 2)
    _draw_text(frame, "SHARP SOLUTIONS", (102, 31), .58*shrink, WHITE, 2)
    _draw_text(frame, "AR PORTAL  //  CINEMATIC V4", (102, 57), .38*shrink, BLUE)
    _draw_text(frame, f"{fps:.0f} FPS", (max(15, w-94), 26), .49*shrink, WHITE)
    _draw_text(frame, "LOCAL CAMERA", (max(15, w-132), 47), .30*shrink, CYAN)
    _draw_text(frame, f"FX {effect_index+1:02d}  {EFFECT_NAMES[effect_index]}",
               (16, min(head-12, 79)), .41*shrink, CYAN)
    status = "PORTAL LOCKED // TWO HANDS" if tracking else gesture_status
    _draw_text(frame, status, (16, h-foot+22), .43*shrink, CYAN if tracking else WHITE)
    right_status = f"{'AUTO' if automatic else 'MANUAL'} / {'ECO' if eco else 'HIGH'} / {'G' if gestures else '-'}"
    _draw_text(frame, right_status, (max(15,w-200), h-foot+23), .33*shrink, BLUE)
    # Dynamic energy gauge is an aesthetic interaction, NOT a physical measurement.
    gauge_x, gauge_y, gauge_w = 16, h-foot+31, min(152,max(80,w//5))
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x+gauge_w, gauge_y+6), (70, 52, 35), -1)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x+int(gauge_w*energy), gauge_y+6), CYAN, -1)
    if gesture_progress > 0 and not tracking:
        cx, cy = min(w-28, 205), h-foot+18
        cv2.circle(frame, (cx, cy), 12, BLUE, 2, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (12,12), -90, 0,
                    int(360*np.clip(gesture_progress,0,1)), CYAN, 3, cv2.LINE_AA)
    if show_help:
        _draw_text(frame, "1-9,0 FX  SPACE AUTO  M GESTURE  P WARP  E ORBITS",
                   (16, h-20), .37*shrink, WHITE)
        _draw_text(frame, "L ECO  G GRID  B BLOOM  H HAND  D DEBUG  F FULL  Q QUIT",
                   (16, h-6), .30*shrink, BLUE)
    else:
        _draw_text(frame, "TAB: CONTROLS  /  Q: EXIT", (16, h-14), .39*shrink, WHITE)
