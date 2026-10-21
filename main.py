"""NV. AR PORTAL V4 | CINEMATIC UPGRADE

An original, webcam-driven *2.5D* holographic portal. Uses a hand landmark
model locally. This application does not capture audio or implement uploads,
recording, facial identification, remote inference or background account access.
The separate download_model.py fetches the AI model with explicit user action.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from cinematic import (EFFECT_NAMES, apply_vignette, draw_anchors,
                       draw_holographic_portal, render_hud, vignette_factor)
from interaction import (OneEuroQuadSmoother, PinchEffectGesture,
                         draw_hand_skeleton, portal_energy)

MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
CAMERA_INDEX = 0
CAMERA_WIDTH, CAMERA_HEIGHT = 960, 540
PORTAL_MIN_AREA = 2500
AUTO_EFFECT_SECONDS = 5.0
WINDOW_NAME = "NV. AR PORTAL V4 | CINEMATIC LOCAL EXPERIENCE"

# OpenCV colors are BGR, not RGB.
NAVY = (27, 13, 8)     # Brand #080d1b
CYAN = (255, 224, 77)  # Brand #4de0ff


def landmark_point(hand, index: int, width: int, height: int) -> np.ndarray:
    item = hand[index]
    return np.array([np.clip(item.x, 0, 1) * (width - 1),
                     np.clip(item.y, 0, 1) * (height - 1)], dtype=np.float32)


def valid_quad(quad: np.ndarray | None) -> bool:
    if quad is None or np.asarray(quad).shape != (4, 2):
        return False
    if not np.isfinite(quad).all():
        return False
    poly = np.rint(quad).astype(np.int32)
    return (bool(cv2.isContourConvex(poly))
            and cv2.contourArea(poly) >= PORTAL_MIN_AREA
            and np.linalg.norm(quad[0] - quad[1]) > 65
            and np.linalg.norm(quad[3] - quad[2]) > 65)


def hand_portal(landmarks, width: int, height: int) -> np.ndarray | None:
    """Top/bottom thumb/index pair from each hand defines the four corners."""
    if len(landmarks) != 2:
        return None
    left, right = sorted(landmarks, key=lambda hand: hand[0].x)
    edges = []
    for hand in (left, right):
        thumb = landmark_point(hand, 4, width, height)
        index = landmark_point(hand, 8, width, height)
        if np.linalg.norm(thumb - index) < 24:
            return None
        top, bottom = sorted((thumb, index), key=lambda p: float(p[1]))
        edges.append((top, bottom))
    quad = np.asarray([edges[0][0], edges[1][0], edges[1][1], edges[0][1]],
                      dtype=np.float32)
    return quad if valid_quad(quad) else None


def apply_filter(src: np.ndarray, index: int) -> np.ndarray:
    """Original V3's first six filters (retained to preserve working behavior)."""
    if index == 0:
        b, g, r = cv2.split(src)
        result = cv2.merge((np.roll(b, -8, axis=1), g, np.roll(r, 8, axis=1)))
        for y in range(24, result.shape[0], 58):
            result[y:y+2] = np.roll(result[y:y+2], 11, axis=1)
        return result
    if index == 1:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        lines = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 115)
        result = np.full_like(src, NAVY)
        result[lines > 0] = CYAN
        return result
    if index == 2:
        return cv2.applyColorMap(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
    if index == 3:
        return cv2.bitwise_not(src)
    if index == 4:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inverted, (19, 19), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    if index == 5:
        h, w = src.shape[:2]
        low = cv2.resize(src, (max(1, w // 18), max(1, h // 18)), interpolation=cv2.INTER_AREA)
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST)
    raise ValueError("Base effect index must be 0..5")


def approximate_hand_mask(hands, width: int, height: int) -> np.ndarray:
    """Fast geometric silhouette, NOT pixel-accurate segmentation."""
    mask = np.zeros((height, width), dtype=np.uint8)
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
              (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for hand in hands:
        positions = np.array([landmark_point(hand, i, width, height) for i in range(21)],
                             dtype=np.float32)
        palm_width = float(np.linalg.norm(positions[5] - positions[17]))
        thickness = int(np.clip(palm_width * .33, 10, 34))
        palm = np.rint(positions[[0, 1, 2, 5, 9, 13, 17]]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(palm), 255)
        for chain in chains:
            pts = np.rint(positions[list(chain)]).astype(np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(mask, (int(x), int(y)), thickness // 2, 255, -1, cv2.LINE_AA)
    return mask


def hands_in_foreground(effected: np.ndarray, background: np.ndarray, hands) -> np.ndarray:
    h, w = background.shape[:2]
    mask = approximate_hand_mask(hands, w, h)
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[:, :, None] / 255
    return np.uint8(np.clip(effected * (1 - alpha) + background * alpha, 0, 255))


def draw_trails(frame: np.ndarray, trail_history: deque, elapsed: float) -> None:
    """Short fading corner traces. Kept in RAM for a few frames only."""
    if len(trail_history) < 2:
        return
    overlay = frame.copy()
    history = list(trail_history)
    for idx in range(4):
        for j in range(1, len(history)):
            opacity_scale = j / len(history)
            if opacity_scale < .30:
                continue
            p1 = tuple(np.rint(history[j-1][idx]).astype(int))
            p2 = tuple(np.rint(history[j][idx]).astype(int))
            cv2.line(overlay, p1, p2, CYAN, 1 if j < len(history)-2 else 2, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(frame, .72, overlay, .28, 0)


def run() -> int:
    if not MODEL.is_file():
        print(f"Missing hand model: {MODEL}\nRun download_model.py or copy the model from V3.")
        return 1
    try:
        import mediapipe as mp
    except ImportError as error:
        print(f"MediaPipe unavailable: {error}. Use your working V3 Python environment.")
        return 1

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not camera.isOpened():
        camera.release()
        print("Webcam unavailable. Check camera permissions, camera index and other apps.")
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=.56,
        min_hand_presence_confidence=.56,
        min_tracking_confidence=.56,
    )
    smoother = OneEuroQuadSmoother()
    gestures = PinchEffectGesture()
    trails = deque(maxlen=9)
    missed_frames = 0
    auto = False                  # Launch V4 hologram; no unexpected effect cycling.
    use_gestures = True
    perspective = True
    foreground = True
    scanlines = True
    grid = True
    bloom = True
    orbitals = True
    cinematic_grade = True
    eco = False
    debug = False
    help_overlay = True
    fullscreen = False
    chosen_effect = 6            # Hologram initially.
    start = time.monotonic()
    cycle_start = start - chosen_effect * AUTO_EFFECT_SECONDS
    previous_time = start
    previous_stamp = -1
    fps_smooth = 30.0
    vignette = None
    vignette_size = None
    window_created = False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, live = camera.read()
                if not ok or live is None:
                    print("Webcam stopped sending frames. Close other camera apps and retry.")
                    break
                live = cv2.flip(live, 1)
                height, width = live.shape[:2]
                t = time.monotonic()
                elapsed = t-start
                rgb = np.ascontiguousarray(cv2.cvtColor(live, cv2.COLOR_BGR2RGB))
                stamp = max(previous_stamp+1, int(elapsed*1000))
                previous_stamp = stamp
                prediction = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
                hands = prediction.hand_landmarks
                quad = hand_portal(hands, width, height)
                gesture = gestures.update(hands, width, height, t, enabled=use_gestures)

                effect = (int((t-cycle_start)/AUTO_EFFECT_SECONDS) % len(EFFECT_NAMES)
                          if auto else chosen_effect)
                if gesture.fired:
                    chosen_effect = (effect + 1) % len(EFFECT_NAMES)
                    effect = chosen_effect
                    auto = False
                if quad is not None:
                    missed_frames = 0
                    filtered_quad = smoother.update(quad, t)
                    if valid_quad(filtered_quad):
                        trails.append(filtered_quad.copy())
                    else:
                        smoother.reset()
                        trails.clear()
                        filtered_quad = quad
                else:
                    missed_frames += 1
                    if missed_frames > 2:
                        smoother.reset()
                        trails.clear()
                    filtered_quad = smoother.previous if missed_frames <= 2 else None
                tracking = quad is not None and filtered_quad is not None
                energy = portal_energy(filtered_quad, width) if tracking else .3

                if cinematic_grade:
                    if vignette_size != (height, width):
                        vignette = vignette_factor(height, width)
                        vignette_size = (height, width)
                    frame = apply_vignette(live, vignette)
                else:
                    frame = live.copy()
                original_background = frame.copy()
                if filtered_quad is not None and valid_quad(filtered_quad):
                    draw_trails(frame, trails, elapsed)
                    frame = draw_holographic_portal(
                        frame, filtered_quad, effect, apply_filter, elapsed,
                        perspective=perspective, scanlines=scanlines,
                        grid=grid, bloom=bloom, energy=energy,
                        orbitals=orbitals, eco=eco)
                    if foreground and tracking:
                        frame = hands_in_foreground(frame, original_background, hands)
                    draw_anchors(frame, filtered_quad, elapsed)
                if debug:
                    draw_hand_skeleton(frame, hands)

                fps = 1. / max(.001, t-previous_time)
                previous_time = t
                fps_smooth = fps_smooth*.88 + min(120, fps)*.12
                render_hud(frame, effect, fps_smooth, tracking, auto,
                           perspective, gesture.status, gesture.progress,
                           use_gestures, eco, orbitals, energy, debug,
                           help_overlay, elapsed)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')):
                    break
                if key == 32:     # Space
                    auto = not auto
                    chosen_effect = effect
                    cycle_start = time.monotonic() - effect*AUTO_EFFECT_SECONDS
                elif ord('1') <= key <= ord('9'):
                    chosen_effect = key - ord('1')
                    auto = False
                elif key == ord('0'):
                    chosen_effect = 9
                    auto = False
                elif key in (ord('m'), ord('M')):
                    use_gestures = not use_gestures
                    gestures.reset()
                elif key in (ord('p'), ord('P')):
                    perspective = not perspective
                elif key in (ord('h'), ord('H')):
                    foreground = not foreground
                elif key in (ord('s'), ord('S')):
                    scanlines = not scanlines
                elif key in (ord('g'), ord('G')):
                    grid = not grid
                elif key in (ord('b'), ord('B')):
                    bloom = not bloom
                elif key in (ord('e'), ord('E')):
                    orbitals = not orbitals
                elif key in (ord('l'), ord('L')):
                    eco = not eco
                elif key in (ord('c'), ord('C')):
                    cinematic_grade = not cinematic_grade
                elif key in (ord('d'), ord('D')):
                    debug = not debug
                elif key == 9:
                    help_overlay = not help_overlay
                elif key in (ord('f'), ord('F')):
                    try:
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    except cv2.error:
                        fullscreen = False
                        print("Fullscreen unsupported on this window backend.")
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except (RuntimeError, ValueError, cv2.error, OSError) as error:
        print(f"NV. AR Portal V4 stopped: {error}")
        return 1
    finally:
        camera.release()
        if window_created:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
