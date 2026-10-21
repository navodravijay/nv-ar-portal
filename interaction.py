"""NV. V4 interaction primitives, independent of webcam/MediaPipe.

All functions operate on in-memory landmarks and timestamps only. No logging of
biometric features, frame capture, networking, or persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class GestureFeedback:
    fired: bool = False
    progress: float = 0.0
    status: str = "SHOW TWO HANDS"


class PinchEffectGesture:
    """Hold a thumb/index pinch with ONE hand to advance the active effect.

    A release is required before another event, making it edge-triggered.
    Single-hand requirement prevents inadvertent changes while using portal.
    Uses palm-relative distances, not fixed on-screen pixels.
    """

    def __init__(self, hold_seconds: float = 0.72, pinch_ratio: float = 0.32,
                 release_ratio: float = 0.51, cooldown_seconds: float = 1.0):
        if hold_seconds <= 0 or not 0 < pinch_ratio < release_ratio:
            raise ValueError("Invalid gesture thresholds")
        self.hold_seconds = hold_seconds
        self.pinch_ratio = pinch_ratio
        self.release_ratio = release_ratio
        self.cooldown_seconds = cooldown_seconds
        self.started_at: float | None = None
        self.armed = True
        self.last_fire = -float("inf")
        self.is_down = False

    @staticmethod
    def relative_pinch(hand, width: int, height: int) -> float:
        def point(i):
            return np.array([float(hand[i].x) * width,
                             float(hand[i].y) * height], dtype=np.float64)
        palm_width = np.linalg.norm(point(5) - point(17))
        # Missing/collapsed landmarks must NEVER trigger a false pinch.
        if palm_width < 8:
            return float("inf")
        gap = np.linalg.norm(point(4) - point(8))
        return float(gap / palm_width)

    def reset(self) -> None:
        self.started_at = None
        self.is_down = False
        self.armed = True

    def update(self, hands, width: int, height: int, now: float,
               enabled: bool = True) -> GestureFeedback:
        if not enabled:
            self.reset()
            return GestureFeedback(status="GESTURES OFF")
        if len(hands) != 1:
            self.reset()
            return GestureFeedback(status="ONE-HAND PINCH TO CHANGE FX" if len(hands) == 0 else "TWO-HAND PORTAL")
        ratio = self.relative_pinch(hands[0], width, height)
        if self.is_down:
            pinched = ratio < self.release_ratio
        else:
            pinched = ratio < self.pinch_ratio
        self.is_down = pinched
        if not pinched:
            self.started_at = None
            self.armed = True
            return GestureFeedback(status="PINCH THUMB + INDEX / HOLD")
        if not self.armed:
            return GestureFeedback(status="RELEASE TO REARM")
        if self.started_at is None:
            self.started_at = now
        progress = float(np.clip((now - self.started_at) / self.hold_seconds, 0, 1))
        if progress >= 1.0 and now - self.last_fire >= self.cooldown_seconds:
            self.armed = False
            self.last_fire = now
            self.started_at = None
            return GestureFeedback(fired=True, progress=1.0, status="EFFECT ADVANCED")
        if progress >= 1.0:
            return GestureFeedback(progress=1.0, status="WAIT / RELEASE")
        return GestureFeedback(progress=progress, status="HOLD PINCH TO SWITCH")


class OneEuroQuadSmoother:
    """One Euro low-pass filter with velocity-adaptive response for quads."""

    def __init__(self, min_cutoff: float = 1.45, beta: float = 0.025,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.previous: np.ndarray | None = None
        self.derivative: np.ndarray | None = None
        self.last_time: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def reset(self):
        self.previous = None
        self.derivative = None
        self.last_time = None

    def update(self, quad: np.ndarray, now: float) -> np.ndarray:
        current = np.asarray(quad, dtype=np.float32)
        if current.shape != (4, 2) or not np.isfinite(current).all():
            raise ValueError("Need finite, four-corner quad")
        if self.previous is None or self.last_time is None:
            self.previous = current.copy()
            self.derivative = np.zeros_like(current)
            self.last_time = now
            return current.copy()
        dt = float(np.clip(now - self.last_time, .008, .1))
        derivative = (current - self.previous) / dt
        d_alpha = self._alpha(self.d_cutoff, dt)
        self.derivative = (1 - d_alpha) * self.derivative + d_alpha * derivative
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        alpha = self._alpha(cutoff, dt)
        output = self.previous + alpha * (current - self.previous)
        self.previous = output.astype(np.float32)
        self.last_time = now
        return self.previous.copy()


def portal_energy(quad: np.ndarray, frame_width: int) -> float:
    """Stable dimensionless charge, 0.3..1, based on distance between hands."""
    if frame_width < 1:
        return .3
    left = (quad[0] + quad[3]) * .5
    right = (quad[1] + quad[2]) * .5
    ratio = np.linalg.norm(right - left) / frame_width
    return float(np.clip(.3 + (ratio - .18) * 1.5, .30, 1.0))


def draw_hand_skeleton(frame: np.ndarray, hands) -> None:
    """Optional landmark debug overlay. Never shows or stores identity data."""
    h, w = frame.shape[:2]
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
              (0, 13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17))
    for hand in hands:
        pts = np.array([(int(np.clip(p.x, 0, 1) * (w - 1)),
                         int(np.clip(p.y, 0, 1) * (h - 1))) for p in hand])
        for chain in chains:
            cv2.polylines(frame, [pts[list(chain)]], False, (255, 224, 77), 1, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 141, 77), -1, cv2.LINE_AA)
