"""Offline unit tests: synthetic pixels/landmarks only. No webcam or network."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import (apply_filter, approximate_hand_mask, draw_trails, hand_portal,
                  hands_in_foreground, valid_quad)
from cinematic import (BLUE, CYAN, NAVY, ECO_FILTER_SIZE, EFFECT_NAMES,
                       apply_vignette, draw_anchors, draw_holographic_portal,
                       effect_filter, render_hud, vignette_factor)
from interaction import (OneEuroQuadSmoother, PinchEffectGesture,
                         draw_hand_skeleton, portal_energy)


def synthetic_hand(x=.25, top=.25, bottom=.76, pinched=False):
    palm = [SimpleNamespace(x=x+((i % 4)-1.5)*.019,
                            y=(top+bottom)/2+((i//4)-2)*.015)
            for i in range(21)]
    palm[0] = SimpleNamespace(x=x, y=(top+bottom)/2)
    palm[4] = SimpleNamespace(x=x, y=top)
    palm[8] = SimpleNamespace(x=x, y=bottom)
    palm[5] = SimpleNamespace(x=x-.050, y=.50)
    palm[17] = SimpleNamespace(x=x+.050, y=.50)
    if pinched:
        palm[4] = SimpleNamespace(x=x+.005, y=.54)
        palm[8] = SimpleNamespace(x=x+.012, y=.54)
    return palm


class V4Tests(unittest.TestCase):
    def setUp(self):
        h, w = 360, 640
        x = np.arange(w, dtype=np.uint16)[None, :]
        y = np.arange(h, dtype=np.uint16)[:, None]
        self.frame = np.dstack((np.broadcast_to((x % 256).astype('uint8'), (h,w)),
                                np.broadcast_to((y % 256).astype('uint8'), (h,w)),
                                np.broadcast_to(((x+y)//2 % 256).astype('uint8'), (h,w)))).copy()
        self.quad = np.array([[140, 80], [475, 68], [465, 291], [157, 261]], dtype=np.float32)

    def test_ten_effects_and_input_unchanged(self):
        self.assertEqual(len(EFFECT_NAMES), 10)
        before = self.frame.copy()
        for i in range(10):
            with self.subTest(i=i):
                out = effect_filter(self.frame, i, apply_filter)
                self.assertEqual(out.shape, self.frame.shape)
                self.assertEqual(out.dtype, np.uint8)
                self.assertTrue(np.any(out != self.frame))
        np.testing.assert_array_equal(self.frame, before)
        with self.assertRaises(ValueError):
            effect_filter(self.frame, 10, apply_filter)

    def test_valid_portal_from_landmarks(self):
        left = synthetic_hand(.23, .22, .75)
        right = synthetic_hand(.78, .21, .77)
        quad = hand_portal([right,left], 640, 360)
        self.assertTrue(valid_quad(quad))
        self.assertIsNone(hand_portal([left], 640, 360))
        self.assertIsNone(hand_portal([], 640, 360))

    def test_reject_degenerate_portal(self):
        self.assertFalse(valid_quad(None))
        self.assertFalse(valid_quad(np.ones((4,2), dtype=np.float32)))
        self.assertFalse(valid_quad(np.array([[np.nan, 0]]*4)))

    def test_perspective_eco_and_energy_effects(self):
        for warp in (False, True):
            for eco in (False, True):
                for i in range(10):
                    with self.subTest(i=i, warp=warp, eco=eco):
                        out = draw_holographic_portal(
                            self.frame, self.quad, i, apply_filter,
                            elapsed=1.37, perspective=warp, eco=eco,
                            orbitals=True, energy=.8)
                        self.assertEqual(out.shape, self.frame.shape)
                        self.assertEqual(out.dtype, np.uint8)
                        self.assertTrue(np.any(out[155:240,200:420] != self.frame[155:240,200:420]))
                        np.testing.assert_array_equal(out[0,0], self.frame[0,0])

    def test_optional_render_modes(self):
        out = draw_holographic_portal(self.frame, self.quad, 6, apply_filter,
                                      grid=False, scanlines=False, bloom=False,
                                      orbitals=False)
        self.assertEqual(out.shape, self.frame.shape)
        np.testing.assert_array_equal(draw_holographic_portal(self.frame, None, 6, apply_filter), self.frame)
        self.assertEqual(ECO_FILTER_SIZE, (320, 180))

    def test_one_euro_smoothing_and_reset(self):
        smoother = OneEuroQuadSmoother()
        base = smoother.update(self.quad, 1.0)
        np.testing.assert_array_equal(base, self.quad)
        advanced = smoother.update(self.quad + 36, 1.04)
        self.assertTrue(np.all(advanced > base))
        self.assertTrue(np.all(advanced < self.quad + 36))
        smoother.reset()
        np.testing.assert_array_equal(smoother.update(self.quad + 10, 2), self.quad+10)
        with self.assertRaises(ValueError):
            smoother.update(np.ones((4,2)) * np.nan, 3)

    def test_pinch_hold_fires_only_once_until_release(self):
        gesture = PinchEffectGesture()
        hand = synthetic_hand(pinched=True)
        self.assertFalse(gesture.update([hand], 640, 360, 10.).fired)
        self.assertGreater(gesture.update([hand], 640, 360, 10.5).progress, .5)
        self.assertTrue(gesture.update([hand], 640, 360, 10.75).fired)
        self.assertFalse(gesture.update([hand], 640, 360, 11.2).fired)
        self.assertFalse(gesture.update([synthetic_hand()], 640, 360, 12.).fired)
        self.assertFalse(gesture.update([hand], 640, 360, 12.1).fired)
        self.assertTrue(gesture.update([hand], 640, 360, 12.9).fired)

    def test_no_gesture_with_two_hands_or_disabled(self):
        gesture = PinchEffectGesture()
        p = synthetic_hand(pinched=True)
        gesture.update([p], 640, 360, 1)
        self.assertFalse(gesture.update([p,p], 640, 360, 2).fired)
        self.assertFalse(gesture.update([p], 640, 360, 3, enabled=False).fired)
        self.assertFalse(gesture.update([], 640, 360, 4).fired)

    def test_pinch_scale_independent(self):
        h = synthetic_hand(pinched=True)
        self.assertAlmostEqual(PinchEffectGesture.relative_pinch(h, 640, 360),
                               PinchEffectGesture.relative_pinch(h, 1280, 720), places=4)
        collapsed = [SimpleNamespace(x=.5, y=.5) for _ in range(21)]
        self.assertEqual(PinchEffectGesture.relative_pinch(collapsed, 640, 360), float('inf'))

    def test_energy_and_landmark_overlay(self):
        e = portal_energy(self.quad, 640)
        self.assertGreaterEqual(e, .3)
        self.assertLessEqual(e, 1.0)
        with_overlay = self.frame.copy()
        draw_hand_skeleton(with_overlay, [synthetic_hand()])
        self.assertTrue(np.any(with_overlay != self.frame))

    def test_hand_occlusion_mask(self):
        hand = synthetic_hand()
        mask = approximate_hand_mask([hand], 640, 360)
        self.assertTrue(np.count_nonzero(mask) > 100)
        over = draw_holographic_portal(self.frame, self.quad, 6, apply_filter)
        replaced = hands_in_foreground(over, self.frame, [hand])
        self.assertEqual(replaced.dtype, np.uint8)

    def test_hud_trails_anchor_and_vignette(self):
        vignette = vignette_factor(360, 640)
        out = apply_vignette(self.frame, vignette)
        trail = deque([self.quad-10, self.quad-5, self.quad])
        draw_trails(out, trail, 2.)
        draw_anchors(out, self.quad, 2.)
        render_hud(out, 6, 28.0, True, False, True,
                   'TWO-HAND PORTAL', 0.4, True, False, True, .75,
                   False, True, 2.)
        self.assertTrue(np.any(out != self.frame))
        self.assertEqual((NAVY, BLUE, CYAN), ((27, 13, 8), (255, 141, 77), (255, 224, 77)))


if __name__ == '__main__':
    unittest.main()


class OfflineIntegrationTests(unittest.TestCase):
    """One simulated end-to-end frame: no actual webcam or MediaPipe installed."""

    def test_run_one_mocked_frame_and_release_camera(self):
        from unittest.mock import patch, MagicMock
        import sys
        from types import SimpleNamespace
        import main

        class FakeCamera:
            def __init__(self):
                self.released = False
            def isOpened(self): return True
            def set(self, *args): return True
            def read(self): return True, np.zeros((360, 640, 3), np.uint8)
            def release(self): self.released = True

        class FakeDetector:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def detect_for_video(self, img, timestamp):
                return SimpleNamespace(hand_landmarks=[synthetic_hand(.23,.2,.76),
                                                       synthetic_hand(.77,.2,.76)])
        detector = FakeDetector()
        fake_mp = SimpleNamespace(
            Image=lambda **kwargs: object(), ImageFormat=SimpleNamespace(SRGB='srgb'),
            tasks=SimpleNamespace(BaseOptions=lambda **kwargs: object(),
                vision=SimpleNamespace(
                    HandLandmarkerOptions=lambda **kwargs: object(),
                    RunningMode=SimpleNamespace(VIDEO='video'),
                    HandLandmarker=SimpleNamespace(create_from_options=lambda _: detector))))
        camera = FakeCamera()
        with patch.object(main, 'MODEL', Path(main.__file__)), \
             patch.dict(sys.modules, {'mediapipe': fake_mp}), \
             patch.object(main.cv2, 'VideoCapture', return_value=camera), \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'imshow') as imshow, \
             patch.object(main.cv2, 'waitKey', return_value=ord('q')), \
             patch.object(main.cv2, 'destroyAllWindows'):
            self.assertEqual(main.run(), 0)
        self.assertTrue(camera.released)
        self.assertTrue(imshow.called)
