"""Camera capture and attention estimation (all local, CPU only).

A single thread owns the camera. It keeps the latest frame for anyone who needs
a snapshot, and runs the YuNet face detector (~5 ms on 320x240) at a fixed rate.

"Looking at the lamp" is approximated from the five YuNet landmarks: when the head
turns, the nose tip slides away from the midpoint between the eyes. The ratio
(nose offset / eye distance) is a cheap, robust head-yaw proxy that needs no
extra model. Engagement uses hysteresis so a blink, a nod or one missed detection
never toggles the character.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..config import ASSETS, Config

log = logging.getLogger("vision")
DETECT_SIZE = (320, 240)
INTER_EYE_M = 0.063


@dataclass
class Face:
    cx: float  # normalised image centre, 0..1 (left..right)
    cy: float  # 0..1 (top..bottom)
    size_px: float  # face box width at full resolution
    eye_px: float  # inter-eye distance at full resolution
    yaw_ratio: float  # nose offset / eye distance (0 = facing camera)
    pitch_ratio: float  # nose drop below the eye line / eye distance
    score: float
    box: tuple[int, int, int, int]  # full-resolution x, y, w, h
    attending: bool = False


@dataclass
class AttentionState:
    face: Face | None
    score: float  # smoothed 0..1
    engaged: bool
    t: float


class WaveDetector:
    """A wave is fast side-to-side motion beside or above the face, repeated.

    Motion pixels (frame differencing, no hand model) are taken from a band
    around the face, with the face itself, the neck and the chest masked out so
    talking, nodding and breathing do not count. Dense optical flow on those
    pixels gives which way they are moving; a wave is that horizontal direction
    flipping several times within a short window while the face stays put
    (someone walking past moves everything at once). A few ms at 160x120.
    """

    W, H = 160, 120
    MIN_STEP = 0.08  # face widths per frame that counts as a deliberate swing

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev: np.ndarray | None = None
        self.hist: list[tuple[float, float | None, float]] = []  # (t, horizontal step in face widths, face x)
        self.last_wave_t = -1e9

    def update(self, frame: np.ndarray, face: Face | None, now: float) -> bool:
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self.prev = self.prev, g
        if prev is None or face is None:
            self.hist.clear()
            return False
        fh_full, fw_full = frame.shape[:2]
        x, y, w, h = face.box
        fx, fy, fw, fh = x * self.W / fw_full, y * self.H / fh_full, w * self.W / fw_full, h * self.H / fh_full
        mask = (cv2.absdiff(g, prev) > 18).astype(np.uint8)
        mask[: max(0, int(fy - 1.6 * fh))] = 0
        mask[min(self.H, int(fy + 1.8 * fh)):] = 0
        mask[max(0, int(fy - 0.3 * fh)):, max(0, int(fx - 0.35 * fw)):int(fx + 1.35 * fw)] = 0  # face, neck, chest
        step = None
        if int(mask.sum()) > 0.08 * fw * fh:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 9, 2, 5, 1.1, 0)
            dx = float(np.median(flow[..., 0][mask > 0])) / max(fw, 1.0)
            if abs(dx) >= self.MIN_STEP:
                step = dx
        self.hist.append((now, step, (fx + fw / 2) / max(fw, 1.0)))
        self.hist = [s for s in self.hist if now - s[0] <= self.cfg.wave_window_s]
        if now - self.last_wave_t < self.cfg.wave_cooldown_s or not self._is_wave():
            return False
        self.last_wave_t = now
        self.hist.clear()
        return True

    def _is_wave(self) -> bool:
        steps = [s for _, s, _ in self.hist if s is not None]
        if len(steps) < 5 or len(steps) < 0.4 * len(self.hist):
            return False
        faces = [f for _, _, f in self.hist]
        if max(faces) - min(faces) > 0.5:  # the whole person moved
            return False
        reversals = sum(1 for a, b in zip(steps, steps[1:]) if (a > 0) != (b > 0))
        return reversals >= self.cfg.wave_reversals


class Vision:
    def __init__(self, cfg: Config, on_change: Callable[[str, AttentionState], None]):
        self.cfg = cfg
        self.on_change = on_change
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_t = 0.0
        self.state = AttentionState(None, 0.0, False, 0.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._attend_since: float | None = None
        self._away_since: float | None = None
        self._last_attend = 0.0
        self.fps = 0.0
        self.detect_ms = 0.0
        self.wave = WaveDetector(cfg)

    # ------------------------------------------------------------ public
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._frame_t

    def jpeg(self, width: int, quality: int = 80, annotate: bool = False) -> bytes | None:
        frame, _ = self.latest()
        if frame is None:
            return None
        if annotate and self.state.face is not None:
            f = self.state.face
            x, y, w, h = f.box
            color = (120, 220, 120) if f.attending else (80, 200, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def focal_px(self) -> float:
        return (self.cfg.camera_width / 2) / math.tan(math.radians(self.cfg.camera_hfov_deg) / 2)

    def face_depth_m(self, face: Face) -> float:
        if face.eye_px <= 1:
            return self.cfg.face_assumed_depth_m
        return float(np.clip(self.focal_px() * INTER_EYE_M / face.eye_px, 0.3, 2.5))

    # ------------------------------------------------------------ thread
    def _open(self) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else (cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.cfg.camera_index, backend)
        if sys.platform.startswith("linux"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        model = ASSETS / "face_detection_yunet_2023mar.onnx"
        try:  # silence the OpenCV 5 DNN backend notice
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        detector = cv2.FaceDetectorYN.create(str(model), "", DETECT_SIZE, 0.7, 0.3, 20)
        cap = self._open()
        if not cap.isOpened():
            log.error("camera %s could not be opened; vision disabled", self.cfg.camera_index)
            return
        period = 1.0 / self.cfg.attention_hz
        next_detect = 0.0
        frames, fps_t = 0, time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                log.warning("camera read failed; reopening")
                cap.release()
                time.sleep(1.0)
                cap = self._open()
                continue
            with self._lock:
                self._frame, self._frame_t = frame, now
            frames += 1
            if now - fps_t >= 2.0:
                self.fps, frames, fps_t = frames / (now - fps_t), 0, now
            if now >= next_detect:
                next_detect = now + period
                t0 = time.perf_counter()
                small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_AREA)
                _, dets = detector.detect(small)
                self.detect_ms = 0.9 * self.detect_ms + 0.1 * (time.perf_counter() - t0) * 1000
                self._update(self._best_face(dets, frame.shape[1], frame.shape[0]), now)
            # Waves are fast; every camera frame (not just detector frames) keeps the hand trackable.
            if self.wave.update(frame, self.state.face, now):
                self.on_change("wave", self.state)
        cap.release()

    def _best_face(self, dets, fw: int, fh: int) -> Face | None:
        if dets is None or len(dets) == 0:
            return None
        d = max(dets, key=lambda r: r[2] * r[3])
        sx, sy = fw / DETECT_SIZE[0], fh / DETECT_SIZE[1]
        d = [float(v) for v in d]
        x, y, w, h = d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy
        if w < self.cfg.min_face_px:
            return None
        rex, rey, lex, ley, nx, ny = (d[4] * sx, d[5] * sy, d[6] * sx, d[7] * sy, d[8] * sx, d[9] * sy)
        eye_dist = math.hypot(lex - rex, ley - rey)
        if eye_dist < 1:
            return None
        mid_x, mid_y = (rex + lex) / 2, (rey + ley) / 2
        yaw = (nx - mid_x) / eye_dist
        pitch = (ny - mid_y) / eye_dist
        face = Face(cx=(x + w / 2) / fw, cy=(y + h / 2) / fh, size_px=w, eye_px=eye_dist,
                    yaw_ratio=yaw, pitch_ratio=pitch, score=float(d[14]),
                    box=(int(x), int(y), int(w), int(h)))
        # Facing the camera: nose roughly centred, and not looking far down (phone) or up.
        face.attending = bool(abs(yaw) < self.cfg.max_yaw_ratio and 0.15 < pitch < 1.0)
        return face

    def _update(self, face: Face | None, now: float) -> None:
        attending = bool(face and face.attending)
        target = 1.0 if attending else 0.0
        score = self.state.score + (target - self.state.score) * 0.25
        engaged = self.state.engaged
        if attending:
            self._last_attend = now
            self._away_since = None
            if self._attend_since is None:
                self._attend_since = now
        else:
            # Ignore single-frame dropouts before resetting the attend timer.
            if self._attend_since is not None and now - self._last_attend > 0.25:
                self._attend_since = None
            if self._away_since is None:
                self._away_since = now
        if not engaged and self._attend_since is not None and now - self._attend_since >= self.cfg.engage_after_s:
            engaged = True
        elif engaged and self._away_since is not None and now - self._away_since >= self.cfg.disengage_after_s:
            engaged = False
        prev = self.state
        self.state = AttentionState(face, score, engaged, now)
        if engaged != prev.engaged:
            self.on_change("engaged" if engaged else "disengaged", self.state)
        elif (face is None) != (prev.face is None):
            self.on_change("face_lost" if face is None else "face_seen", self.state)
