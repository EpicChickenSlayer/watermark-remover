#!/usr/bin/env python3
"""
Watermark Remover  -  Tkinter desktop app for images and videos
===============================================================

How it works (two AI models, both run locally on your Mac):

  1. DETECT   Microsoft Florence-2 ("open-vocabulary detection") is asked to
              find the "watermark" in the picture and returns bounding boxes.
  2. MASK     The boxes are padded slightly and drawn into a binary mask.
  3. INPAINT  LaMa ("Large Mask Inpainting") fills the masked pixels with
              content that matches the surroundings. Only a crop around each
              watermark is sent to LaMa, so the rest of the picture stays
              pixel-identical and large images/frames are handled quickly.

Video mode is built for STATIC watermarks (a logo/text that stays in one place):

  * Florence-2 looks at a handful of frames spread across the video; only a
    region that is found in most of them is kept (one-off false detections are
    discarded).
  * ONE mask is built from that and reused for every frame, so detection
    happens once, not on every frame.
  * Frames are decoded (OpenCV), inpainted (LaMa) and encoded (ffmpeg) in three
    overlapping threads. The original audio is kept.
  * Before the long job starts you get a before/after preview to confirm.

Setup (Python 3.10+):

    pip install torch "transformers>=4.56" opencv-python pillow numpy certifi
    brew install ffmpeg            # needed for video mode only

On the first run the app downloads the models:
  * Florence-2   -> Hugging Face cache (~/.cache/huggingface)
  * LaMa weights -> ~/.cache/watermark_remover/big-lama.pt

Run:  python watermark_remover.py

Code layout
-----------
  1. Configuration constants
  2. Errors + validation helpers
  3. Image / mask / geometry helpers (no AI, no GUI)
  4. AI models  (WatermarkDetector, LamaInpainter, ModelManager)
  5. Region jobs shared by image + video (plan_jobs / apply_job / preview)
  6. Image pipeline
  7. Video pipeline (sampling, consensus mask, ffmpeg streaming)
  8. Tkinter GUI (App) - the heavy work runs in a background thread and talks
     to the GUI through a queue, so the window never freezes.
"""
from __future__ import annotations

import os

# Must be set BEFORE torch is imported: any op that Apple's MPS backend does not
# support yet silently falls back to the CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import base64
import hashlib
import importlib.util
import logging
import math
import queue
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, UnidentifiedImageError
from tkinter import filedialog, messagebox, ttk

log = logging.getLogger("watermark_remover")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP_TITLE = "Watermark Remover"

# --- Detection (Florence-2) -------------------------------------------------
# "-base" is roughly 3x smaller/faster than "-large" but a little less accurate.
DETECTOR_MODEL_ID = "florence-community/Florence-2-large"
DETECTION_PROMPT = "watermark"   # try "logo" or "text watermark" for stubborn cases
MAX_BOX_FRACTION = 0.25          # ignore boxes covering > 25% of the image (usually false positives)
MIN_BOX_PX = 4                   # ignore boxes smaller than this (width or height)

# --- Mask ---------------------------------------------------------------------
MASK_PAD_RATIO = 0.004           # grow each box by this fraction of the longest image side ...
MASK_PAD_MIN_PX = 4              # ... but by at least this many pixels

# --- Inpainting (LaMa) -----------------------------------------------------------
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"   # integrity check of the download
VERIFY_CHECKSUM = True
CACHE_DIR = Path.home() / ".cache" / "watermark_remover"
CONTEXT_MARGIN_MIN_PX = 96       # context kept around each watermark when cropping for LaMa
MAX_INFER_SIDE = 1536            # crops larger than this are down-scaled for inference only
FEATHER_SIGMA = 1.5              # softens the seam when pasting the result back

# --- Video ---------------------------------------------------------------------
SAMPLE_FRAMES = 9                # frames examined (spread over the video) to find a static watermark
STATIC_MIN_FRACTION = 0.30       # a region must be found in at least this share of the sampled frames (3 of 9)
CLUSTER_OVERLAP = 0.30           # boxes from different frames overlapping this much = the same watermark
VIDEO_ENCODER = "auto"           # "auto" (hardware if available) | "videotoolbox" | "libx264"
BITS_PER_PIXEL = 0.15            # hardware-encoder bitrate = width * height * fps * this ...
MIN_BITRATE, MAX_BITRATE = 2_000_000, 100_000_000   # ... clamped to this range (bits/s)
X264_CRF = 18                    # software-encoder quality (lower = better/larger)
QUEUE_DEPTH = 8                  # frames buffered between the reader / inpainter / writer threads
COPYABLE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}   # copied as-is; anything else becomes AAC

# --- File types ---------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}      # accepted as input
OUTPUT_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}                         # written as H.264 + audio

Box = tuple[int, int, int, int]                      # x1, y1, x2, y2 in pixels
ProgressFn = Callable[[Optional[float], str], None]  # (fraction 0-1 or None = "busy", status text)
ConfirmFn = Callable[[np.ndarray], bool]             # (preview RGB image) -> user said "go ahead"


# =============================================================================
# 2. ERRORS + VALIDATION
# =============================================================================
class UserFacingError(Exception):
    """An error whose message is safe and useful to show directly to the user."""


class ProcessingCancelled(Exception):
    """Raised inside the worker when the user pressed Cancel (or closed the window)."""


def check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise ProcessingCancelled()


def missing_packages() -> list[str]:
    """Names of required pip packages that are not installed."""
    needed = {"torch": "torch", "transformers": "transformers"}
    return [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]


def find_tool(name: str) -> Optional[str]:
    """Locate an executable. Also checks Homebrew folders, which a Finder-launched app may not have on PATH."""
    found = shutil.which(name)
    if found:
        return found
    for folder in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
        cand = Path(folder) / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def require_ffmpeg() -> str:
    path = find_tool("ffmpeg")
    if not path:
        raise UserFacingError("Video mode needs ffmpeg, which was not found.\n\nInstall it with:\n    brew install ffmpeg")
    return path


def open_capture(path: str) -> "cv2.VideoCapture":
    """Open a video with OpenCV, honouring phone-rotation metadata."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:  # noqa: BLE001 - property missing in some OpenCV builds; default is already "on"
        pass
    return cap


def validate_image_file(path: str) -> None:
    """Raise UserFacingError unless `path` is a readable image of a supported type."""
    p = Path(path)
    if not p.is_file():
        raise UserFacingError(f"File not found:\n{path}")
    if p.suffix.lower() not in IMAGE_EXTS:
        raise UserFacingError(
            f"'{p.name}' is not a supported image type.\n"
            f"Supported: {', '.join(sorted(IMAGE_EXTS))}"
        )
    try:
        with Image.open(p) as im:
            im.verify()  # cheap integrity check, does not decode the pixels
    except Image.DecompressionBombError as exc:
        raise UserFacingError("This image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UserFacingError(f"'{p.name}' is not a valid image file (or it is corrupted).") from exc


def validate_video_file(path: str) -> None:
    """Raise UserFacingError unless OpenCV can open `path` and read a frame."""
    p = Path(path)
    if not p.is_file():
        raise UserFacingError(f"File not found:\n{path}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise UserFacingError(
            f"'{p.name}' is not a supported video type.\n"
            f"Supported: {', '.join(sorted(VIDEO_EXTS))}"
        )
    cap = open_capture(str(p))
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    if not ok:
        raise UserFacingError(f"'{p.name}' could not be opened as a video (it may be corrupted).")


def validate_output_path(src: str, dst: str, mode: str) -> None:
    """Raise UserFacingError if `dst` is not a sensible place to write the result."""
    d = Path(dst)
    allowed = IMAGE_EXTS if mode == "image" else OUTPUT_VIDEO_EXTS
    if d.suffix.lower() not in allowed:
        raise UserFacingError(f"Output must end with one of: {', '.join(sorted(allowed))}")
    if src and d.resolve() == Path(src).resolve():
        raise UserFacingError("The output file must be different from the input file.")
    if not d.parent.is_dir():
        raise UserFacingError(f"Folder does not exist:\n{d.parent}")
    if not os.access(d.parent, os.W_OK):
        raise UserFacingError(f"You do not have permission to write to:\n{d.parent}")


# =============================================================================
# 3. IMAGE / MASK / GEOMETRY HELPERS  (pure numpy / OpenCV / Pillow)
# =============================================================================
@dataclass
class LoadedImage:
    rgb: np.ndarray                 # H x W x 3, uint8
    alpha: Optional[np.ndarray]     # H x W, uint8, or None if the image has no transparency
    icc: Optional[bytes]            # embedded colour profile, preserved on save


def load_image(path: str) -> LoadedImage:
    """Read an image, apply its EXIF rotation and split off any alpha channel."""
    try:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            im = ImageOps.exif_transpose(im)
            has_alpha = "A" in im.getbands() or "transparency" in im.info
            im = im.convert("RGBA" if has_alpha else "RGB")
            arr = np.array(im)
    except Image.DecompressionBombError as exc:
        raise UserFacingError("This image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise UserFacingError(f"Could not read the image: {exc}") from exc
    if arr.shape[2] == 4:
        return LoadedImage(np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3]), icc)
    return LoadedImage(arr, None, icc)


def save_image(rgb: np.ndarray, alpha: Optional[np.ndarray], icc: Optional[bytes], dst: Path) -> None:
    """Write the result; the file format follows the extension of `dst`."""
    ext = dst.suffix.lower()
    if alpha is not None and ext in {".png", ".webp", ".tif", ".tiff"}:
        im = Image.fromarray(np.dstack([rgb, alpha]))      # RGBA
    else:
        im = Image.fromarray(rgb)                          # RGB (JPEG/BMP cannot hold alpha)
    kwargs: dict = {}
    if ext in {".jpg", ".jpeg"}:
        kwargs.update(quality=95, subsampling=0)
    elif ext == ".webp":
        kwargs.update(quality=95)
    if icc:
        kwargs["icc_profile"] = icc
    try:
        im.save(dst, **kwargs)
    except (OSError, ValueError) as exc:
        raise UserFacingError(f"Could not save the result:\n{exc}") from exc


def parse_boxes(answer, size: tuple[int, int]) -> list[Box]:
    """
    Turn Florence-2's post-processed answer into clean integer boxes.
    Accepts {"bboxes": [[x1,y1,x2,y2], ...]} and, defensively, {"quad_boxes": [[x1,y1,...,x4,y4], ...]}.
    """
    w, h = size
    raw: list[list[float]] = []
    if isinstance(answer, dict):
        raw += [list(b) for b in answer.get("bboxes", []) or []]
        for quad in answer.get("quad_boxes", []) or []:
            xs, ys = quad[0::2], quad[1::2]
            raw.append([min(xs), min(ys), max(xs), max(ys)])
    boxes: list[Box] = []
    for x1, y1, x2, y2 in raw:
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def filter_boxes(boxes: list[Box], w: int, h: int) -> tuple[list[Box], int]:
    """Drop implausible boxes. Returns (kept boxes, number rejected for being too large)."""
    kept: list[Box] = []
    too_big = 0
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        if bw * bh > MAX_BOX_FRACTION * w * h:
            too_big += 1
            continue
        kept.append((x1, y1, x2, y2))
    return kept, too_big


def build_mask(boxes: list[Box], w: int, h: int) -> np.ndarray:
    """Binary uint8 mask (1 = remove). Boxes are padded to also cover soft edges and shadows."""
    mask = np.zeros((h, w), np.uint8)
    pad = max(MASK_PAD_MIN_PX, round(MASK_PAD_RATIO * max(w, h)))
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 1
    return mask


def mask_regions(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding rectangles (x, y, w, h) of each connected blob in the mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)]


def _scaled(report: ProgressFn, lo: float, hi: float) -> ProgressFn:
    """Wrap `report` so a 0-1 sub-progress is mapped onto the range lo-hi of the overall bar."""
    def inner(frac: Optional[float], text: str) -> None:
        report(None if frac is None else lo + (hi - lo) * frac, text)
    return inner


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


# =============================================================================
# 4. AI MODELS
# =============================================================================
class WatermarkDetector:
    """Florence-2 open-vocabulary detector: image -> list of watermark boxes."""

    TASK = "<OPEN_VOCABULARY_DETECTION>"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise UserFacingError(
                "The detection model needs 'torch' and a recent 'transformers' (4.56 or newer).\n"
                "Run:  pip install -U torch transformers\n\n"
                f"({exc})"
            ) from exc

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info("Loading %s on %s", DETECTOR_MODEL_ID, self.device)
        try:
            self.processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
            self.model = (
                Florence2ForConditionalGeneration.from_pretrained(DETECTOR_MODEL_ID, dtype=torch.float32)
                .to(self.device)
                .eval()
            )
        except (OSError, ValueError) as exc:
            raise UserFacingError(
                "Could not load the detection model. Check your internet connection "
                f"(needed on the first run).\n\n{exc}"
            ) from exc

    def detect(self, image: Image.Image) -> list[Box]:
        """Return watermark boxes (pixel coordinates). Falls back to CPU if the GPU path fails."""
        try:
            return self._detect(image)
        except Exception as exc:  # noqa: BLE001 - any GPU-side failure -> retry on CPU
            if self.device == "cpu":
                raise
            log.warning("Detection failed on %s (%s); retrying on CPU", self.device, exc)
            self.model.to("cpu")
            self.device = "cpu"
            return self._detect(image)

    def _detect(self, image: Image.Image) -> list[Box]:
        inputs = self.processor(
            text=self.TASK + DETECTION_PROMPT, images=image, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=self.TASK, image_size=image.size)
        return parse_boxes(parsed.get(self.TASK, parsed), image.size)


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when available (avoids certificate errors on python.org builds of macOS Python)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def ensure_lama_weights(report: ProgressFn) -> Path:
    """Return the path of big-lama.pt, downloading and verifying it on first use."""
    path = CACHE_DIR / "big-lama.pt"
    if path.exists():
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    md5 = hashlib.md5()
    try:
        req = urllib.request.Request(LAMA_URL, headers={"User-Agent": "WatermarkRemover/1.0"})
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                md5.update(chunk)
                done += len(chunk)
                report(done / total if total else None, f"Downloading inpainting model… {done / 1e6:.0f} MB")
    except OSError as exc:  # includes URLError / timeouts
        tmp.unlink(missing_ok=True)
        raise UserFacingError(
            f"Could not download the inpainting model. Check your internet connection.\n\n{exc}"
        ) from exc
    if VERIFY_CHECKSUM and md5.hexdigest() != LAMA_MD5:
        tmp.unlink(missing_ok=True)
        raise UserFacingError(
            "The downloaded inpainting model failed its integrity check and was deleted.\n"
            "Please try again."
        )
    tmp.replace(path)
    return path


class LamaInpainter:
    """LaMa big-lama (TorchScript): (RGB image, mask) -> RGB image with the masked area filled in."""

    def __init__(self, weights: Path) -> None:
        try:
            import torch
        except ImportError as exc:
            raise UserFacingError("PyTorch is not installed.\nRun:  pip install torch") from exc
        self._torch = torch
        self._weights = weights
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        log.info("Loading LaMa on %s", self.device)
        try:
            self.model = self._load(self.device)
        except Exception as exc:  # noqa: BLE001 - corrupt file, MPS problem, ...
            if self.device.type == "cpu":
                raise UserFacingError(f"Could not load the inpainting model:\n{exc}") from exc
            log.warning("LaMa failed to load on MPS (%s); using CPU", exc)
            self.device = torch.device("cpu")
            self.model = self._load(self.device)

    def _load(self, device):
        # The LaMa checkpoint is a TorchScript file. PyTorch warns (FutureWarning) that TorchScript
        # loading is being phased out on Python 3.14+. It still works, so hide just that one message,
        # only around this one call; every other warning is left alone.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*torch\.jit\.load", category=FutureWarning)
            model = self._torch.jit.load(str(self._weights), map_location="cpu")
        return model.to(device).eval()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image: HxWx3 uint8 RGB, mask: HxW (non-zero = fill). Returns HxWx3 uint8. Falls back to CPU on GPU failure."""
        try:
            return self._run(image, mask)
        except Exception as exc:  # noqa: BLE001
            if self.device.type == "cpu":
                raise
            log.warning("Inpainting failed on MPS (%s); switching to CPU", exc)
            self.device = self._torch.device("cpu")
            self.model = self._load(self.device)
            return self._run(image, mask)

    def _run(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = image.shape[:2]
        pad_h, pad_w = (-h) % 8, (-w) % 8            # LaMa wants sizes divisible by 8
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        img_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255).unsqueeze(0)
        msk_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            out = self.model(img_t.to(self.device), msk_t.to(self.device))
        out = out[0].permute(1, 2, 0).float().cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError("inpainting produced invalid values")
        return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)[:h, :w]


class ModelManager:
    """
    Loads each model lazily on first use and keeps it in memory for later runs.
    Callers pass a `report` already scaled to the part of the progress bar they want to use.
    """

    def __init__(self) -> None:
        self._detector: Optional[WatermarkDetector] = None
        self._inpainter: Optional[LamaInpainter] = None

    def get_detector(self, report: ProgressFn) -> WatermarkDetector:
        if self._detector is None:
            report(None, "Loading detection model… (the first run downloads it and can take a while)")
            self._detector = WatermarkDetector()
        return self._detector

    def get_inpainter(self, report: ProgressFn) -> LamaInpainter:
        if self._inpainter is None:
            weights = ensure_lama_weights(report)
            report(None, "Loading inpainting model…")
            self._inpainter = LamaInpainter(weights)
        return self._inpainter


# =============================================================================
# 5. REGION JOBS  (shared by the image and video pipelines)
# =============================================================================
@dataclass
class RegionJob:
    """One crop of the picture to inpaint. Built once from the mask, reused for every video frame."""
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray    # uint8 (h, w) crop of the mask, 1 = fill
    blend: np.ndarray   # float32 (h, w, 1) feathered paste-back weights


def plan_jobs(mask: np.ndarray) -> list[RegionJob]:
    """
    Split the mask into crops (each watermark plus some surrounding context).
    Overlapping crops are handled in order: pixels filled by one crop are not filled again by the next.
    """
    h, w = mask.shape
    remaining = mask.copy()            # holes not filled yet
    jobs: list[RegionJob] = []
    for x, y, rw, rh in mask_regions(mask):
        if not remaining[y:y + rh, x:x + rw].any():
            continue                   # already covered by a neighbouring crop
        margin = max(CONTEXT_MARGIN_MIN_PX, rw, rh)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + rw + margin), min(h, y + rh + margin)
        crop_mask = remaining[y0:y1, x0:x1].copy()
        m = crop_mask.astype(np.float32)
        blend = np.maximum(cv2.GaussianBlur(m, (0, 0), FEATHER_SIGMA), m)[..., None]
        jobs.append(RegionJob(x0, y0, x1, y1, crop_mask, blend))
        remaining[y0:y1, x0:x1] = 0
    return jobs


def inpaint_crop(inpainter, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run LaMa on one crop, shrinking it first if it is very large (result is scaled back up)."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_INFER_SIDE / max(h, w))
    if scale >= 1.0:
        return inpainter.inpaint(img, mask)
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    k = int(np.ceil(1 / scale)) + 1                      # grow mask so no watermark sliver survives the resize
    grown = cv2.dilate(mask, np.ones((k, k), np.uint8))
    small = inpainter.inpaint(
        cv2.resize(img, size, interpolation=cv2.INTER_AREA),
        cv2.resize(grown, size, interpolation=cv2.INTER_NEAREST),
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_job(inpainter, frame: np.ndarray, job: RegionJob, bgr: bool = False) -> None:
    """Inpaint one crop of `frame` IN PLACE. Pixels outside the (feathered) mask are left untouched."""
    crop = frame[job.y0:job.y1, job.x0:job.x1]
    if bgr:                                              # OpenCV frames are BGR; LaMa wants RGB
        crop = crop[..., ::-1]
    crop = np.ascontiguousarray(crop)
    result = inpaint_crop(inpainter, crop, job.mask)
    blended = crop.astype(np.float32) * (1 - job.blend) + result.astype(np.float32) * job.blend
    blended = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    frame[job.y0:job.y1, job.x0:job.x1] = blended[..., ::-1] if bgr else blended


def _fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize to fit inside max_w x max_h (up to 4x enlargement), keeping the aspect ratio."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 4.0)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def make_preview(before: np.ndarray, after: np.ndarray, mask: np.ndarray, jobs: list[RegionJob]) -> np.ndarray:
    """
    RGB preview image: the whole frame with the detected area outlined in red (left),
    and a zoomed before / after of the first region (right).
    """
    marked = before.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (255, 0, 0), max(2, max(mask.shape) // 500))
    left = _fit(marked, 560, 420)
    j = jobs[0]
    crop_before = _fit(before[j.y0:j.y1, j.x0:j.x1], 400, 200)
    crop_after = _fit(after[j.y0:j.y1, j.x0:j.x1], 400, 200)

    pad, label_h = 12, 22
    right_w = max(crop_before.shape[1], crop_after.shape[1])
    height = max(left.shape[0] + label_h,
                 crop_before.shape[0] + crop_after.shape[0] + 2 * label_h + pad) + 2 * pad
    width = pad + left.shape[1] + pad + right_w + pad
    canvas = np.full((height, width, 3), 38, np.uint8)

    def put(img: np.ndarray, x: int, y: int, label: str) -> int:
        cv2.putText(canvas, label, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        canvas[y + label_h:y + label_h + img.shape[0], x:x + img.shape[1]] = img
        return y + label_h + img.shape[0]

    put(left, pad, pad, "Detected area (red)")
    x_right = 2 * pad + left.shape[1]
    y_next = put(crop_before, x_right, pad, "Before (zoomed)")
    put(crop_after, x_right, y_next + pad, "After (zoomed)")
    return canvas


# =============================================================================
# 6. IMAGE PIPELINE
# =============================================================================
@dataclass
class RemovalResult:
    regions_removed: int      # 0 -> nothing was written to disk
    rejected_too_large: int   # detections ignored because they covered too much of the image


def remove_watermark_from_image(src: str, dst: str, models, report: ProgressFn,
                                cancel: Optional[threading.Event] = None) -> RemovalResult:
    """
    Full pipeline: load -> detect -> mask -> inpaint each region -> save.
    `models` only needs get_detector(report) and get_inpainter(report), so it is easy to stub in tests.
    """
    report(0.02, "Reading image…")
    loaded = load_image(src)
    h, w = loaded.rgb.shape[:2]
    check_cancel(cancel)

    # --- detect -----------------------------------------------------------------
    detector = models.get_detector(_scaled(report, 0.05, 0.30))
    check_cancel(cancel)
    report(0.32, "Looking for watermarks…")
    boxes, too_big = filter_boxes(detector.detect(Image.fromarray(loaded.rgb)), w, h)
    if not boxes:
        return RemovalResult(0, too_big)
    check_cancel(cancel)
    report(0.48, f"Found {len(boxes)} watermark region(s)")

    # --- inpaint ------------------------------------------------------------------
    jobs = plan_jobs(build_mask(boxes, w, h))
    inpainter = models.get_inpainter(_scaled(report, 0.50, 0.64))
    out = loaded.rgb.copy()
    for i, job in enumerate(jobs):
        check_cancel(cancel)
        report(0.65 + 0.30 * i / len(jobs), f"Removing watermark {i + 1} of {len(jobs)}…")
        apply_job(inpainter, out, job)

    # --- save ---------------------------------------------------------------------
    check_cancel(cancel)
    report(0.96, "Saving…")
    save_image(out, loaded.alpha, loaded.icc, Path(dst))
    report(1.0, "Done")
    return RemovalResult(len(jobs), too_big)


# =============================================================================
# 7. VIDEO PIPELINE
# =============================================================================
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int     # 0 if the container does not say


@dataclass
class VideoResult:
    regions_removed: int          # 0 -> no consistent watermark found, nothing was written
    rejected_too_large: int = 0
    frames: int = 0
    seconds: float = 0.0
    note: str = ""


def probe_video(path: str) -> VideoInfo:
    """Size, frame rate and frame count (size is taken from a real decoded frame, so rotation is right)."""
    cap = open_capture(path)
    try:
        if not cap.isOpened():
            raise UserFacingError("The video could not be opened.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise UserFacingError("No frames could be read from the video.")
        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if not (1.0 <= fps <= 240.0):          # also catches NaN
        fps = 30.0
    return VideoInfo(w, h, fps, max(0, total))


def sample_indices(total: int, k: int) -> list[int]:
    """Frame numbers to examine: k spots spread evenly over the video (avoiding the very first/last frame)."""
    if total <= 0:
        return list(range(0, k * 15, 15))     # length unknown: just look near the start
    k = min(k, total)
    return sorted({min(total - 1, int((i + 0.5) * total / k)) for i in range(k)})


def read_sample_frames(path: str, total: int, k: int, cancel: Optional[threading.Event]) -> list[np.ndarray]:
    """Decode the sample frames (BGR). Seeks when the length is known, otherwise reads sequentially."""
    wanted = sample_indices(total, k)
    frames: list[np.ndarray] = []
    cap = open_capture(path)
    try:
        if total > 0:
            for idx in wanted:
                check_cancel(cancel)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
        else:
            want, i = set(wanted), 0
            while i <= max(wanted):
                ok, fr = cap.read()
                if not ok:
                    break
                if i in want:
                    frames.append(fr)
                i += 1
    finally:
        cap.release()
    if not frames:                              # seeking failed for this codec -> fall back to the first frame
        cap = open_capture(path)
        try:
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
        finally:
            cap.release()
    return frames


def _overlap(a: Box, b: Box) -> float:
    """How much two boxes agree: IoU, or 1-sided containment when one box sits inside the other."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter)
    contain = inter / min(area_a, area_b)
    return max(iou, contain if contain >= 0.8 else 0.0)


def _median_box(boxes: list[Box]) -> Box:
    arr = np.array(boxes)
    return tuple(int(v) for v in np.median(arr, axis=0))   # type: ignore[return-value]


def consensus_boxes(per_frame: list[list[Box]], n_frames: int) -> list[Box]:
    """
    Keep only watermark regions that appear in enough of the sampled frames (a static watermark is
    found again and again; a false detection usually shows up once). Each kept region is the union
    of its boxes across frames, so a slightly jittery detection still covers the whole watermark.
    """
    min_support = max(1, math.ceil(STATIC_MIN_FRACTION * n_frames))
    clusters: list[dict] = []                               # {"boxes": [...], "frames": {frame numbers}}
    for fi, boxes in enumerate(per_frame):
        for b in boxes:
            best, best_score = None, 0.0
            for c in clusters:
                score = _overlap(b, _median_box(c["boxes"]))
                if score >= CLUSTER_OVERLAP and score > best_score:
                    best, best_score = c, score
            if best is None:
                clusters.append({"boxes": [b], "frames": {fi}})
            else:
                best["boxes"].append(b)
                best["frames"].add(fi)
    kept: list[Box] = []
    for c in clusters:
        if len(c["frames"]) >= min_support:
            arr = np.array(c["boxes"])
            kept.append((int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 2].max()), int(arr[:, 3].max())))
    return kept


def pick_encoder(ffmpeg: str) -> str:
    """'videotoolbox' (Apple hardware H.264) when this ffmpeg has it, otherwise 'libx264'."""
    if VIDEO_ENCODER in ("videotoolbox", "libx264"):
        return VIDEO_ENCODER
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return "videotoolbox" if "h264_videotoolbox" in out else "libx264"


def probe_audio_codec(src: str) -> Optional[str]:
    """Codec name of the first audio stream, or None (no audio / ffprobe missing)."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_name", "-of", "default=nw=1:nk=1", src],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_ffmpeg_cmd(ffmpeg: str, src: str, out: str, w: int, h: int, fps: float,
                     encoder: str, audio_codec: Optional[str]) -> list[str]:
    """ffmpeg reads raw BGR frames from stdin, takes the audio from the original file, writes H.264 + audio."""
    bitrate = int(min(MAX_BITRATE, max(MIN_BITRATE, w * h * fps * BITS_PER_PIXEL)))
    if encoder == "videotoolbox":
        venc = ["-c:v", "h264_videotoolbox", "-b:v", str(bitrate)]
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(X264_CRF)]
    aenc = ["-c:a", "copy"] if audio_codec in COPYABLE_AUDIO else ["-c:a", "aac", "-b:a", "192k"]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.5f}", "-i", "pipe:0",
        "-i", src,
        "-map", "0:v:0", "-map", "1:a:0?",                  # '?' = fine if the source has no audio
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",             # H.264/yuv420p needs even dimensions
        *venc, "-pix_fmt", "yuv420p",
        *aenc,
        "-movflags", "+faststart",
        out,
    ]


def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
    """Put into a bounded queue, giving up (False) if `stop` is set while waiting."""
    while True:
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            if stop.is_set():
                return False


def _drain(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _stream_video(src: str, dst: str, info: VideoInfo, jobs: list[RegionJob], inpainter,
                  report: ProgressFn, cancel: threading.Event, ffmpeg: str) -> tuple[int, str]:
    """
    Decode -> inpaint -> encode, in three overlapping threads:

        reader thread  --q_in-->  this thread (LaMa)  --q_out-->  writer thread --> ffmpeg stdin

    Output goes to a temporary file that replaces `dst` only on success, so a cancelled or failed run
    never leaves a half-written video behind. Returns (frames written, note).
    """
    w, h, total = info.width, info.height, info.total_frames
    out_path = Path(dst)
    tmp_out = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    cmd = build_ffmpeg_cmd(ffmpeg, src, str(tmp_out), w, h, info.fps, pick_encoder(ffmpeg), probe_audio_codec(src))
    log.info("ffmpeg: %s", " ".join(cmd))

    err_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_file)
    except OSError as exc:
        err_file.close()
        raise UserFacingError(f"Could not start ffmpeg:\n{exc}") from exc

    cap = open_capture(src)
    stop = threading.Event()
    q_in: queue.Queue = queue.Queue(QUEUE_DEPTH)
    q_out: queue.Queue = queue.Queue(QUEUE_DEPTH)
    state: dict = {"read_error": None, "write_error": None}

    def ffmpeg_message() -> str:
        err_file.seek(0)
        tail = err_file.read().decode("utf-8", "replace").strip()[-700:]
        return "ffmpeg stopped unexpectedly." + (f"\n\n{tail}" if tail else "")

    def reader() -> None:
        try:
            while not stop.is_set():
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                if fr.shape[:2] != (h, w):
                    fr = cv2.resize(fr, (w, h))
                if not _put(q_in, fr, stop):
                    return
        except Exception as exc:  # noqa: BLE001
            state["read_error"] = exc
        _put(q_in, None, stop)                    # end-of-video marker

    def writer() -> None:
        broken = False
        while True:
            item = q_out.get()
            if item is None:
                return
            if broken:
                continue                          # keep draining so the main loop never blocks
            try:
                proc.stdin.write(item.tobytes())
            except (OSError, ValueError) as exc:  # BrokenPipeError: ffmpeg died
                state["write_error"] = exc
                broken = True

    reader_t = threading.Thread(target=reader, name="frame-reader", daemon=True)
    writer_t = threading.Thread(target=writer, name="frame-writer", daemon=True)
    reader_t.start()
    writer_t.start()

    done, t0, last_report, success, note = 0, time.monotonic(), 0.0, False, ""
    try:
        while True:
            if cancel.is_set():
                raise ProcessingCancelled()
            if state["write_error"] is not None or proc.poll() is not None:
                raise UserFacingError(ffmpeg_message())
            try:
                frame = q_in.get(timeout=0.5)
            except queue.Empty:
                if not reader_t.is_alive() and q_in.empty():
                    break
                continue
            if frame is None:
                break
            for job in jobs:
                apply_job(inpainter, frame, job, bgr=True)
            if not _put(q_out, frame, stop):
                break
            done += 1
            now = time.monotonic()
            if now - last_report >= 0.2:
                last_report = now
                rate = done / max(now - t0, 1e-6)
                if total > 0:
                    left = _fmt_duration(max(0, total - done) / rate)
                    report(0.28 + 0.71 * min(done / total, 1.0),
                           f"Frame {done:,} of {total:,}  ·  {rate:.1f} fps  ·  about {left} left")
                else:
                    report(None, f"Frame {done:,}  ·  {rate:.1f} fps")

        if state["read_error"] is not None:
            raise UserFacingError(f"Reading the video failed:\n{state['read_error']}")
        if done == 0:
            raise UserFacingError("No frames could be read from the video.")

        # finish: flush the writer, close ffmpeg's stdin so it can finalise the file
        report(0.995, "Finishing the video file…")
        q_out.put(None)
        writer_t.join()
        try:
            proc.stdin.close()
        except OSError as exc:
            state["write_error"] = state["write_error"] or exc
        rc = proc.wait()
        if rc != 0 or state["write_error"] is not None:
            raise UserFacingError(ffmpeg_message())
        os.replace(tmp_out, out_path)
        success = True
        if total > 0 and done < 0.98 * total:
            note = (f"The video stopped after {done:,} of about {total:,} frames "
                    "(the file may be damaged or its length mis-reported).")
        return done, note
    finally:
        stop.set()
        if not success:
            try:
                proc.kill()                       # unblocks the writer if it is stuck on a full pipe
            except OSError:
                pass
        _drain(q_in)
        reader_t.join(timeout=5)
        while writer_t.is_alive():                # make sure the writer gets its end marker
            try:
                q_out.put(None, timeout=0.2)
                break
            except queue.Full:
                pass
        writer_t.join(timeout=10)
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        cap.release()
        err_file.close()
        if not success:
            tmp_out.unlink(missing_ok=True)


def process_video(src: str, dst: str, models, report: ProgressFn,
                  cancel: Optional[threading.Event] = None,
                  confirm: Optional[ConfirmFn] = None) -> VideoResult:
    """
    Full video pipeline:
      sample frames -> detect on each -> keep the region found consistently (static watermark)
      -> one mask -> [optional before/after preview the user must accept]
      -> every frame: inpaint the crop, encode with ffmpeg (audio is carried over)
    """
    cancel = cancel or threading.Event()
    ffmpeg = require_ffmpeg()

    report(0.01, "Reading video…")
    info = probe_video(src)
    w, h = info.width, info.height
    frames = read_sample_frames(src, info.total_frames, SAMPLE_FRAMES, cancel)
    if not frames:
        raise UserFacingError("No frames could be read from the video.")

    # --- detect on the sampled frames, keep what is consistent ---------------------
    detector = models.get_detector(_scaled(report, 0.02, 0.08))
    per_frame: list[list[Box]] = []
    rejected = 0
    for i, fr in enumerate(frames):
        check_cancel(cancel)
        report(0.08 + 0.12 * i / len(frames), f"Looking for the watermark… (frame {i + 1} of {len(frames)})")
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        boxes, too_big = filter_boxes(detector.detect(Image.fromarray(rgb)), rgb.shape[1], rgb.shape[0])
        per_frame.append(boxes)
        rejected += too_big
    boxes, too_big = filter_boxes(consensus_boxes(per_frame, len(frames)), w, h)
    rejected += too_big
    if not boxes:
        return VideoResult(0, rejected)

    # --- one mask for the whole video ------------------------------------------------
    mask = build_mask(boxes, w, h)
    jobs = plan_jobs(mask)
    report(0.20, f"Found {len(jobs)} watermark region(s) that stay in place")
    inpainter = models.get_inpainter(_scaled(report, 0.20, 0.26))

    # --- let the user check the result on one frame before the long job -----------------
    if confirm is not None:
        check_cancel(cancel)
        report(0.27, "Preparing preview…")
        before = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB)
        after = before.copy()
        for job in jobs:
            apply_job(inpainter, after, job)
        if not confirm(make_preview(before, after, mask, jobs)):
            raise ProcessingCancelled()

    # --- every frame --------------------------------------------------------------------
    report(0.28, "Removing the watermark from every frame…")
    t0 = time.monotonic()
    done, note = _stream_video(src, dst, info, jobs, inpainter, report, cancel, ffmpeg)
    report(1.0, "Done")
    return VideoResult(len(jobs), rejected, done, time.monotonic() - t0, note)


# =============================================================================
# 8. TKINTER GUI
# =============================================================================
class App:
    """
    Main window. Threading model:
      * The Tk main thread owns every widget.
      * Processing runs in a daemon worker thread, which only ever puts small
        tuples into `self.msgs`; `_poll_queue` (called by Tk every 100 ms) reads
        them and updates the widgets.
      * Cancel sets `self.cancel_event`; the worker checks it between steps/frames.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.models = ModelManager()
        self.msgs: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self._busy_bar = False

        self.mode = tk.StringVar(value="image")
        self.in_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.status = tk.StringVar(value="Pick a file to get started.")
        self.preview_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # macOS: Cmd+Q (and "Quit" in the app menu) bypasses WM_DELETE_WINDOW; by default Tk then calls
        # the C exit() directly, which skips Python's shutdown cleanup and makes Python print
        # "resource_tracker: ... leaked semaphore objects". Send it down the same clean-close path.
        try:
            self.root.createcommand("tk::mac::Quit", self._on_close)
        except Exception:  # noqa: BLE001 - harmless if this Tk build does not allow it
            pass
        if sys.platform == "darwin":
            # Cmd+Q / "Quit" in the app menu: by default Tk calls exit() directly, which ends Python
            # without its normal shutdown (leaving a "leaked semaphore" notice, and ffmpeg's partial
            # file behind if a video is being written). Send it through the same clean shutdown as the
            # window's close button instead.
            try:
                self.root.createcommand("::tk::mac::Quit", self._on_close)
            except tk.TclError:
                pass
        self.root.after(100, self._poll_queue)
        self.root.after(300, self._warn_if_packages_missing)

    # ---- UI construction -------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.minsize(660, 340)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frm = ttk.Frame(self.root, padding=20)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        # Row 0: mode selector (+ video preview option)
        ttk.Label(frm, text="Mode:").grid(row=0, column=0, sticky="w", pady=(0, 10))
        mode_box = ttk.Frame(frm)
        mode_box.grid(row=0, column=1, columnspan=2, sticky="w", pady=(0, 10))
        self.rb_image = ttk.Radiobutton(mode_box, text="Image", value="image",
                                        variable=self.mode, command=self._on_mode_change)
        self.rb_video = ttk.Radiobutton(mode_box, text="Video", value="video",
                                        variable=self.mode, command=self._on_mode_change)
        self.chk_preview = ttk.Checkbutton(mode_box, text="Preview before processing a video",
                                           variable=self.preview_var)
        self.rb_image.pack(side="left", padx=(0, 16))
        self.rb_video.pack(side="left")
        self.chk_preview.pack(side="left", padx=(28, 0))

        # Row 1: input file
        ttk.Label(frm, text="Input file:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.in_path, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=8, pady=4)
        self.btn_browse = ttk.Button(frm, text="Browse", command=self._on_browse)
        self.btn_browse.grid(row=1, column=2)

        # Row 2: output file
        ttk.Label(frm, text="Output file:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_path, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=8, pady=4)
        self.btn_save = ttk.Button(frm, text="Save to", command=self._on_save_to)
        self.btn_save.grid(row=2, column=2)

        # Row 3: action buttons
        actions = ttk.Frame(frm)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(18, 14))
        actions.columnconfigure(0, weight=1)
        self.btn_run = ttk.Button(actions, text="Remove Watermark", command=self._on_remove)
        self.btn_run.grid(row=0, column=0, sticky="ew", ipady=6)
        self.btn_cancel = ttk.Button(actions, text="Cancel", command=self._on_cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=1, padx=(8, 0), ipady=6)

        # Rows 4-5: progress bar + status line
        self.bar = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.bar.grid(row=4, column=0, columnspan=3, sticky="ew")
        ttk.Label(frm, textvariable=self.status, wraplength=600, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Lock the inputs while a job runs; the Cancel button is the mirror image."""
        state = "normal" if enabled else "disabled"
        for w in (self.btn_browse, self.btn_save, self.btn_run, self.rb_image, self.rb_video, self.chk_preview):
            w.config(state=state)
        self.btn_cancel.config(state="disabled" if enabled else "normal")

    # ---- button handlers --------------------------------------------------------
    def _on_mode_change(self) -> None:
        """File types differ between modes, so clear any previous selection."""
        self.in_path.set("")
        self.out_path.set("")
        self._show_progress(0.0, "Pick a file to get started.")

    def _on_browse(self) -> None:
        mode = self.mode.get()
        exts = IMAGE_EXTS if mode == "image" else VIDEO_EXTS
        pattern = " ".join(f"*{e}" for e in sorted(exts))
        path = filedialog.askopenfilename(
            title=f"Select {mode}",
            filetypes=[(f"{mode.capitalize()} files", pattern), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            (validate_image_file if mode == "image" else validate_video_file)(path)
        except UserFacingError as exc:
            messagebox.showerror("Invalid file", str(exc))
            return
        self.in_path.set(path)
        self.out_path.set("")        # force a fresh choice so we never overwrite by accident
        self._show_progress(0.0, "File selected. Now choose where to save the result.")

    def _on_save_to(self) -> None:
        mode = self.mode.get()
        src = self.in_path.get()
        if not src:
            messagebox.showwarning("No input file", "Please choose an input file first.")
            return
        p = Path(src)
        if mode == "image":
            ext = p.suffix.lower() if p.suffix.lower() in IMAGE_EXTS else ".png"
            types = [("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("WebP", "*.webp"),
                     ("TIFF", "*.tif *.tiff"), ("BMP", "*.bmp")]
        else:
            ext = p.suffix.lower() if p.suffix.lower() in OUTPUT_VIDEO_EXTS else ".mp4"
            types = [("MP4 video", "*.mp4"), ("QuickTime", "*.mov"), ("M4V video", "*.m4v")]
        path = filedialog.asksaveasfilename(
            title="Save result as",
            initialdir=str(p.parent),
            initialfile=f"{p.stem}_no_watermark{ext}",
            defaultextension=ext,
            filetypes=types,
        )
        if not path:
            return
        try:
            validate_output_path(src, path, mode)
        except UserFacingError as exc:
            messagebox.showerror("Cannot save there", str(exc))
            return
        self.out_path.set(path)
        self._show_progress(0.0, "Ready. Click “Remove Watermark”.")

    def _on_remove(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        mode, src, dst = self.mode.get(), self.in_path.get(), self.out_path.get()

        if not src:
            messagebox.showwarning("No input file", "Please choose an input file with Browse.")
            return
        if not dst:
            messagebox.showwarning("No output file", "Please choose where to save the result with Save to.")
            return
        missing = missing_packages()
        if missing:
            messagebox.showerror("Missing packages", "Please install:\n\n    pip install -U " + " ".join(missing))
            return
        try:                                  # re-check: the file may have changed since it was picked
            if mode == "image":
                validate_image_file(src)
            else:
                validate_video_file(src)
                require_ffmpeg()
            validate_output_path(src, dst, mode)
        except UserFacingError as exc:
            messagebox.showerror("Problem with the files", str(exc))
            return

        want_preview = bool(self.preview_var.get())    # read here: Tk variables must be touched from the main thread only
        self.cancel_event.clear()
        self._set_controls_enabled(False)
        self._show_progress(0.0, "Starting…")
        self.worker = threading.Thread(target=self._worker, args=(mode, src, dst, want_preview), daemon=True)
        self.worker.start()

    def _on_cancel(self) -> None:
        self.cancel_event.set()
        self.btn_cancel.config(state="disabled")
        self.status.set("Cancelling…")

    def _on_close(self) -> None:
        """Window close button: stop a running job cleanly (ffmpeg + partial file) before quitting."""
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "Processing is still running. Stop it and quit?"):
                return
            self.cancel_event.set()
            self.worker.join(timeout=5)
        self.root.destroy()

    # ---- background work ----------------------------------------------------------
    def _worker(self, mode: str, src: str, dst: str, want_preview: bool) -> None:
        """Runs in the worker thread. Must NOT touch any Tk widget - only post to the queue."""
        cancel = self.cancel_event

        def report(frac: Optional[float], text: str) -> None:
            self.msgs.put(("progress", frac, text))

        def confirm(preview_rgb: np.ndarray) -> bool:
            """Ask the user (via the GUI thread) whether the preview looks right; blocks until answered."""
            ok, buf = cv2.imencode(".png", cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                return True
            answered, answer = threading.Event(), {"ok": False}
            self.msgs.put(("preview", base64.b64encode(buf.tobytes()).decode("ascii"), answer, answered))
            while not answered.wait(0.25):
                if cancel.is_set():
                    return False
            return bool(answer["ok"])

        try:
            if mode == "image":
                result = remove_watermark_from_image(src, dst, self.models, report, cancel)
                self.msgs.put(("done_image", result, dst))
            else:
                result = process_video(src, dst, self.models, report, cancel, confirm if want_preview else None)
                self.msgs.put(("done_video", result, dst))
        except ProcessingCancelled:
            self.msgs.put(("cancelled",))
        except UserFacingError as exc:
            self.msgs.put(("error", str(exc)))
        except MemoryError:
            self.msgs.put(("error", "Ran out of memory. Try a smaller file."))
        except Exception as exc:  # noqa: BLE001 - last-resort guard so the GUI never hangs
            log.exception("Unexpected failure")
            self.msgs.put(("error", f"Unexpected error: {exc}\n\n(Details were printed in the Terminal.)"))

    # ---- queue -> widgets (main thread) ---------------------------------------------
    def _poll_queue(self) -> None:
        try:
            while True:
                try:
                    msg = self.msgs.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(msg)
                except Exception:  # noqa: BLE001 - one bad update must not kill the polling loop
                    log.exception("GUI update failed for %r", msg[0])
        finally:
            self.root.after(100, self._poll_queue)

    def _handle(self, msg: tuple) -> None:
        kind, *payload = msg
        if kind == "progress":
            self._show_progress(*payload)
        elif kind == "preview":
            self._show_preview(*payload)
        elif kind == "done_image":
            self._on_done_image(*payload)
        elif kind == "done_video":
            self._on_done_video(*payload)
        elif kind == "cancelled":
            self._on_cancelled()
        elif kind == "error":
            self._on_error(payload[0])

    def _show_progress(self, frac: Optional[float], text: str) -> None:
        """frac=None -> indeterminate 'busy' animation; otherwise a 0-1 fraction."""
        self.status.set(text)
        if frac is None:
            if not self._busy_bar:
                self.bar.config(mode="indeterminate")
                self.bar.start(12)
                self._busy_bar = True
        else:
            if self._busy_bar:
                self.bar.stop()
                self.bar.config(mode="determinate")
                self._busy_bar = False
            self.bar["value"] = max(0.0, min(1.0, frac)) * 100

    def _show_preview(self, png_b64: str, answer: dict, answered: threading.Event) -> None:
        """Modal-ish dialog with the before/after preview. The worker thread waits on `answered`."""
        self._show_progress(0.0, "Waiting for your confirmation…")

        def finish(ok: bool) -> None:
            answer["ok"] = ok
            answered.set()
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        try:
            dlg = tk.Toplevel(self.root)
            dlg.title("Check the detected watermark")
            dlg.transient(self.root)
            photo = tk.PhotoImage(data=png_b64)
            label = ttk.Label(dlg, image=photo)
            label.image = photo                         # keep a reference or Tk drops the picture
            label.pack(padx=12, pady=(12, 6))
            ttk.Label(dlg, justify="center", wraplength=620,
                      text="Left: the area that will be removed from every frame (red outline). "
                           "Right: zoomed before / after on a sample frame.\n"
                           "If it looks right, process the whole video.").pack(padx=12, pady=(0, 6))
            buttons = ttk.Frame(dlg)
            buttons.pack(pady=(4, 12))
            ttk.Button(buttons, text="Looks good - process the video",
                       command=lambda: finish(True)).pack(side="left", padx=6)
            ttk.Button(buttons, text="Cancel", command=lambda: finish(False)).pack(side="left", padx=6)
            dlg.protocol("WM_DELETE_WINDOW", lambda: finish(False))
            dlg.grab_set()
        except Exception:  # noqa: BLE001 - preview could not be shown: ask in plain words instead
            log.exception("Could not show the preview")
            finish(messagebox.askyesno("Preview unavailable",
                                       "The preview could not be displayed.\n\nProcess the whole video anyway?"))

    def _on_done_image(self, result: RemovalResult, dst: str) -> None:
        self._set_controls_enabled(True)
        if result.regions_removed == 0:
            self._show_progress(0.0, "No watermark detected — nothing was saved.")
            hint = ""
            if result.rejected_too_large:
                hint = ("\n\nSomething was detected but it covered too much of the picture to be a "
                        "normal watermark, so it was ignored.")
            messagebox.showinfo("No watermark found", "The AI could not find a watermark in this image." + hint)
            return
        self._show_progress(1.0, f"Done — saved to {dst}")
        messagebox.showinfo("Finished", f"Watermark removed.\n\nSaved to:\n{dst}")

    def _on_done_video(self, result: VideoResult, dst: str) -> None:
        self._set_controls_enabled(True)
        if result.regions_removed == 0:
            self._show_progress(0.0, "No watermark detected — nothing was saved.")
            hint = ""
            if result.rejected_too_large:
                hint = ("\n\nSomething was detected but it covered too much of the frame to be a "
                        "normal watermark, so it was ignored.")
            messagebox.showinfo(
                "No watermark found",
                "The AI could not find a watermark that stays in the same place in this video.\n\n"
                "(Video mode is built for static watermarks; a moving one is not handled yet.)" + hint)
            return
        self._show_progress(1.0, f"Done — saved to {dst}")
        text = (f"Watermark removed from {result.frames:,} frames in {_fmt_duration(result.seconds)}."
                f"\n\nSaved to:\n{dst}")
        if result.note:
            text += f"\n\nNote: {result.note}"
        messagebox.showinfo("Finished", text)

    def _on_cancelled(self) -> None:
        self._set_controls_enabled(True)
        self._show_progress(0.0, "Cancelled — nothing was saved.")

    def _on_error(self, message: str) -> None:
        self._set_controls_enabled(True)
        self._show_progress(0.0, "Failed — see the message for details.")
        messagebox.showerror("Error", message)

    def _warn_if_packages_missing(self) -> None:
        missing = missing_packages()
        if missing:
            messagebox.showwarning(
                "Missing packages",
                "The AI features need extra packages. Install them with:\n\n"
                "    pip install -U " + " ".join(missing),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
