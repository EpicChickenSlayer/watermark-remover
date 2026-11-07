# Watermark Remover

A small desktop app (Python + Tkinter) that removes **visible watermarks from images and videos** using two AI models that run locally on your computer. Nothing is uploaded anywhere.

Built and tested on an Apple Silicon Mac (it uses the GPU through PyTorch's MPS backend and falls back to the CPU automatically).

> Only use this on pictures and videos that you own or have permission to edit.

## How it works

1. **Detect** – Microsoft's [Florence-2](https://huggingface.co/florence-community) is asked to find the "watermark" in the picture and returns bounding boxes.
2. **Mask** – the boxes are padded slightly and drawn into a mask.
3. **Inpaint** – [LaMa](https://github.com/advimman/lama) (Large Mask Inpainting) fills the masked pixels with content that matches the surroundings. Only a crop around each watermark is sent to LaMa, so the rest of the picture stays pixel-identical and large images are handled quickly.

**Video mode** is built for *static* watermarks (a logo or text that stays in one place):

- Florence-2 looks at 9 frames spread across the video. A region has to be found in at least 3 of them to count, so one-off false detections are discarded.
- One mask is built from that and reused for every frame, so detection happens once, not on every frame.
- Frames are decoded (OpenCV), inpainted (LaMa) and encoded (ffmpeg) in three overlapping threads. The original audio is kept.
- Before the long job starts you get a before/after preview to confirm.
- The result is written to a temporary file and moved into place only when it finishes, so a cancelled or failed run never leaves a broken file or overwrites an existing one.

## Features

- Image mode and Video mode, selected in the window
- **Browse** for the input, **Save to** for the output, **Remove Watermark** to start
- Progress bar and status line (video shows frame count, fps and time left)
- **Cancel** button; closing the window or pressing Cmd+Q stops a running job cleanly
- Error handling for wrong or corrupt files, missing packages, missing ffmpeg and unwritable folders
- Processing runs in background threads, so the window never freezes

Supported files:

| | Formats |
|---|---|
| Image in/out | `.jpg` `.jpeg` `.png` `.webp` `.bmp` `.tif` `.tiff` (transparency and colour profiles are preserved where the format allows) |
| Video in | `.mp4` `.mov` `.m4v` `.avi` `.mkv` `.webm` |
| Video out | `.mp4` `.mov` `.m4v` (H.264 + the original audio) |

## Requirements

- Python 3.10 or newer, with Tkinter (on Homebrew Python: `brew install python-tk`)
- [ffmpeg](https://ffmpeg.org) – **video mode only**: `brew install ffmpeg`
- The Python packages in `requirements.txt`

## Install and run

```bash
git clone git@github.com:EpicChickenSlayer/watermark-remover.git
cd watermark-remover

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python watermark_remover.py
```

The first run downloads the models: Florence-2-large (a couple of GB, into the Hugging Face cache) and the ~200 MB LaMa weights (into `~/.cache/watermark_remover/`, checksum-verified). After that everything works offline.

## Using it

1. Choose **Image** or **Video**.
2. Click **Browse** and pick a file.
3. Click **Save to** and choose where the result goes.
4. Click **Remove Watermark**.

In video mode a preview window shows the detected area (red outline) and a zoomed before/after on a sample frame. Click *Looks good* to process the whole video, or *Cancel*. The "Preview before processing a video" checkbox turns this off.

## Tuning

All settings are constants near the top of `watermark_remover.py`:

| Constant | What it does |
|---|---|
| `DETECTION_PROMPT` | What Florence-2 looks for. Try `"logo"` or `"text watermark"` if nothing is found. |
| `DETECTOR_MODEL_ID` | `florence-community/Florence-2-large` (default, more accurate) or `florence-community/Florence-2-base` (about 3x smaller and faster). |
| `MAX_BOX_FRACTION` | Detections covering more than this share of the picture are ignored as false positives. |
| `MASK_PAD_RATIO`, `MASK_PAD_MIN_PX` | How far the mask extends beyond the detected box (helps with soft edges and shadows). |
| `SAMPLE_FRAMES`, `STATIC_MIN_FRACTION` | How many frames are checked in video mode, and how many must agree. |
| `VIDEO_ENCODER` | `"auto"` (Apple hardware H.264 if your ffmpeg has it, otherwise libx264), `"videotoolbox"` or `"libx264"`. |

## Limitations

- **Static watermarks only in video mode.** A watermark that moves or changes position is not handled. If none is found consistently, nothing is saved and you are told so.
- Each frame is filled separately, so the patch can shimmer slightly over a fast-moving background.
- Variable-frame-rate videos are written at a constant frame rate.
- Video processing takes minutes, not seconds. The status line shows the time left.
- Watermarks that cover a large part of the picture (more than `MAX_BOX_FRACTION`) are ignored by design.

## Troubleshooting

- **"No watermark found"** – change `DETECTION_PROMPT`, or the watermark may be too faint or too large.
- **"Video mode needs ffmpeg"** – `brew install ffmpeg`.
- **A `torch.jit.load` FutureWarning on Python 3.14+** – the LaMa weights are a TorchScript file, which PyTorch is phasing out. It still works and the warning is hidden in the app.

## Credits

- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) by Microsoft (MIT licence) for detection.
- [LaMa](https://github.com/advimman/lama) by Samsung Research (Apache-2.0 licence) for inpainting. The `big-lama` weights are the TorchScript build distributed by the lama-cleaner / IOPaint project.
- The two-model idea (Florence-2 to find, LaMa to fill) is the same one used by [D-Ogi/WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI). This is an independent implementation.
