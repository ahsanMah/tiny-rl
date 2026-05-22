from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_random_video_dataset(
    *,
    num_videos: int,
    frames: int,
    height: int,
    width: int,
    channels: int,
    seed: int = 0,
) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((num_videos, frames, height, width, channels))


def load_video_frames(
    path: str | Path,
    *,
    target_fps: float = 8.0,
    spatial_downsample: int = 2,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, float | tuple[int, int] | int]]:
    path = Path(path)
    reader = iio.get_reader(path, format="ffmpeg")
    meta = reader.get_meta_data()

    if spatial_downsample <= 0:
        raise ValueError(f"spatial_downsample must be > 0, got {spatial_downsample}")

    source_fps = float(meta.get("fps", target_fps))
    frame_step = max(int(round(source_fps / target_fps)), 1)
    actual_fps = source_fps / frame_step

    frames: list[np.ndarray] = []
    for index, frame in enumerate(reader):
        if index % frame_step != 0:
            continue
        frame = np.asarray(frame)
        frame = frame[::spatial_downsample, ::spatial_downsample]
        height = frame.shape[0] - (frame.shape[0] % 4)
        width = frame.shape[1] - (frame.shape[1] % 4)
        if height < 4 or width < 4:
            raise ValueError(
                f"Downsampled frame is too small for UNet: {frame.shape[:2]}"
            )
        frame = frame[:height, :width]
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    reader.close()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    frame_array = np.stack(frames, axis=0)
    source_size = meta.get("size")
    if source_size is None:
        source_size = (int(frame_array.shape[2]), int(frame_array.shape[1]))

    info: dict[str, float | tuple[int, int] | int] = {
        "source_fps": source_fps,
        "actual_fps": actual_fps,
        "frame_step": frame_step,
        "source_size": tuple(source_size),
        "processed_size": (int(frame_array.shape[2]), int(frame_array.shape[1])),
        "spatial_downsample": spatial_downsample,
        "num_frames": int(frame_array.shape[0]),
    }
    return frame_array, info


def frames_to_clips(
    frames: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> mx.array:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, H, W, C), got {frames.shape}")
    if frames.shape[0] < clip_length:
        raise ValueError(
            f"Need at least {clip_length} frames, but only loaded {frames.shape[0]}"
        )

    clip_stride = 1 if clip_stride is None else clip_stride
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be > 0, got {clip_stride}")

    clips = []
    for start in range(0, frames.shape[0] - clip_length + 1, clip_stride):
        clips.append(frames[start : start + clip_length])
        if max_clips is not None and len(clips) >= max_clips:
            break

    if not clips:
        raise ValueError("No clips could be formed from the loaded video")

    return mx.array(np.stack(clips, axis=0))


def load_video_dataset(
    path: str | Path,
    *,
    clip_length: int,
    target_fps: float = 8.0,
    spatial_downsample: int = 2,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> tuple[mx.array, dict[str, float | tuple[int, int] | int]]:
    frames, info = load_video_frames(
        path,
        target_fps=target_fps,
        spatial_downsample=spatial_downsample,
    )
    clips = frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    info["num_clips"] = int(clips.shape[0])
    info["clip_length"] = int(clips.shape[1])
    return clips, info


def to_uint8_video(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)


def _load_overlay_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=10)
    except TypeError:
        return ImageFont.load_default()


def _annotate_action(frame: np.ndarray, action: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        return frame
    mode = "RGB" if frame.shape[-1] == 3 else "RGBA"
    img = Image.fromarray(frame, mode=mode)
    draw = ImageDraw.Draw(img)
    draw.text(
        (1, 1),
        f"action={int(action)}",
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
        font=_load_overlay_font(),
    )
    return np.asarray(img)


def save_clip_previews(
    clips: mx.array,
    output_dir: str | Path,
    *,
    max_clips: int = 4,
    fps: float = 8.0,
    actions: mx.array | np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips_np = np.asarray(clips)
    preview_count = min(max_clips, int(clips_np.shape[0]))
    clips_uint8 = to_uint8_video(clips_np[:preview_count])

    num_clips, num_frames, height, width, channels = clips_uint8.shape
    if channels not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count for preview: {channels}")

    if actions is not None and channels in (3, 4):
        actions_np = np.asarray(actions)[:preview_count]
        if actions_np.shape[:2] != (num_clips, num_frames):
            raise ValueError(
                f"actions shape {tuple(actions_np.shape)} does not match clips "
                f"({num_clips}, {num_frames})"
            )
        annotated = np.empty_like(clips_uint8)
        for clip_idx in range(num_clips):
            for frame_idx in range(num_frames):
                annotated[clip_idx, frame_idx] = _annotate_action(
                    clips_uint8[clip_idx, frame_idx],
                    int(actions_np[clip_idx, frame_idx]),
                )
        clips_uint8 = annotated

    sheet = np.zeros((num_clips * height, num_frames * width, channels), dtype=np.uint8)
    for clip_idx in range(num_clips):
        for frame_idx in range(num_frames):
            y0 = clip_idx * height
            x0 = frame_idx * width
            sheet[y0 : y0 + height, x0 : x0 + width] = clips_uint8[clip_idx, frame_idx]

    if channels == 1:
        sheet = sheet[..., 0]
    iio.imwrite(output_dir / "clips_sheet.png", sheet)

    frame_duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    for clip_idx in range(num_clips):
        gif_frames = clips_uint8[clip_idx]
        if channels == 1:
            gif_frames = gif_frames[..., 0]
        iio.mimsave(
            output_dir / f"clip_{clip_idx:03d}.gif",
            list(gif_frames),
            duration=frame_duration_ms,
            loop=0,
        )
