from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
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
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_videos, frames, height, width, channels)).astype(np.float32)


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
) -> np.ndarray:
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

    if not clips:
        raise ValueError("No clips could be formed from the loaded video")

    return np.stack(clips, axis=0)


def load_video_dataset(
    path: str | Path,
    *,
    clip_length: int,
    target_fps: float = 8.0,
    spatial_downsample: int = 2,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> tuple[np.ndarray, dict[str, float | tuple[int, int] | int]]:
    frames, info = load_video_frames(
        path,
        target_fps=target_fps,
        spatial_downsample=spatial_downsample,
    )
    clips = frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
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
        f"a={int(action)}",
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
        font=_load_overlay_font(),
    )
    return np.asarray(img)


def save_diffusion_mp4(
    context_frames: np.ndarray,
    intermediates: list[np.ndarray],
    output_path: str | Path,
    *,
    fps: float = 8.0,
) -> None:
    """Save a diffusion denoising trajectory as an MP4.

    Each video frame shows a horizontal strip:
        [ctx_0 | ctx_1 | … | ctx_L-1 ‖ generated_at_step_t]

    Context frames are static across all video frames; the generated column
    animates from pure noise (step 0) to the clean prediction (step N-1).
    Multiple batch items are stacked vertically.

    Args:
        context_frames: ``(B, L, H, W, C)`` conditioning frames (static).
        intermediates: list of ``(B, 1, H, W, C)`` arrays, one per denoising
            step, as returned by ``sample_euler(return_intermediates=True)``.
        output_path: destination ``.mp4`` file path.
        fps: playback speed of the output video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx_np = to_uint8_video(np.asarray(context_frames))  # (B, L, H, W, C)
    B, L, H, W, C = ctx_np.shape

    def _to_rgb(arr: np.ndarray) -> np.ndarray:
        """Ensure last dim is 3 (broadcast grayscale, strip alpha)."""
        if arr.shape[-1] == 1:
            return np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            return arr[..., :3]
        return arr

    ctx_rgb = _to_rgb(ctx_np)  # (B, L, H, W, 3)
    sep = np.zeros((H, 2, 3), dtype=np.uint8)  # thin black column between ctx and gen

    video_frames: list[np.ndarray] = []
    for step_x in intermediates:
        gen_np = to_uint8_video(np.asarray(step_x))  # (B, 1, H, W, C)
        gen_rgb = _to_rgb(gen_np[:, 0])  # (B, H, W, 3)

        rows: list[np.ndarray] = []
        for b in range(B):
            ctx_strip = np.concatenate(
                [ctx_rgb[b, i] for i in range(L)], axis=1
            )  # (H, L*W, 3)
            row = np.concatenate(
                [ctx_strip, sep, gen_rgb[b]], axis=1
            )  # (H, L*W+2+W, 3)
            rows.append(row)

        video_frames.append(np.concatenate(rows, axis=0))  # (B*H, total_W, 3)

    iio.mimsave(str(output_path), video_frames, fps=fps)


def save_video_grid(
    videos: np.ndarray,
    output_path: str | Path,
    *,
    grid_size: int = 4,
    fps: float = 8.0,
) -> None:
    """Tile up to ``grid_size**2`` videos into a square grid and save as MP4.

    Args:
        videos: ``(N, T, H, W, C)`` array of videos in ``[-1, 1]``. Only the
            first ``grid_size**2`` are used; if fewer are given the remaining
            grid cells are left black.
        output_path: destination ``.mp4`` file path.
        grid_size: number of videos per row/column (``4`` -> 4x4 = 16 videos).
        fps: playback speed of the output video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vids = to_uint8_video(np.asarray(videos))  # (N, T, H, W, C)
    n, t, h, w, c = vids.shape
    cells = grid_size * grid_size
    grid = np.zeros((cells, t, h, w, c), dtype=np.uint8)
    grid[: min(n, cells)] = vids[:cells]

    # (cells, T, H, W, C) -> (T, grid_size*H, grid_size*W, C)
    grid = grid.reshape(grid_size, grid_size, t, h, w, c)
    grid = grid.transpose(2, 0, 3, 1, 4, 5)
    grid = grid.reshape(t, grid_size * h, grid_size * w, c)

    iio.mimsave(str(output_path), list(grid), fps=fps)


def save_clip_previews(
    clips: np.ndarray,
    output_dir: str | Path,
    *,
    max_clips: int = 4,
    fps: float = 8.0,
    actions: np.ndarray | None = None,
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
