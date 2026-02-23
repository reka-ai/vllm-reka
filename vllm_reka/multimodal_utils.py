# ABOUTME: Image and video processing utilities for Reka multimodal models.
# ABOUTME: Includes ImageProcessor, VideoProcessor, and YasaVideoBackend (OpenCV-based).
import math
import os
import tempfile
from functools import partial
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import SiglipImageProcessor

from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings,
    placeholder_token_id,
) -> torch.Tensor:
    if isinstance(placeholder_token_id, (list, tuple)):
        is_multimodal = torch.isin(
            input_ids,
            torch.tensor(placeholder_token_id, device=input_ids.device),
        )
    else:
        is_multimodal = (input_ids == placeholder_token_id)
    return _merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeddings, is_multimodal)

DEFAULT_VIDEO_NUM_FRAMES = 32

# Token IDs shared across multimodal models
_IMAGE_PLACEHOLDER_TOKEN_ID = 100278
_START_IMAGE_TOKEN = 100279
_END_IMAGE_TOKEN = 100280
_START_VIDEO_TOKEN = 100284
_END_VIDEO_TOKEN = 100285

USE_IMAGE_PATCHING = os.getenv("USE_IMAGE_PATCHING", "1") == "1"


def _get_default_video_num_frames(ctx) -> int:
    """Default video frame count from --media-io-kwargs, or vLLM default."""
    mm_config = ctx.get_mm_config()
    return (mm_config.media_io_kwargs.get("video")
            or {}).get("num_frames", DEFAULT_VIDEO_NUM_FRAMES)


@VIDEO_LOADER_REGISTRY.register("yasa")
class YasaVideoBackend(VideoLoader):
    """YASA video loader using OpenCV with timestamp extraction.

    Samples frames uniformly and returns timestamps for each frame.
    PyAV integration planned for more efficient long video handling.
    """

    @staticmethod
    def _sample_indices(num_frames: int, total_frames: int,
                        sampling: str) -> np.ndarray:
        if num_frames == -1 or total_frames <= num_frames:
            return np.arange(total_frames, dtype=int)

        if sampling == "uniform":
            return np.linspace(0, total_frames - 1, num_frames, dtype=int)

        if sampling == "random":
            return np.sort(
                np.random.choice(np.arange(total_frames),
                                 num_frames,
                                 replace=False)).astype(int)

        if sampling == "chunk":
            # Split timeline into chunks and pick one random frame per chunk.
            chunk_size = total_frames // num_frames
            extra_frames = total_frames % num_frames
            sampled_frames = []
            for i in range(num_frames):
                start = i * chunk_size + min(i, extra_frames)
                end = start + chunk_size + (1 if i < extra_frames else 0)
                sampled_frames.append(int(np.random.randint(start, end)))
            return np.array(sampled_frames, dtype=int)

        raise ValueError(f"Unsupported video sampling mode: {sampling}. "
                         "Expected one of: chunk, uniform, random.")

    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = DEFAULT_VIDEO_NUM_FRAMES,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video from bytes using OpenCV + tempfile
        and uniformly sample exactly `num_frames` frames.
        """
        # Write bytes → tempfile (OpenCV limitation)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        try:
            os.write(fd, data)
            os.close(fd)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if total_frames <= 0 or fps <= 0:
                raise RuntimeError("Invalid video metadata")
            sampling = kwargs.get("sampling", "uniform")
            frame_indices = cls._sample_indices(num_frames, total_frames,
                                                sampling=sampling)
            frames = np.empty((len(frame_indices), height, width, 3),
                              dtype=np.uint8)
            timestamps = [idx / fps for idx in frame_indices]
            # -------- Efficient decode path ----------
            target_set = set(frame_indices)
            out_idx = 0

            for idx in range(total_frames):
                ok = cap.grab()
                if not ok:
                    break

                if idx in target_set:
                    ret, frame = cap.retrieve()
                    if not ret:
                        raise RuntimeError(f"Failed retrieving frame {idx}")
                    frames[out_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_idx += 1

                    if out_idx == len(frame_indices):
                        break
            cap.release()
            if out_idx != len(frame_indices):
                raise RuntimeError(
                    f"Expected {len(frame_indices)} frames, got {out_idx}")
            metadata = {
                "total_frames": total_frames,
                "fps": fps,
                "duration": total_frames / fps,
                "frame_indices": frame_indices.tolist(),
                "timestamps": timestamps,
                "backend": f"opencv-tempfile-{sampling}",
                "sampling": sampling,
            }
            return frames, metadata
        finally:
            if os.path.exists(path):
                os.unlink(path)


def _ensure_divides(length: int, patch_size: int) -> int:
    """Rounds `length` to the nearest multiple of `patch_size`."""
    return max(round(length / patch_size) * patch_size, patch_size)


def _find_best_resize(
    original_size: tuple[int, int],
    target_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """
    Finds the optimal resized dimensions that:
    - Preserve aspect ratio.
    - Are divisible by `patch_size`.
    - Approximate `target_resolution` (if upscaling allowed
    or original is larger).
    """
    width, height = original_size
    if not allow_upscale and (width * height
                              <= target_resolution * target_resolution):
        return (_ensure_divides(width, patch_size),
                _ensure_divides(height, patch_size))

    aspect_ratio = width / height
    new_height = int(target_resolution / math.sqrt(aspect_ratio))
    new_width = int(new_height * aspect_ratio)
    return (_ensure_divides(new_width, patch_size),
            _ensure_divides(new_height, patch_size))


def _generate_grids(total_tiles: int) -> list[tuple[int, int]]:
    """Generates all possible (x, y) grids for a given tile count."""
    grids = []
    for x in range(1, total_tiles + 1):
        if total_tiles % x == 0:
            grids.append((x, total_tiles // x))
    return grids


def _get_refine_size(
    original_size: tuple[int, int],
    grid: tuple[int, int],
    target_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """
    Computes the refined size for splitting an image into `grid` tiles,
    ensuring each tile is divisible by `patch_size`.
    """
    grid_x, grid_y = grid
    tile_width = _ensure_divides(original_size[0] // grid_x, patch_size)
    tile_height = _ensure_divides(original_size[1] // grid_y, patch_size)

    # Ensure tiles approximate the target resolution
    best_tile_size = _find_best_resize(
        (tile_width, tile_height),
        target_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )
    return (best_tile_size[0] * grid_x, best_tile_size[1] * grid_y)


def _split_into_patches(image: Image.Image,
                        grid: tuple[int, int]) -> list[Image.Image]:
    """Splits an image into `grid[0] x grid[1]` patches."""
    width, height = image.size
    patch_width, patch_height = width // grid[0], height // grid[1]
    return [
        image.crop((j, i, j + patch_width, i + patch_height))
        for i in range(0, height, patch_height)
        for j in range(0, width, patch_width)
    ]


class ImageProcessor:

    DO_RESCALE: bool = True
    DO_NORMALIZE: bool = True
    RESCALE_FACTOR: float = 0.00392156862745098
    IMAGE_MEAN: list[float] = [0.5, 0.5, 0.5]
    IMAGE_STD: list[float] = [0.5, 0.5, 0.5]
    RESAMPLE: int = 3

    def __init__(self, config):
        self.config = config
        self.vision_preprocessor = SiglipImageProcessor(
            do_resize=True,
            size={
                "height": self.config.vision_config.image_size,
                "width": self.config.vision_config.image_size
            },
            resample=self.RESAMPLE,
            image_mean=self.IMAGE_MEAN,
            image_std=self.IMAGE_STD,
            return_tensors="pt")

    def batch_preprocess(self,
                         images: list[Image.Image]) -> dict[str, torch.Tensor]:
        processed_images, patches_per_image = self.preprocess(images)
        result = {
            "pixel_values": processed_images,
            "patches_per_image": patches_per_image
        }
        return result

    def preprocess(
            self,
            images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        preprocess_images_func = partial(
            self._preprocess_anyres_image_uhd,
            max_tiles=self.config.vision_max_tiles_num,
            target_resolution=self.config.vision_config.image_size,
            patch_size=self.config.vision_config.patch_size,
            never_split=False) if USE_IMAGE_PATCHING else partial(
                self._preprocess_image_default,
                target_resolution=self.config.vision_config.image_size)
        preprocessed_images: list[Image.Image] = []
        patches_per_image: list[int] = []
        for image in images:
            image_patches = preprocess_images_func(image=image)
            patches_per_image.append(len(image_patches))
            preprocessed_images.extend(image_patches)
        processed_images = self.vision_preprocessor.preprocess(
            preprocessed_images, return_tensors="pt")["pixel_values"]
        # get flattended 4 output tensor
        return processed_images, torch.tensor(patches_per_image)

    def get_max_dummy_image(self) -> Image.Image:
        width, height = (self.config.vision_config.image_size *
                         self.config.vision_max_tiles_num,
                         self.config.vision_config.image_size)
        return Image.new('RGB', (width, height), color='black')

    def get_max_yasa_image_tokens(self) -> int:
        if not USE_IMAGE_PATCHING:
            return self.config.num_query_tokens + 2  # +2 for start/end tokens
        return self.config.num_query_tokens * (
            self.config.vision_max_tiles_num + 1) + 2  # +2 for start/end tokens

    def get_num_image_tokens(self, image: Image.Image) -> int:

        if not USE_IMAGE_PATCHING:
            result = self.config.num_query_tokens
            return result

        width, height = image.size
        image_size_sq = self.config.vision_config.image_size**2
        area_ratio = (width * height) / image_size_sq
        num_tiles = min(math.ceil(area_ratio),
                        self.config.vision_max_tiles_num)
        if num_tiles <= 1:
            result = self.config.num_query_tokens
            return result
        else:
            # prepend original image resized
            result = (num_tiles + 1) * self.config.num_query_tokens
            return result

    def get_num_image_tokens_from_size(self, width: int, height: int) -> int:
        if not USE_IMAGE_PATCHING:
            return self.config.num_query_tokens
        area_ratio = (width * height) / (self.config.vision_config.image_size**
                                         2)
        num_tiles = min(math.ceil(area_ratio),
                        self.config.vision_max_tiles_num)
        if num_tiles <= 1:
            return self.config.num_query_tokens
        else:
            # prepend original image resized
            return (num_tiles + 1) * self.config.num_query_tokens

    @staticmethod
    def _preprocess_image_default(image: Image.Image,
                                  target_resolution: int) -> list[Image.Image]:
        return [image.resize((target_resolution, target_resolution))]

    @staticmethod
    def _preprocess_anyres_image_uhd(
        image: Image.Image,
        max_tiles: int = 9,
        target_resolution: int = 448,
        patch_size: int = 14,
        never_split: bool = False,
    ) -> list[Image.Image]:
        """
        Processes an image by:
        - Splitting into tiles if too large (while preserving aspect ratio).
        - Ensuring all patches are divisible by `patch_size`.
        - Returning the resized source image + patches (if split).
        """
        width, height = image.size
        log_aspect_ratio = math.log(width / height)
        area_ratio = (width * height) / (target_resolution * target_resolution)
        num_tiles = min(math.ceil(area_ratio), max_tiles)
        if never_split or num_tiles <= 1:
            best_size = _find_best_resize((width, height),
                                          target_resolution,
                                          patch_size,
                                          allow_upscale=True)
            return [image.resize(best_size, Image.Resampling.BICUBIC)]

        # Generate candidate grids (e.g., for 9 tiles: [(1,9), (3,3), (9,1)])
        candidate_grids = _generate_grids(num_tiles)
        if num_tiles < max_tiles:  # Also check grids for num_tiles + 1
            candidate_grids.extend(_generate_grids(num_tiles + 1))
        # Select grid with closest aspect ratio match
        best_grid = min(
            candidate_grids,
            key=lambda g: abs(log_aspect_ratio - math.log(g[0] / g[1])),
            default=(1, 1),
        )
        refine_size = _get_refine_size((width, height),
                                       best_grid,
                                       target_resolution,
                                       patch_size,
                                       allow_upscale=True)
        patches = _split_into_patches(
            image.resize(refine_size, Image.Resampling.BICUBIC), best_grid)
        result = [image] + patches
        return result


class VideoProcessor:
    """Video frame processor for YASA MMLM.

    Processes video frames without patching - each frame is resized to
    target resolution and treated as a single patch.
    """

    DO_RESCALE: bool = True
    DO_NORMALIZE: bool = True
    RESCALE_FACTOR: float = 0.00392156862745098
    IMAGE_MEAN: list[float] = [0.5, 0.5, 0.5]
    IMAGE_STD: list[float] = [0.5, 0.5, 0.5]
    RESAMPLE: int = 3

    def __init__(self, config):
        self.config = config
        self.vision_preprocessor = SiglipImageProcessor(
            do_resize=True,
            size={
                "height": self.config.vision_config.image_size,
                "width": self.config.vision_config.image_size
            },
            resample=self.RESAMPLE,
            image_mean=self.IMAGE_MEAN,
            image_std=self.IMAGE_STD,
            return_tensors="pt")

    def frames_to_pil_images(self, frames: npt.NDArray) -> list[Image.Image]:
        """Convert video frames (N, H, W, C) to list of PIL Images."""
        return [Image.fromarray(frame) for frame in frames]

    def preprocess_video_frames(
            self, frames: list[Image.Image]) -> tuple[torch.Tensor, int]:
        """Preprocess video frames without patching.

        Each frame is resized to target resolution (no tiling/splitting).

        Args:
            frames: List of PIL Images (video frames)

        Returns:
            Tuple of (pixel_values tensor [N, C, H, W], num_frames)
        """
        target_resolution = self.config.vision_config.image_size

        # Resize each frame to target resolution (no patching)
        preprocessed_frames = [
            frame.resize((target_resolution, target_resolution),
                         Image.Resampling.BICUBIC) for frame in frames
        ]

        # Process through vision preprocessor
        processed = self.vision_preprocessor.preprocess(
            preprocessed_frames, return_tensors="pt")["pixel_values"]

        return processed, len(frames)

    def batch_preprocess_videos(
            self, videos: list[tuple[npt.NDArray,
                                     dict[str, Any]]]) -> dict[str, Any]:
        """Batch preprocess multiple videos.

        Args:
            videos: List of (frames_array, metadata) tuples

        Returns:
            Dict with pixel_values, frames_per_video, and timestamps_per_video
        """
        all_pixel_values = []
        frames_per_video = []
        timestamps_per_video = []

        for video in videos:
            frames, metadata = video
            # Convert numpy frames to PIL images
            pil_frames = self.frames_to_pil_images(frames)

            # Preprocess without patching
            pixel_values, num_frames = self.preprocess_video_frames(pil_frames)

            all_pixel_values.append(pixel_values)
            frames_per_video.append(num_frames)

            # Extract timestamps from metadata
            timestamps = metadata.get("timestamps", [])
            if not timestamps:
                # Fallback: compute from fps and frame indices
                fps = metadata.get("fps", 1.0)
                frame_indices = metadata.get("frames_indices",
                                             list(range(num_frames)))
                timestamps = [
                    idx / fps if fps > 0 else 0.0 for idx in frame_indices
                ]
            timestamps_per_video.append(timestamps)

        # Concatenate all pixel values
        if all_pixel_values:
            combined_pixel_values = torch.cat(all_pixel_values, dim=0)
        else:
            combined_pixel_values = torch.empty(0)

        return {
            "video_pixel_values": combined_pixel_values,
            "frames_per_video": torch.tensor(frames_per_video),
            "timestamps_per_video": timestamps_per_video,
        }

    def get_num_video_tokens(self, num_frames: int) -> int:
        """Get number of tokens for a video with given number of frames.

        Each frame = 1 patch = num_query_tokens tokens (no patching).
        """
        return num_frames * self.config.num_query_tokens

    def get_max_video_tokens(self,
                             max_frames: int = DEFAULT_VIDEO_NUM_FRAMES
                             ) -> int:
        """Get maximum number of tokens for a video."""
        return self.get_num_video_tokens(max_frames)

    def get_max_dummy_video(self,
                            num_frames: int = DEFAULT_VIDEO_NUM_FRAMES
                            ) -> npt.NDArray:
        """Create dummy video frames for profiling."""
        image_size = self.config.vision_config.image_size
        return np.zeros((num_frames, image_size, image_size, 3),
                        dtype=np.uint8)
