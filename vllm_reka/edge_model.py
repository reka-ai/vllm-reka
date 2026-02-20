# ABOUTME: YasaMMLMV2ForConditionalGeneration (Edge) with ConvNextV2 vision encoder.
# ABOUTME: Supports image and video inputs with adaptive pooling and sinusoidal position embeddings.

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (BatchFeature, ConvNextImageProcessor,
                          ConvNextV2Model, PreTrainedTokenizer)

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.inputs import (MultiModalFieldConfig,
                                    MultiModalKwargsItems, NestedTensors)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix)

from vllm_reka.multimodal_utils import (DEFAULT_VIDEO_NUM_FRAMES,
                                        ImageProcessor,
                                        merge_multimodal_embeddings,
                                        _IMAGE_PLACEHOLDER_TOKEN_ID,
                                        _START_IMAGE_TOKEN, _END_IMAGE_TOKEN,
                                        _START_VIDEO_TOKEN, _END_VIDEO_TOKEN,
                                        _get_default_video_num_frames)

# Default image size for ConvNextV2
DEFAULT_IMAGE_SIZE = 224
USE_IMAGE_PATCHING = os.getenv("USE_IMAGE_PATCHING", "1") == "1"


class YasaMMLMV2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


class YasaMMLMV2ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


YasaMMLMV2ImageInputs = Union[YasaMMLMV2ImagePixelInputs,
                              YasaMMLMV2ImageEmbeddingInputs]


class YasaMMLMV2VideoPixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


class YasaMMLMV2VideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


YasaMMLMV2VideoInputs = Union[YasaMMLMV2VideoPixelInputs,
                              YasaMMLMV2VideoEmbeddingInputs]


def get_2d_sincos_pos_embed(embed_dim: int, image_size) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings using torch (more efficient).

    Args:
        embed_dim: embedding dimension
        image_size: image_size or (image_height, image_width)

    Returns:
        pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = torch.arange(grid_h_size, dtype=torch.float32)
    grid_w = torch.arange(grid_w_size, dtype=torch.float32)
    # meshgrid with indexing='xy' matches numpy behavior (w goes first)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack([grid_w, grid_h], dim=0)  # [2, H, W]

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim,
                                                   grid)  # (H, W, D)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int,
                                       grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                               grid[0])  # (H, W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                               grid[1])  # (H, W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=-1)  # (H, W, D)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                       pos: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: output dimension for each position
        pos: positions to be encoded, shape (H, W)

    Returns:
        emb: positional embeddings, shape (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    # outer product: (H, W) x (D/2,) -> (H, W, D/2)
    out = torch.einsum("hw,d->hwd", pos, omega)

    emb_sin = torch.sin(out)  # (H, W, D/2)
    emb_cos = torch.cos(out)  # (H, W, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (H, W, D)
    return emb


class YasaMMLMV2ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP):
    """YASA Edge MMLM model with ConvNextV2 vision encoder."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<REKA_IMG_TOKEN>"
        if modality.startswith("video"):
            return "<REKA_IMG_TOKEN>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        # Initialize ConvNextV2 vision encoder
        if not hasattr(config.vision_config, "num_stages"):
            config.vision_config.num_stages = 4
        self.vision_model = ConvNextV2Model(config.vision_config)

        # Get vision hidden size - compute from vision_config
        if hasattr(config.vision_config, "hidden_sizes"):
            vision_hidden_size = config.vision_config.hidden_sizes[-1]
        else:
            vision_hidden_size = 2816  # Default for ConvNextV2-Huge
        self.vision_hidden_size = vision_hidden_size

        # Adaptive pooling to get fixed spatial dimensions
        # For num_query_tokens=49, we want 7x7 spatial output
        pool_size = int(config.num_query_tokens**0.5)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(pool_size)

        # Language projection: vision_hidden_size -> text_hidden_size
        self.language_projection = nn.Sequential(
            nn.Linear(
                vision_hidden_size,
                config.text_config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
            ),
        )

        # Ensure text_config has architectures set for vLLM registry
        if (not hasattr(config.text_config, "architectures")
                or not config.text_config.architectures):
            config.text_config.architectures = ["YasaCausalLM"]

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.register_buffer(
            "vision_pos_embed",
            get_2d_sincos_pos_embed(vision_hidden_size,
                                    image_size=50).to(torch.bfloat16),
            persistent=False,
        )
        self.add_vision_pos_embed = config.use_vision_pos_embed

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        """Validate pixel values shape for ConvNextV2."""
        image_size = getattr(self.config.vision_config, "image_size",
                             DEFAULT_IMAGE_SIZE)
        expected_dims = (3, image_size, image_size)
        actual_dims = tuple(data.shape[1:])
        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> YasaMMLMV2ImageInputs | None:
        """Parse image input."""
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        num_images = kwargs.pop("num_images", None)
        tiles_per_image = kwargs.pop("tiles_per_image", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            num_images_list = cast(torch.Tensor, num_images).flatten().tolist()
            tiles_per_image_list = (
                cast(torch.Tensor, tiles_per_image).flatten().tolist()
                if tiles_per_image is not None else [1] *
                len(num_images_list)  # Default: 1 tile per image
            )
            # Handle V1 engine batching - may be list if shapes differ
            if isinstance(pixel_values, list):
                pixel_values = torch.cat(pixel_values, dim=0)
            else:
                pixel_values = cast(torch.Tensor, pixel_values)
            if pixel_values.ndim != 4:
                B, N, C, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B * N, C, H, W)
            assert pixel_values.ndim == 4, (
                f"Expected pixel_values to be a 4D tensor [N, C, H, W]"
                f", got {pixel_values.ndim}D tensor")
            total_tiles = sum(tiles_per_image_list)
            assert total_tiles == pixel_values.shape[0], (
                f"Expected sum(tiles_per_image)={total_tiles} "
                f"to equal total tiles={pixel_values.shape[0]}")
            metadata = {
                "num_images": num_images_list,
                "tiles_per_image": tiles_per_image_list,
            }
            return YasaMMLMV2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
                metadata=metadata,
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            num_images_list = (cast(torch.Tensor, num_images).tolist()
                               if num_images else [])
            return YasaMMLMV2ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
                metadata={"num_images": num_images_list},
            )
        raise ValueError(
            "Either pixel_values or image_embeds must be provided.")

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> YasaMMLMV2VideoInputs | None:
        """Parse video input - mirrors _parse_and_validate_image_input.

        Video frames go through the same vision pipeline as images.
        """
        video_frames_pixels = kwargs.pop("video_frames_pixels", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_num_frames = kwargs.pop("video_num_frames", None)
        if video_frames_pixels is None and video_embeds is None:
            return None
        if video_frames_pixels is not None:
            video_num_frames = cast(torch.Tensor,
                                    video_num_frames).flatten().tolist()
            video_frames_pixels = cast(torch.Tensor, video_frames_pixels)
            if video_frames_pixels.ndim != 4:
                B, N, C, H, W = video_frames_pixels.shape
                video_frames_pixels = video_frames_pixels.reshape(
                    B * N, C, H, W)
            assert video_frames_pixels.ndim == 4, (
                f"Expected video_frames_pixels to be a 4D tensor, "
                f"got {video_frames_pixels.ndim}D tensor")
            assert sum(video_num_frames) == video_frames_pixels.shape[0], (
                f"Expected total number of frames and video frame pixels to "
                f"be the same, got {sum(video_num_frames)} and "
                f"{video_frames_pixels.shape[0]}.")
            metadata = {"frames_per_video": video_num_frames}
            return YasaMMLMV2VideoPixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(video_frames_pixels),
                metadata=metadata,
            )
        if video_embeds is not None:
            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            frames_per_video = cast(torch.Tensor, video_num_frames).tolist()
            return YasaMMLMV2VideoEmbeddingInputs(
                type="video_embeds",
                data=video_embeds,
                metadata={"frames_per_video": frames_per_video},
            )
        raise ValueError(
            "Either video_frames_pixels or video_embeds must be provided.")

    def _image_pixels_to_features(self,
                                  pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from ConvNextV2."""
        # ConvNextV2 returns BaseModelOutputWithPoolingAndNoAttention
        # last_hidden_state shape: [B, C, H, W]
        pixel_values = pixel_values.to(
            dtype=self.vision_model.embeddings.patch_embeddings.weight.dtype)
        outputs = self.vision_model(pixel_values)
        return outputs.last_hidden_state

    def _process_image_pixels(
            self, inputs: YasaMMLMV2ImagePixelInputs) -> torch.Tensor:
        """Process pixel values through vision model."""
        pixel_values = inputs["data"]
        # Get vision features: [B, C, H, W]
        vision_features = self._image_pixels_to_features(pixel_values)
        return vision_features

    def _process_image_input(
            self, image_input: YasaMMLMV2ImageInputs) -> torch.Tensor:
        """Process image input to get embeddings.

        Follows training logic from _encode_vision_adaptive_2d_avg_pooling:
        1. Get vision features from CNN: [B, C, H, W]
        2. Convert to sequence format: [B, H*W, C]
        3. Add positional embeddings (if enabled)
        4. Convert back to spatial format: [B, C, H, W]
        5. Apply adaptive pooling
        6. Flatten and project to language space
        """
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        # Process through vision model: [B, C, H, W]
        image_features = self._process_image_pixels(image_input)
        batch_size, channels, h, w = image_features.shape
        seq_length = h * w

        # Convert to sequence format for positional embeddings: [B, H*W, C]
        image_features = image_features.flatten(2)  # [B, C, H*W]
        image_features = image_features.permute(0, 2,
                                                1).contiguous()  # [B, H*W, C]

        # Add positional embeddings BEFORE pooling (matching training)
        if self.add_vision_pos_embed:
            vision_pos_embed = (
                self.vision_pos_embed.view(
                    -1, self.vision_hidden_size).unsqueeze(
                        0)  # [1, max_seq_length, hidden_size]
                .to(image_features.device, dtype=image_features.dtype))
            # Slice to match actual sequence length
            image_features = (image_features +
                              vision_pos_embed[:, :seq_length, :])
        # Convert back to spatial format for adaptive pooling: [B, C, H, W]
        image_features = image_features.permute(0, 2,
                                                1).contiguous()  # [B, C, H*W]
        image_features = image_features.reshape(batch_size, channels, h, w)

        # Apply adaptive pooling: [B, C, pool_h, pool_w]
        image_features = self.adaptive_pooling(image_features)

        # Flatten spatial dims and transpose: [B, pool_h*pool_w, C]
        image_features = image_features.flatten(2).permute(0, 2,
                                                           1).contiguous()

        # Project to language model dimension
        # (NO layer norm - matching training)
        vision_embeds = self.language_projection(image_features)

        # Flatten batch and sequence: [B * seq_len, hidden_size]
        return vision_embeds.reshape(-1, vision_embeds.size(-1))

    def _get_image_multimodal_embeddings(
            self, **kwargs) -> list[torch.Tensor] | None:
        """Get embeddings for all images."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)
        metadata = image_input.get("metadata", {})
        num_images_list = metadata.get("num_images", [])
        tiles_per_image_list = metadata.get("tiles_per_image", [])
        tokens_per_tile = getattr(self.config, "num_query_tokens", 64)

        if not num_images_list:
            raise ValueError("Missing `num_images` in image metadata.")
        if not tiles_per_image_list:
            raise ValueError("Missing `tiles_per_image` in image metadata.")

        # Compute token counts per image
        tokens_per_img = [
            n_tiles * tokens_per_tile for n_tiles in tiles_per_image_list
        ]
        # Split efficiently with torch.split
        embeddings_list = list(
            torch.split(vision_embeddings, tokens_per_img, dim=0))
        return embeddings_list

    def _get_video_multimodal_embeddings(
            self, **kwargs) -> list[torch.Tensor] | None:
        """Get video embeddings - mirrors _get_image_multimodal_embeddings."""
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is None:
            return None
        if video_input["type"] == "video_embeds":
            vision_embeddings = video_input["data"]
        else:
            # Reuse same vision pipeline (video frames = image patches)
            vision_embeddings = self._process_image_input(video_input)
        metadata = video_input.get("metadata", {})
        frames_per_video = metadata.get("frames_per_video", [])
        tokens_per_frame = getattr(self.config, "num_query_tokens", 49)
        if not frames_per_video:
            raise ValueError("Missing `frames_per_video` in video metadata.")
        tokens_per_video = [f * tokens_per_frame for f in frames_per_video]
        embeddings_list = list(
            torch.split(vision_embeddings, tokens_per_video, dim=0))
        return embeddings_list

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor]:
        """Get embeddings for all multimodal inputs."""
        all_embeddings: list[torch.Tensor] = []
        # Process images
        image_embs = self._get_image_multimodal_embeddings(**kwargs)
        if image_embs:
            all_embeddings.extend(image_embs)
        # Process videos
        video_embs = self._get_video_multimodal_embeddings(**kwargs)
        if video_embs:
            all_embeddings.extend(video_embs)
        return all_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
    ) -> torch.Tensor:
        """Get input embeddings with multimodal tokens merged."""
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        if (isinstance(multimodal_embeddings, (list, tuple))
                and len(multimodal_embeddings) > 0):
            mm_concat = torch.cat(list(multimodal_embeddings), dim=0)
            placeholder_count = ((
                input_ids == _IMAGE_PLACEHOLDER_TOKEN_ID).sum().item())
            if mm_concat.shape[0] == placeholder_count:
                return merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    mm_concat,
                    _IMAGE_PLACEHOLDER_TOKEN_ID,
                )
            padded_embeddings = self._pad_and_validate_image_embeddings(
                input_ids=input_ids,
                multimodal_embeddings=multimodal_embeddings)
            return self._merge_padded_image_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                padded_embeddings=padded_embeddings,
            )

        return merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            _IMAGE_PLACEHOLDER_TOKEN_ID,
        )

    def _pad_and_validate_image_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Pad image embeddings with start/end tokens."""
        start_tok = torch.tensor([_START_IMAGE_TOKEN], device=input_ids.device)
        end_tok = torch.tensor([_END_IMAGE_TOKEN], device=input_ids.device)
        start_emb = self.language_model.get_input_embeddings(
            start_tok).squeeze(0)
        end_emb = self.language_model.get_input_embeddings(end_tok).squeeze(0)

        padded_list: list[torch.Tensor] = []
        for img_emb in multimodal_embeddings:
            if img_emb.ndim == 1:
                img_emb = img_emb.unsqueeze(0)
            padded = torch.vstack(
                (start_emb.unsqueeze(0), img_emb, end_emb.unsqueeze(0)))
            padded_list.append(padded)

        expected_mask_ids = torch.tensor(
            [
                _START_IMAGE_TOKEN, _IMAGE_PLACEHOLDER_TOKEN_ID,
                _END_IMAGE_TOKEN
            ],
            device=input_ids.device,
        )
        expected_count = torch.isin(input_ids, expected_mask_ids).sum().item()
        produced_count = int(sum(p.shape[0] for p in padded_list))
        if produced_count != expected_count:
            raise ValueError(
                f"YasaMMLMV2: produced image tokens={produced_count} != "
                f"reserved placeholders={expected_count}. "
                f"Check num_images and num_query_tokens.")

        return tuple(padded_list)

    def _merge_padded_image_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        padded_embeddings: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Merge padded image embeddings into input embeddings."""
        return merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            padded_embeddings,
            [
                _START_IMAGE_TOKEN, _IMAGE_PLACEHOLDER_TOKEN_ID,
                _END_IMAGE_TOKEN
            ],
        )

    def forward(
        self,
        *,
        intermediate_tensors: IntermediateTensors | None,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Union[IntermediateTensors, torch.Tensor]:
        """Run forward pass for Yasa Edge MMLM."""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            vision_embeddings = self.embed_multimodal(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Configure weight name mapping
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.": "",  # Strip model. prefix
                "vision_model.backbone.": "vision_model.",  # Strip backbone
                "lm_head.":
                "language_model.lm_head.",  # External lm_head → internal
            })
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "qformer",
                "query_tokens",
                "audio_",
                "optimizer",
                "grad_acc",
                "casual_mask",
                "audio_language_projection",
                "audio_query_tokens",
                "audio2qformer_state",
                "audioQformer2lan",
                "audio_ln",
                "vision_ln",
            ],
        )
        return loader.load_weights(mapper.apply(weights))


class YasaMMLMV2ImageProcessor:
    """Image processor for YasaMMLMV2 using ConvNextImageProcessor."""

    MAX_TILES = 4

    def __init__(self, config):
        self.config = config
        image_size = getattr(config.vision_config, "image_size",
                             DEFAULT_IMAGE_SIZE)
        self.image_processor = ConvNextImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            crop_size={
                "height": image_size,
                "width": image_size
            },
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],  # ImageNet mean
            image_std=[0.229, 0.224, 0.225],  # ImageNet std
        )
        self.grid_points = [(2, 2), (1, 2), (2, 1), (1, 3), (3, 1)]
        self.image_size = image_size
        self.patch_size = self.config.vision_config.patch_size

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Preprocess images for ConvNextV2."""
        tiles_per_image = []
        all_tiles = []
        if USE_IMAGE_PATCHING:
            for image in images:
                image_tiles = ImageProcessor._preprocess_anyres_image_uhd(
                    image,
                    self.MAX_TILES,
                    self.image_size,
                    self.patch_size,
                    False,
                )
                tiles_per_image.append(len(image_tiles))
                all_tiles.extend(image_tiles)
        else:
            for image in images:
                tiles_per_image.append(1)
                all_tiles.append(image)
        processed = self.image_processor(all_tiles, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "num_images": torch.ones(len(images), dtype=torch.long),
            "tiles_per_image": torch.tensor(tiles_per_image),
        }

    def get_num_image_tokens(self) -> int:
        """Get number of tokens per image."""
        return self.config.num_query_tokens

    def get_max_dummy_image(self) -> Image.Image:
        """Create dummy image for profiling."""
        return Image.new("RGB", (self.image_size, self.image_size),
                         color="black")


class YasaMMLMV2VideoProcessor:
    """Video frame processor for YasaMMLMV2 using ConvNextImageProcessor.

    Processes video frames without tiling - each frame is resized to
    image_size and run through the same preprocessor as images.
    """

    def __init__(self, config):
        self.config = config
        self._image_processor = YasaMMLMV2ImageProcessor(config)

    def _frames_to_pil(
            self,
            frames: list[Image.Image] | list[np.ndarray]) -> list[Image.Image]:
        """Convert video frames to list of PIL Images."""
        out: list[Image.Image] = []
        for f in frames:
            if isinstance(f, Image.Image):
                out.append(f)
            else:
                out.append(Image.fromarray(np.asarray(f)))
        return out

    def preprocess_video_frames(
        self, frames: list[Image.Image] | list[np.ndarray]
    ) -> tuple[torch.Tensor, int]:
        """Preprocess video frames (no tiling).

        Returns:
            Tuple of (pixel_values [N, C, H, W], num_frames).
        """
        pil_frames = self._frames_to_pil(frames)
        # Use ConvNextImageProcessor directly so each frame = 1
        # patch (no tiling)
        processed = self._image_processor.image_processor(pil_frames,
                                                          return_tensors="pt")
        pixel_values = processed["pixel_values"]
        return pixel_values, len(pil_frames)

    def get_num_video_tokens(self, num_frames: int) -> int:
        """Tokens for a video with given number of frames."""
        return num_frames * self.config.num_query_tokens

    def get_max_video_tokens(self,
                             max_frames: int = DEFAULT_VIDEO_NUM_FRAMES
                             ) -> int:
        """Maximum number of tokens for a video."""
        return self.get_num_video_tokens(max_frames)

    def get_max_dummy_video(
            self,
            num_frames: int = DEFAULT_VIDEO_NUM_FRAMES
    ) -> tuple[np.ndarray, dict]:
        """Create dummy video frames with metadata for profiling."""
        image_size = self._image_processor.image_size
        frames = np.zeros((num_frames, image_size, image_size, 3),
                          dtype=np.uint8)
        metadata = {
            "timestamps": list(range(num_frames)),
            "fps": 1.0,
        }
        return frames, metadata


class YasaMMLMV2ProcessingInfo(BaseProcessingInfo):
    """Processing info for YASA Edge MMLM model."""

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer)
        return tokenizer

    def get_image_processor(self) -> YasaMMLMV2ImageProcessor:
        """Get the YASA Edge image processor."""
        return YasaMMLMV2ImageProcessor(self.get_hf_config())

    def get_video_processor(self) -> YasaMMLMV2VideoProcessor:
        """Get the YASA Edge video processor."""
        return YasaMMLMV2VideoProcessor(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_yasa_num_video_tokens(),
        }

    def get_max_image_tokens(self) -> int:
        config = self.get_hf_config()
        tokens_per_tile = getattr(config, "num_query_tokens", 64)
        max_tiles = (YasaMMLMV2ImageProcessor.MAX_TILES
                     if USE_IMAGE_PATCHING else 1)
        return (tokens_per_tile * max_tiles) + 2  # +2 for start/end tokens

    def get_max_yasa_num_video_tokens(self,
                                      max_frames: int | None = None) -> int:
        """Maximum number of tokens for a video.

        Each frame = num_query_tokens tokens (no tiling). +2
        for <video></video>.
        """
        if max_frames is None:
            max_frames = _get_default_video_num_frames(self.ctx)
        video_processor = self.get_video_processor()
        base_tokens = video_processor.get_max_video_tokens(max_frames)
        return base_tokens + 2  # start/end video tokens

    def get_max_yasa_dummy_video(self,
                                 num_frames: int | None = None
                                 ) -> tuple[np.ndarray, dict]:
        """Create dummy video frames with metadata for profiling."""
        if num_frames is None:
            num_frames = _get_default_video_num_frames(self.ctx)
        video_processor = self.get_video_processor()
        return video_processor.get_max_dummy_video(num_frames)


class YasaMMLMV2DummyInputsBuilder(
        BaseDummyInputsBuilder[YasaMMLMV2ProcessingInfo]):
    """Dummy inputs builder for YASA Edge MMLM model."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Build the text input corresponding to mm_counts."""
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        parts = ["<REKA_IMG_TOKEN>"] * num_images
        parts.extend(["<REKA_IMG_TOKEN>"] * num_videos)
        return " ".join(parts)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        """Build the multimodal input for maximum placeholder tokens."""
        mm_data: MultiModalDataDict = {}
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            dummy_image = self.get_dummy_image()
            mm_data["image"] = [dummy_image] * num_images
        num_videos = mm_counts.get("video", 0)
        if num_videos > 0:
            dummy_video = self.info.get_max_yasa_dummy_video()
            mm_data["video"] = [dummy_video] * num_videos
        return mm_data

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        mm_data: MultiModalDataDict = {}
        parts = []

        if num_images > 0:
            dummy_image = self.get_dummy_image()
            mm_data["image"] = [dummy_image] * num_images
            parts.extend(["<REKA_IMG_TOKEN>"] * num_images)
        if num_videos > 0:
            dummy_video = self.info.get_max_yasa_dummy_video()
            mm_data["video"] = [dummy_video] * num_videos
            parts.extend(["<REKA_IMG_TOKEN>"] * num_videos)

        prompt_text = " ".join(parts)
        return ProcessorInputs(
            prompt=prompt_text,
            mm_data=mm_data,
        )

    def get_dummy_image(self) -> Image.Image:
        """Create dummy image for profiling."""
        image_processor = self.info.get_image_processor()
        return image_processor.get_max_dummy_image()


class YasaMMLMV2MultiModalProcessor(
        BaseMultiModalProcessor[YasaMMLMV2ProcessingInfo]):
    """Multimodal processor for YASA Edge MMLM model."""

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor()
        video_processor = self.info.get_video_processor()
        ids = tokenizer.tiktoken.encode(prompt, allowed_special="all")
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Process images
        images = mm_data.get("images", [])
        if images:
            image_inputs = image_processor.preprocess(list(images))
            result["pixel_values"] = image_inputs["pixel_values"]
            result["num_images"] = image_inputs["num_images"]
            result["tiles_per_image"] = image_inputs["tiles_per_image"]

        # Process videos
        videos = mm_data.get("videos", [])
        if videos:
            video_frames_pixels_list = []
            frame_counts = []
            all_timestamps = []
            for video in videos:
                frames, metadata = video
                # frames: (N, H, W, C) ndarray or list of frames
                if hasattr(frames, "shape") and len(frames.shape) == 4:
                    frame_list = [frames[i] for i in range(frames.shape[0])]
                elif isinstance(frames, (list, tuple)):
                    frame_list = list(frames)
                else:
                    frame_list = [frames]
                (pixel_values, num_frames
                 ) = video_processor.preprocess_video_frames(frame_list)
                video_frames_pixels_list.append(pixel_values)
                frame_counts.append(num_frames)
                timestamps = metadata.get("timestamps",
                                          list(range(num_frames)))
                all_timestamps.extend(timestamps)
            result["video_frames_pixels"] = torch.cat(video_frames_pixels_list,
                                                      dim=0)
            result["video_num_frames"] = torch.tensor(frame_counts)
            result["video_timestamps"] = torch.tensor(all_timestamps,
                                                      dtype=torch.float32)
        return BatchFeature(result)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        cfg = {}
        # Image fields
        num_images = hf_inputs.get("num_images")
        tiles_per_image = hf_inputs.get("tiles_per_image")
        if num_images is not None:
            # tiles_per_image is required when we have images to process
            assert tiles_per_image is not None, "tiles_per_image is required"
            # Use tiles_per_image for pixel_values (multiple tiles per image)
            cfg["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", tiles_per_image, dim=0)
            cfg["num_images"] = MultiModalFieldConfig.batched("image")
            cfg["tiles_per_image"] = MultiModalFieldConfig.batched("image")
        # Video fields
        frame_counts = hf_inputs.get("video_num_frames")
        if frame_counts is not None:
            cfg["video_frames_pixels"] = MultiModalFieldConfig.flat_from_sizes(
                "video", frame_counts, dim=0)
            cfg["video_num_frames"] = MultiModalFieldConfig.batched("video")
            cfg["video_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
                "video", frame_counts, dim=0)
        return cfg

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Get prompt updates for image and video tokens."""
        tokens_per_tile = int(self.info.get_hf_config().num_query_tokens)

        updates: list[PromptUpdate] = []
        num_images = mm_items.get_count("image", strict=False)

        if num_images and num_images > 0:
            # Access tiles_per_image from MultiModalKwargsItems structure
            image_items = out_mm_kwargs.get("image", [])

            def _get_image_target(_: int):
                return [_IMAGE_PLACEHOLDER_TOKEN_ID]

            def _get_image_replacement(item_idx: int):
                # Get tiles for this specific image from processed outputs
                if image_items and item_idx < len(image_items):
                    tiles_field = image_items[item_idx].get("tiles_per_image")
                    if tiles_field and isinstance(tiles_field.data,
                                                  torch.Tensor):
                        n_tiles = int(tiles_field.data.item())
                    else:
                        n_tiles = 1
                else:
                    n_tiles = 1
                tokens_for_this_image = n_tiles * tokens_per_tile
                full = (
                    [_START_IMAGE_TOKEN] +
                    ([_IMAGE_PLACEHOLDER_TOKEN_ID] * tokens_for_this_image) +
                    [_END_IMAGE_TOKEN])
                return PromptUpdateDetails.select_token_id(
                    full, _IMAGE_PLACEHOLDER_TOKEN_ID)

            updates.append(
                PromptReplacement(
                    modality="image",
                    target=_get_image_target,
                    replacement=_get_image_replacement,
                ))

        # Video prompt updates: <video> +
        # (num_frames * tokens_per_tile) placeholders + </video>
        num_videos = mm_items.get_count("video", strict=False)
        if num_videos and num_videos > 0:
            video_items = out_mm_kwargs.get("video", [])
            default_frames = _get_default_video_num_frames(self.info.ctx)
            video_frame_counts = []
            for i in range(num_videos):
                if i < len(video_items):
                    data = video_items[i]
                    vnf = data.get("video_num_frames")
                    if vnf is not None and hasattr(vnf, "data"):
                        raw = vnf.data
                        if isinstance(raw, torch.Tensor):
                            video_frame_counts.append(int(raw.item()))
                        elif isinstance(raw, (int, float)):
                            video_frame_counts.append(int(raw))
                        else:
                            video_frame_counts.append(default_frames)
                    else:
                        video_frame_counts.append(default_frames)
                else:
                    video_frame_counts.append(default_frames)

            def _get_video_target(_: int):
                return [_IMAGE_PLACEHOLDER_TOKEN_ID]

            def _get_video_replacement(item_idx: int):
                num_frames = video_frame_counts[item_idx]
                tokens_for_video = num_frames * tokens_per_tile
                full = ([_START_VIDEO_TOKEN] +
                        ([_IMAGE_PLACEHOLDER_TOKEN_ID] * tokens_for_video) +
                        [_END_VIDEO_TOKEN])
                return PromptUpdateDetails.select_token_id(
                    full, _IMAGE_PLACEHOLDER_TOKEN_ID)

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=_get_video_target,
                    replacement=_get_video_replacement,
                ))

        return updates


# Register the processor after all classes are defined
YasaMMLMV2ForConditionalGeneration = MULTIMODAL_REGISTRY.register_processor(
    YasaMMLMV2MultiModalProcessor,
    info=YasaMMLMV2ProcessingInfo,
    dummy_inputs=YasaMMLMV2DummyInputsBuilder,
)(YasaMMLMV2ForConditionalGeneration)
