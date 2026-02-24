# ABOUTME: YasaMMLMForConditionalGeneration multimodal model with SigLIP vision encoder.
# ABOUTME: Supports image and video inputs via adaptive pooling and language projection.
import copy
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchFeature, PreTrainedTokenizer

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix)

from vllm_reka.multimodal_utils import (DEFAULT_VIDEO_NUM_FRAMES,
                                        ImageProcessor, VideoProcessor,
                                        merge_multimodal_embeddings,
                                        _IMAGE_PLACEHOLDER_TOKEN_ID,
                                        _START_IMAGE_TOKEN, _END_IMAGE_TOKEN,
                                        _START_VIDEO_TOKEN, _END_VIDEO_TOKEN,
                                        _get_default_video_num_frames)



class YasaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


class YasaImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


YasaImageInputs = Union[YasaImagePixelInputs, YasaImageEmbeddingInputs]


class YasaVideoPixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


class YasaVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    data: torch.Tensor
    metadata: dict[str, Any] | None


YasaVideoInputs = Union[YasaVideoPixelInputs, YasaVideoEmbeddingInputs]


class YasaMMLMForConditionalGeneration(nn.Module, SupportsMultiModal,
                                       SupportsPP):
    """YASA MMLM model for conditional generation with multimodal support."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<REKA_IMG_TOKEN>"
        if modality.startswith("video"):
            return "<video></video>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.vision_pooling = config.vision_pooling
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(
            int(config.num_query_tokens**0.5))
        self.language_projection = nn.Sequential(
            nn.Linear(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
            ),
        )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> YasaVideoInputs | None:
        """
        Parse video input - mirrors _parse_and_validate_image_input.
        Logically they go through the same modeling pipeline.
        video_frame -> image patch
        num frames per video -> num patches per image
        """
        video_frames_pixels = kwargs.pop("video_frames_pixels", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_num_frames = kwargs.pop("video_num_frames", None)
        if video_frames_pixels is None and video_embeds is None:
            return None
        if video_frames_pixels is not None:
            # Flatten pixels to 4D [N, C, H, W]
            video_num_frames = cast(torch.Tensor,
                                    video_num_frames).flatten().tolist()
            video_frames_pixels = cast(torch.Tensor, video_frames_pixels)
            if video_frames_pixels.ndim != 4:
                B, N, C, H, W = video_frames_pixels.shape
                video_frames_pixels = video_frames_pixels.reshape(
                    B * N, C, H, W)
            assert (video_frames_pixels.ndim == 4), (
                f"Expected video_pixel_values to be a 4D tensor, "
                f"got {video_frames_pixels.ndim}D tensor")
            # Convert frames count to list
            assert (sum(video_num_frames) == video_frames_pixels.shape[0]), (
                f"Expected total number of frames and video frame pixels to "
                f"be the same, got {sum(video_num_frames)} and"
                f" {video_frames_pixels.shape[0]}.")
            metadata = {"frames_per_video": video_num_frames}
            return YasaVideoPixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(video_frames_pixels),
                metadata=metadata,
            )

        if video_embeds is not None:
            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            # video_embeds expected shape: [total_frames, embed_dim] or similar
            frames_per_video = cast(torch.Tensor, video_num_frames).tolist()
            return YasaVideoEmbeddingInputs(
                type="video_embeds",
                data=video_embeds,
                metadata={"frames_per_video": frames_per_video},
            )
        raise ValueError(
            "Either video_frames_pixels or video_embeds must be provided.")

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> YasaImageInputs | None:
        """
        Parse image input.
        Expects pixel_values as [total_patches, C, H, W] with
        patches_per_image in metadata.
        """
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        patches_per_image = kwargs.pop("patches_per_image", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            patches_per_image = cast(torch.Tensor,
                                     patches_per_image).flatten().tolist()
            pixel_values = cast(torch.Tensor, pixel_values)
            if pixel_values.ndim != 4:
                B, N, C, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B * N, C, H, W)
            assert (pixel_values.ndim == 4), (
                f"Expected pixel_values to be a 4D tensor [N, C, H, W]"
                f", got {pixel_values.ndim}D tensor")
            assert (sum(patches_per_image) == pixel_values.shape[0]), (
                f"Expected sum(patches_per_image)={sum(patches_per_image)} "
                f"to equal total patches={pixel_values.shape[0]}")
            metadata = {"patches_per_image": patches_per_image}
            return YasaImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
                metadata=metadata,
            )
        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            patches_per_image = cast(
                torch.Tensor,
                patches_per_image).tolist() if patches_per_image else []
            return YasaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
                metadata={"patches_per_image": patches_per_image},
            )
        raise ValueError(
            "Either pixel_values or image_embeds must be provided.")

    def _image_pixels_to_features(self, vision_model: SiglipVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_model(pixel_values)
        return image_features

    def _process_image_pixels(self,
                              inputs: YasaImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["data"]
        # Time vision model inference
        vision_features = self._image_pixels_to_features(
            self.vision_model, pixel_values)
        # Additional validation of vision features
        if len(vision_features.shape) != 3:
            raise ValueError(f"Vision model output should be 3D, "
                             f"got shape: {vision_features.shape}")
        return vision_features

    def _process_image_input(self,
                             image_input: YasaImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        # Process all patches through vision model
        image_embeds = self._process_image_pixels(
            image_input)  # [total_patches, seq_len, vision_hidden_size]
        total_patches, seq_len, vision_hidden_size = image_embeds.shape

        # Infer spatial dims (seq_len = h * w)
        height = width = int(seq_len**0.5)
        if height * width != seq_len:  # handle imperfect squares robustly
            height = width = round(seq_len**0.5)
        image_embeds = image_embeds.permute(0, 2,
                                            1).reshape(total_patches,
                                                       vision_hidden_size,
                                                       height, width)

        # Adaptive 2D pooling:
        # [total_patches, vision_hidden_size, pooled_h, pooled_w]
        image_embeds = self.adaptive_pooling(image_embeds)
        pooled_h, pooled_w = image_embeds.shape[2:]
        pooled_seq_len = pooled_h * pooled_w

        # Check token count alignment
        expected_tokens = getattr(self.config, "num_query_tokens",
                                  pooled_seq_len)
        if pooled_seq_len != expected_tokens:
            raise ValueError(
                f"YASA: pooled_seq_len={pooled_seq_len} "
                f"must equal num_query_tokens={expected_tokens}. "
                "Adjust adaptive pooling or config.num_query_tokens.")
        # Flatten spatial to sequence and project
        image_embeds = image_embeds.flatten(2).permute(0, 2, 1).contiguous()
        # [total_patches, pooled_seq_len, hidden_size]
        vision_embeds = self.language_projection(image_embeds)
        return vision_embeds.reshape(-1, vision_embeds.size(-1))

    def _get_image_multimodal_embeddings(
            self, **kwargs) -> list[torch.Tensor] | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        metadata = image_input.get("metadata", {})
        patches_per_image = metadata.get("patches_per_image", [])
        tokens_per_patch = getattr(self.config, "num_query_tokens", 441)
        if not patches_per_image:
            raise ValueError("Missing `patches_per_image` in image metadata.")
        # Compute token counts per image
        tokens_per_image = [p * tokens_per_patch for p in patches_per_image]
        # Split efficiently with torch.split
        embeddings_list = list(
            torch.split(vision_embeddings, tokens_per_image, dim=0))
        return embeddings_list

    def _get_video_multimodal_embeddings(
            self, **kwargs) -> list[torch.Tensor] | None:
        """Get video embeddings - mirrors _get_image_multimodal_embeddings."""
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is None:
            return None

        # Handle pre-computed embeddings
        if video_input["type"] == "video_embeds":
            # Already embedded, just split by video
            vision_embeddings = video_input["data"]
        else:
            # Reuse same vision pipeline (video frames = image patches)
            vision_embeddings = self._process_image_input(video_input)

        metadata = video_input.get("metadata", {})
        frames_per_video = metadata.get("frames_per_video", [])
        tokens_per_frame = getattr(self.config, "num_query_tokens", 441)
        if not frames_per_video:
            raise ValueError("Missing `frames_per_video` in video metadata.")
        # Compute token counts per video (each frame = 1 patch)
        tokens_per_video = [f * tokens_per_frame for f in frames_per_video]
        # Split efficiently with torch.split
        embeddings_list = list(
            torch.split(vision_embeddings, tokens_per_video, dim=0))
        return embeddings_list

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor]:
        """Get embeddings for all multimodal inputs.
        Both video frames and image patches are processed through the same
        vision encoder pipeline.
        """
        all_embeddings: list[torch.Tensor] = []
        # Process images
        image_embs = self._get_image_multimodal_embeddings(**kwargs)
        if image_embs:
            all_embeddings.extend(image_embs)
        # Process videos (parallel structure)
        video_embs = self._get_video_multimodal_embeddings(**kwargs)
        if video_embs:
            all_embeddings.extend(video_embs)

        return all_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        if isinstance(multimodal_embeddings,
                      (list, tuple)) and len(multimodal_embeddings) > 0:
            mm_concat = torch.cat(list(multimodal_embeddings), dim=0)
            placeholder_count = (
                input_ids == _IMAGE_PLACEHOLDER_TOKEN_ID).sum().item()
            if mm_concat.shape[0] == placeholder_count:
                return merge_multimodal_embeddings(
                    input_ids, inputs_embeds, mm_concat,
                    _IMAGE_PLACEHOLDER_TOKEN_ID)
            padded_embeddings = self._pad_and_validate_image_embeddings(
                input_ids=input_ids,
                multimodal_embeddings=multimodal_embeddings)
            return self._merge_padded_image_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                padded_embeddings=padded_embeddings)

        return merge_multimodal_embeddings(input_ids, inputs_embeds,
                                           multimodal_embeddings,
                                           _IMAGE_PLACEHOLDER_TOKEN_ID)

    def _pad_and_validate_image_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
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
                f"YASA: produced image tokens={produced_count} != "
                f"reserved placeholders={expected_count}. "
                f"Check patches_per_image and num_query_tokens."
                f"Produced count: {produced_count}, "
                f"Expected count: {expected_count}, "
                f"counts: {[p.shape[0] for p in padded_list]}"
                f"input_ids: {input_ids.shape}")

        return tuple(padded_list)

    def _merge_padded_image_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        padded_embeddings: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            padded_embeddings,
            [
                _START_IMAGE_TOKEN, _IMAGE_PLACEHOLDER_TOKEN_ID,
                _END_IMAGE_TOKEN
            ],
        )

    def _merge_placeholder_only(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            _IMAGE_PLACEHOLDER_TOKEN_ID,
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
        """Run forward pass for Yasa MMLM.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"Question: What's the content of the image? Answer:"`.

        Tokenizer outputs:
        `[2, 45641, 35, 653, 18, 5, 1383, 9, 5, 2274, 116, 31652, 35]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        dummy tokens (denoted as `50265`), resulting in:
        `[50265, ..., 50265, 2, 45641, 35, ..., 31652, 35]`.

        We insert 32 tokens since it corresponds to the number of query
        embeddings outputted by the Q-Former and inputted to the language model.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each input image.
        See also:
            :class:`YasaImageInputs`
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.embed_multimodal(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "audio_model.",
                "audio_language_projection.",
                "audio_ln.",
                "audio_tokenizer.",
            ],
        )
        return loader.load_weights(weights)


@lru_cache(maxsize=1)
def _get_tokenizer_without_image_pad(
    tokenizer: PreTrainedTokenizer, ) -> PreTrainedTokenizer:
    """Get a copy of the tokenizer without image pad handling."""
    new_tokenizer = copy.deepcopy(tokenizer)
    return new_tokenizer


class YasaProcessingInfo(BaseProcessingInfo):
    """Processing info for YASA MMLM model."""

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer)
        return tokenizer

    def get_image_processor(self) -> ImageProcessor:
        """Get the YASA image processor."""
        return ImageProcessor(self.get_hf_config())

    def get_video_processor(self) -> VideoProcessor:
        """Get the YASA video processor."""
        return VideoProcessor(self.get_hf_config())

    def get_audio_processor(self):
        # Try to get an HF processor suitable for audio
        return self.ctx.get_hf_processor()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_yasa_num_image_tokens(),
            "video": self.get_max_yasa_num_video_tokens(),
        }

    def get_max_yasa_num_image_tokens(self) -> int:
        """Get number of tokens per image."""
        image_processor = self.get_image_processor()
        tokens = image_processor.get_max_yasa_image_tokens()
        return tokens

    def get_max_yasa_num_video_tokens(self,
                                      max_frames: int | None = None) -> int:
        """Get maximum number of tokens for a video.

        Each frame = 1 patch = num_query_tokens tokens (no patching for video).
        Additional tokens account for timestamps and frame markers.
        """
        if max_frames is None:
            max_frames = _get_default_video_num_frames(self.ctx)
        video_processor = self.get_video_processor()
        base_tokens = video_processor.get_max_video_tokens(max_frames)
        # Account for timestamp tokens
        # (~6 per frame: "t=", digits, "s", newline)
        # and video start/end tokens (2)
        timestamp_overhead = max_frames * 6 + 2
        return base_tokens + timestamp_overhead


class YasaDummyInputsBuilder(BaseDummyInputsBuilder[YasaProcessingInfo]):
    """Dummy inputs builder for YASA MMLM model."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Build the text input corresponding to mm_counts."""
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        parts = []
        parts.extend(["<REKA_IMG_TOKEN>"] * num_images)
        parts.extend(["<video></video>"] * num_videos)
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
            dummy_image = self.get_max_yasa_dummy_image()
            mm_data["image"] = [dummy_image] * num_images

        num_videos = mm_counts.get("video", 0)
        if num_videos > 0:
            dummy_video = self.get_max_yasa_dummy_video()
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
        # image and video use the same placeholder token
        if num_images > 0:
            dummy_image = self.get_max_yasa_dummy_image()
            mm_data["image"] = [dummy_image] * num_images
            parts.extend(["<REKA_IMG_TOKEN>"] * num_images)

        if num_videos > 0:
            dummy_video = self.get_max_yasa_dummy_video()
            mm_data["video"] = [dummy_video] * num_videos
            parts.extend(["<video></video>"] * num_videos)

        prompt_text = " ".join(parts)
        return ProcessorInputs(
            prompt=prompt_text,
            mm_data=mm_data,
        )

    def get_max_yasa_dummy_image(self) -> Image.Image:
        hf_config = self.info.ctx.get_hf_config()
        image_processor = ImageProcessor(hf_config)
        dummy_image = image_processor.get_max_dummy_image()
        return dummy_image

    def get_max_yasa_dummy_video(self,
                                 num_frames: int | None = None
                                 ) -> tuple[np.ndarray, dict]:
        """Create dummy video frames with metadata for profiling."""
        if num_frames is None:
            num_frames = _get_default_video_num_frames(self.info.ctx)
        hf_config = self.info.ctx.get_hf_config()
        video_processor = VideoProcessor(hf_config)
        dummy_frames = video_processor.get_max_dummy_video(num_frames)
        # Create dummy metadata with timestamps
        dummy_metadata = {
            "fps": 1.0,
            "duration": float(num_frames),
            "timestamps": list(range(num_frames)),
            "frames_indices": list(range(num_frames)),
        }
        return (dummy_frames, dummy_metadata)


class YasaMultiModalProcessor(BaseMultiModalProcessor[YasaProcessingInfo]):
    """Multimodal processor for YASA MMLM model."""

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
        text_inputs = tokenizer(prompt,
                                return_tensors="pt",
                                padding=False,
                                **tok_kwargs)
        result = {**text_inputs}
        # get images
        images = mm_data.get("images", [])
        if images:
            image_inputs = image_processor.batch_preprocess(images)
            result["pixel_values"] = image_inputs["pixel_values"]
            result["patches_per_image"] = image_inputs["patches_per_image"]
        # get videos
        videos = mm_data.get("videos", [])
        if videos:
            video_frames_pixels = []
            frame_counts = []
            all_timestamps = []

            for video in videos:
                frames, metadata = video
                # Process frames like image patches
                pil_frames = [Image.fromarray(f) for f in frames]
                target_res = self.info.get_hf_config().vision_config.image_size
                # Resize frames to target resolution
                resized = [
                    f.resize((target_res, target_res),
                             Image.Resampling.BICUBIC) for f in pil_frames
                ]
                # run through vision preprocessor
                frame_pixels = video_processor.vision_preprocessor.preprocess(
                    resized, return_tensors="pt")["pixel_values"]

                video_frames_pixels.append(frame_pixels)
                frame_counts.append(len(frames))
                # Collect timestamps (flattened)
                timestamps = metadata.get("timestamps",
                                          list(range(len(frames))))
                all_timestamps.extend(timestamps)
            # Store video data separately from images
            result["video_frames_pixels"] = torch.cat(video_frames_pixels,
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

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        cfg = {}
        # Image fields - route to "image" modality
        image_counts = hf_inputs.get("patches_per_image")
        if image_counts is not None:
            cfg["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", image_counts, dim=0)
            cfg["patches_per_image"] = MultiModalFieldConfig.batched("image")
        # Video fields - route to "video" modality
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
        out_mm_kwargs: Mapping[str, object],
    ) -> Sequence[PromptUpdate]:
        # One dynamic update per modality; framework resolves it per item.
        tokens_per_patch = int(self.info.get_hf_config().num_query_tokens)
        assert tokens_per_patch > 0
        tokenizer = self.info.get_tokenizer()
        updates: list[PromptUpdate] = []
        # Image prompt updates
        num_images = mm_items.get_count("image", strict=False)
        if num_images and num_images > 0:
            patches_per_image = [
                data.get("patches_per_image").data
                for data in out_mm_kwargs['image']
            ]

            def _get_image_target(_: int):
                return [_IMAGE_PLACEHOLDER_TOKEN_ID]

            def _get_image_replacement(item_idx: int):
                num_patches = patches_per_image[item_idx]
                n_tokens = num_patches * tokens_per_patch
                full = ([_START_IMAGE_TOKEN] +
                        ([_IMAGE_PLACEHOLDER_TOKEN_ID] * n_tokens) +
                        [_END_IMAGE_TOKEN])
                return PromptUpdateDetails.select_token_id(
                    full, _IMAGE_PLACEHOLDER_TOKEN_ID)

            updates.append(
                PromptReplacement(modality="image",
                                  target=_get_image_target,
                                  replacement=_get_image_replacement))

        # Video prompt updates
        num_videos = mm_items.get_count("video", strict=False)
        if num_videos and num_videos > 0:
            # Get frame counts - videos have their own frame count tracking
            video_frame_counts = [
                data.get("video_num_frames").data
                for data in out_mm_kwargs['video']
            ]

            # Get timestamps (flattened, already sliced by framework)
            video_timestamps = [
                data.get("video_timestamps").data
                for data in out_mm_kwargs['video']
            ]

            def _get_video_target(_: int):
                return [_START_VIDEO_TOKEN, _END_VIDEO_TOKEN]

            def _get_video_replacement(item_idx: int):
                num_frames = int(video_frame_counts[item_idx])
                timestamps = video_timestamps[item_idx].tolist()
                # Build:
                # <START_VIDEO>t=0s<START_IMAGE><placeholders><END_IMAGE>\n...
                result_tokens = [_START_VIDEO_TOKEN]
                for frame_idx in range(num_frames):
                    ts = (timestamps[frame_idx]
                          if frame_idx < len(timestamps) else frame_idx)
                    # Add timestamp tokens: "t=Xs"
                    ts_text = f"t={int(ts)}s"
                    ts_tokens = tokenizer.encode(ts_text,
                                                 add_special_tokens=False)
                    result_tokens.extend(ts_tokens)

                    # Add image tokens for this
                    # frame (1 patch per frame, no patching)
                    n_frame_tokens = tokens_per_patch
                    result_tokens.append(_START_IMAGE_TOKEN)
                    result_tokens.extend([_IMAGE_PLACEHOLDER_TOKEN_ID] *
                                         n_frame_tokens)
                    result_tokens.append(_END_IMAGE_TOKEN)
                    # Add newline between frames
                    newline_tokens = tokenizer.encode("\n",
                                                      add_special_tokens=False)
                    result_tokens.extend(newline_tokens)
                result_tokens.append(_END_VIDEO_TOKEN)
                return PromptUpdateDetails.select_token_id(
                    result_tokens, _IMAGE_PLACEHOLDER_TOKEN_ID)

            updates.append(
                PromptReplacement(modality="video",
                                  target=_get_video_target,
                                  replacement=_get_video_replacement))
        return updates


# Register the processor after all classes are defined
YasaMMLMForConditionalGeneration = MULTIMODAL_REGISTRY.register_processor(
    YasaMultiModalProcessor,
    info=YasaProcessingInfo,
    dummy_inputs=YasaDummyInputsBuilder,
)(YasaMMLMForConditionalGeneration)
