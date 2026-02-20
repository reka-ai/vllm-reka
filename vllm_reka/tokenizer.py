# ABOUTME: Tiktoken-based tokenizer with Yasa-style chat template and tool/multimodal support.
# ABOUTME: Extends HuggingFace PreTrainedTokenizer with custom special tokens and chat formatting.

import json
import logging
import os
from typing import Any, Callable, Optional, Union

import tiktoken
from transformers import PreTrainedTokenizer
from transformers.utils.chat_template_utils import render_jinja_template

logger = logging.getLogger(__name__)

TokenBuilder = Callable[[dict[str, Any]], list[str]]

TIKTOKEN_SPECIAL_TOKENS = {
    "<|endofchunk|>": 100277,
    "<REKA_IMG_TOKEN>": 100278,
    "<image>": 100279,
    "</image>": 100280,
    "<REKA_ADO_TOKEN>": 100281,
    "<audio>": 100282,
    "</audio>": 100283,
    "<video>": 100284,
    "</video>": 100285,
    "<transcript>": 100286,
    "</transcript>": 100287,
    "<ocr>": 100288,
    "</ocr>": 100289,
    "<sep>": 100290,
    "<tool>": 100291,
    "<tool_call>": 100292,
    "</tool_call>": 100293,
    "<tool_response>": 100294,
    "</tool_response>": 100295,
}

DEFAULT_CHAT_TEMPLATE = """
{%- macro render_content(content, num_img_tokens, num_video_frames) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- set ns = namespace(out="", prev_was_text=false) -%}
        {%- for item in content -%}
            {%- set item_type = item.get("type") -%}

            {%- if item_type == "text"
                or item.get("text") is not none -%}

                {%- set text = item.get("text", "") -%}
                {%- if text -%}
                    {%- if ns.prev_was_text -%}
                        {%- set ns.out = ns.out ~ " " -%}
                    {%- endif -%}
                    {%- set ns.out = ns.out ~ text -%}
                {%- endif -%}
                {%- set ns.prev_was_text = text != "" -%}

            {%- elif item_type in ["image", "image_url"]
                or item.get("image") is not none
                or item.get("image_url") is not none -%}

                {%- set ns.out =
                    ns.out
                    ~ "<image>"
                    ~ "<REKA_IMG_TOKEN>"
                    ~ "</image>"
                -%}
                {%- set ns.prev_was_text = false -%}

            {%- elif item_type in ["video", "video_url"]
                or item.get("video") is not none
                or item.get("video_url") is not none -%}

                {%- set ns.out =
                    ns.out
                    ~ "<video>"
                    ~ "<REKA_IMG_TOKEN>"
                    ~ "</video>"
                -%}
                {%- set ns.prev_was_text = false -%}
            {%- endif -%}
        {%- endfor -%}
        {{- ns.out -}}
    {%- endif -%}
{%- endmacro -%}


{%- set messages = messages or [] -%}
{%- set ns = namespace(out="", last_query_index=messages|length - 1) -%}

{%- set start_idx = 0 -%}
{%- set system_text = "" -%}

{%- if messages|length > 0
    and messages[0].get("role") in ["system", "developer"] -%}
    {%- set system_text = messages[0].get("content", "") -%}
    {%- set start_idx = 1 -%}
{%- endif -%}


{# ---- find last real user query ---- #}
{%- for msg in messages[::-1] -%}
    {%- set idx = messages|length - 1 - loop.index0 -%}
    {%- if msg.get("role") == "user" -%}
        {%- set content = msg.get("content", "") -%}
        {%- if not (
            content is string
            and content.startswith("<tool_response>")
            and content.endswith("</tool_response>")
        ) -%}
            {%- set ns.last_query_index = idx -%}
            {%- break -%}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- set last_query_index = ns.last_query_index -%}


{# ---- system / tools preamble ---- #}
{%- if tools or system_text -%}
    {%- set pre = namespace(text="") -%}

    {%- if system_text -%}
        {%- set pre.text = "system: " ~ system_text -%}
    {%- endif -%}

    {%- if tools -%}
        {%- if pre.text -%}
            {%- set pre.text = pre.text ~ "\\n\\n" -%}
        {%- else -%}
            {%- set pre.text = "system: " -%}
        {%- endif -%}

        {%- set pre.text = pre.text
            ~ "# Tools\\n\\n"
            ~ "You may call one or more functions.\\n\\n"
            ~ "Function signatures are in <tools></tools>:\\n"
            ~ "<tools>"
        -%}

        {%- for tool in tools -%}
            {%- set pre.text =
                pre.text ~ "\\n" ~ (tool | tojson(ensure_ascii=True))
            -%}
        {%- endfor -%}

        {%- set pre.text = pre.text
            ~ "\\n</tools>\\n\\n"
            ~ "Return calls inside <tool_call></tool_call>:\\n"
            ~ "<tool_call>\\n"
            ~ "{\\"name\\": <fn>, \\"arguments\\": <json>}\\n"
            ~ "</tool_call>"
        -%}
    {%- endif -%}

    {%- set ns.out = ns.out ~ pre.text ~ "\\n\\n<sep>" -%}
{%- endif -%}


{# ---- message loop ---- #}
{%- for idx in range(start_idx, messages|length) -%}
    {%- set message = messages[idx] -%}
    {%- set role = message.get("role") -%}
    {%- set content = message.get("content") -%}

    {%- if role == "user" -%}
        {%- set ns.out =
            ns.out
            ~ "human: "
            ~ render_content(
                content,
                num_img_tokens|default(64, true)|int,
                num_video_frames|default(6, true)|int
            )
            ~ "<sep>"
        -%}

    {%- elif role == "assistant" -%}
        {%- set ns.out = ns.out ~ "assistant: " -%}
        {%- set text = render_content(
            content,
            num_img_tokens|default(64, true)|int,
            num_video_frames|default(6, true)|int
        ) -%}
        {%- set ns.out = ns.out ~ text ~ "\\n\\n<sep>" -%}

    {%- elif role == "tool" -%}
        {%- set ns.out =
            ns.out
            ~ "human: <tool_response>\\n"
            ~ render_content(
                content,
                num_img_tokens|default(64, true)|int,
                num_video_frames|default(6, true)|int
            )
            ~ "\\n</tool_response><sep>"
        -%}
    {%- endif -%}
{%- endfor -%}


{%- if add_generation_prompt
    and (messages|length == 0
         or messages[-1].get("role") != "assistant") -%}

    {%- set ns.out = ns.out ~ "assistant:" -%}
{%- endif -%}

{{- ns.out -}}
"""


def normalize_message_content(content: Any) -> list[dict[str, Any]]:
    """Normalize content into a list of multimodal content dicts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    raise ValueError(
        "Message content must be a string or list of content items.")


def _build_tools_block(tools: list[dict[str, Any]]) -> str:
    """Build the tool instructions block for system prompts."""
    tools_block = ("# Tools\n\n"
                   "You may call one or more functions to "
                   "assist with the user query.\n\n"
                   "You are provided with function signatures "
                   "within <tools></tools> XML tags:\n"
                   "<tools>")
    for tool in tools:
        tools_block += f"\n{json.dumps(tool)}"
    tools_block += (
        "\n</tools>\n\n"
        "For each function call, return a json object "
        "with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>")
    return tools_block


def _render_chat_template(template: str, context: dict[str, Any]) -> str:
    """Render a Jinja chat template with the provided context."""
    rendered, _ = render_jinja_template(
        conversations=[context.get("messages", [])],
        tools=context.get("tools"),
        documents=context.get("documents"),
        chat_template=template,
        return_assistant_tokens_mask=False,
        continue_final_message=context.get("continue_final_message", False),
        add_generation_prompt=context.get("add_generation_prompt", False),
        **{
            key: value
            for key, value in context.items() if key not in {
                "messages",
                "tools",
                "documents",
                "continue_final_message",
                "add_generation_prompt",
            }
        },
    )
    return rendered[0]


def build_chat_prompt(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    continue_final_message: bool,
    tools: Optional[list[dict[str, Any]]],
    image_token_builder: TokenBuilder,
    video_token_builder: TokenBuilder,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Build the Yasa-style chat prompt with system/tools,
    user/assistant/tool turns."""
    if messages is None:
        messages = []
    elif not isinstance(messages, list):
        messages = messages.tolist() if hasattr(messages,
                                                "tolist") else list(messages)

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message at index {idx} must be a dict.")
        if "role" not in message:
            raise ValueError(f"Message at index {idx} is missing 'role'.")
        if "content" not in message and message.get("role") != "assistant":
            raise ValueError(f"Message at index {idx} is missing 'content'.")
    parts: list[str] = []

    def append_text(text: str, prev_was_text: bool,
                    prev_was_media: bool) -> None:
        if not text:
            return
        if prev_was_text:
            parts.append(" ")
        elif prev_was_media:
            pass
        parts.append(text)

    def extract_reasoning_and_content(
        content: Any,
        reasoning_value: Any,
    ) -> tuple[str, str]:
        content_text = ""
        reasoning_text = ""
        if isinstance(content, list):
            text_items = []
            for item in content:
                if item.get("type") != "text" and "text" not in item:
                    raise ValueError(
                        "Assistant message content must be text-only.")
                text = item.get("text", "")
                if text:
                    text_items.append(text)
            content_text = " ".join(text_items)
        elif content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            raise ValueError("Assistant message content must be "
                             "a string or list of content items.")
        if isinstance(reasoning_value, str):
            reasoning_text = reasoning_value
        elif "</think>" in content_text:
            before, after = content_text.split("</think>", 1)
            if "<think>" in before:
                reasoning_text = before.split("<think>")[-1]
            else:
                reasoning_text = before
            reasoning_text = reasoning_text.rstrip("\n").lstrip("\n")
            content_text = after.lstrip("\n")
        return content_text, reasoning_text

    def find_last_query_index(messages: list[dict[str, Any]]) -> int:
        last_query_index = len(messages) - 1
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if (isinstance(content, str)
                    and content.startswith("<tool_response>")
                    and content.endswith("</tool_response>")):
                continue
            last_query_index = idx
            break
        return last_query_index

    last_query_index = find_last_query_index(messages)
    start_idx = 0
    system_text = ""
    if len(messages) > 0 and messages[0].get("role") in ("system",
                                                         "developer"):
        system_text = messages[0].get("content", "")
        if not isinstance(system_text, str):
            raise ValueError("System message content must be a string.")
        start_idx = 1

    if tools or system_text:
        tools_block = _build_tools_block(tools) if tools else None
        if tools_block:
            if system_text:
                parts.append(f"system: {system_text}\n\n{tools_block}")
            else:
                parts.append(f"system: {tools_block}")
        elif system_text:
            parts.append(f"system: {system_text}")
        parts.append("\n\n")
        parts.append("<sep>")

    for idx in range(start_idx, len(messages)):
        message = messages[idx]
        role = message.get("role")
        if role == "user":
            content_items = normalize_message_content(message.get("content"))
            parts.append("human: ")
            prev_was_text = False
            prev_was_media = False
            for item in content_items:
                item_type = item.get("type")
                if item_type == "text" or "text" in item:
                    text = item.get("text", "")
                    append_text(
                        text,
                        prev_was_text=prev_was_text,
                        prev_was_media=prev_was_media,
                    )
                    prev_was_text = bool(text)
                    prev_was_media = False
                elif (item_type in ["image", "image_url"] or "image" in item
                      or "image_url" in item):
                    parts.extend(image_token_builder(item))
                    prev_was_text = False
                    prev_was_media = True
                elif item_type in ["video", "video_url"] or "video" in item:
                    parts.extend(video_token_builder(item))
                    prev_was_text = False
                    prev_was_media = True
                else:
                    raise ValueError(f"Unsupported content type: {item_type}. "
                                     "Only 'text', 'image', 'image_url', "
                                     "'video', and 'video_url' are supported.")
            parts.append("<sep>")
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            if (tool_calls is not None and hasattr(tool_calls, "tolist")
                    and not isinstance(tool_calls, (str, bytes, dict))):
                tool_calls = tool_calls.tolist()
            content = message.get("content")
            if content is None and tool_calls:
                content = ""
            content_text, reasoning_text = extract_reasoning_and_content(
                content, message.get("reasoning_content"))
            content_items = normalize_message_content(content_text)
            if any(item.get("type") != "text" for item in content_items):
                raise ValueError(
                    "Assistant message content must be text-only.")
            parts.append("assistant: ")
            include_thinking = (enable_thinking is True
                                and idx > last_query_index and
                                (idx == len(messages) - 1 or reasoning_text))
            if include_thinking:
                parts.append("<think>\n")
                parts.append(reasoning_text.strip("\n"))
                parts.append("\n</think>\n\n")
            assistant_has_text = False
            for item in content_items:
                text = item.get("text", "")
                if text:
                    if assistant_has_text:
                        parts.append(" ")
                    parts.append(text)
                    assistant_has_text = True
            if tool_calls:
                tool_call_texts = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        raise ValueError(
                            "Tool call entries must be JSON objects.")
                    if tool_call.get("function"):
                        if not isinstance(tool_call["function"], dict):
                            raise ValueError(
                                "Tool call 'function' must be a JSON object.")
                        tool_call = tool_call["function"]
                    arguments = tool_call.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments_json = arguments
                    elif isinstance(arguments, dict):
                        arguments_json = json.dumps(arguments)
                    else:
                        tool_name = tool_call.get("name", "unknown")
                        raise ValueError("Tool call arguments must be a JSON "
                                         "object or string; "
                                         f"got {type(arguments).__name__} for "
                                         f"tool '{tool_name}'.")
                    tool_name = tool_call.get("name", "unknown")
                    tool_call_texts.append("<tool_call>\n"
                                           f'{{"name": "{tool_name}", '
                                           f'"arguments": {arguments_json}}}'
                                           "</tool_call>")
                if assistant_has_text and parts and not parts[-1].endswith(
                        "\n"):
                    parts.append("\n")
                for tool_call_text in tool_call_texts:
                    parts.append(tool_call_text)
                assistant_has_text = True
            if not (continue_final_message and idx == len(messages) - 1):
                parts.append("\n\n")
                parts.append("<sep>")
        elif role == "tool":
            content_items = normalize_message_content(message.get("content"))
            if idx == start_idx or messages[idx - 1].get("role") != "tool":
                parts.append("human: ")
            response_parts = []
            for item in content_items:
                item_type = item.get("type")
                if item_type != "text":
                    raise ValueError(
                        "Unsupported content type: "
                        f"{item_type}. Only text tool responses are supported."
                    )
                text = item.get("text", "")
                if text:
                    response_parts.append(text)
            response_text = " ".join(response_parts)
            append_text(
                f"<tool_response>\n{response_text}\n</tool_response>",
                prev_was_text=False,
                prev_was_media=False,
            )
            if idx == len(messages) - 1 or messages[idx +
                                                    1].get("role") != "tool":
                parts.append("<sep>")
        elif role in ("system", "developer"):
            raise ValueError("System message must be the first message.")
        else:
            raise ValueError(f"Unsupported message role: {role}. "
                             "Only 'system', 'developer', 'user', "
                             "'assistant', and 'tool' roles are supported.")

    if add_generation_prompt and (not messages
                                  or messages[-1].get("role") != "assistant"):
        if enable_thinking is True:
            parts.append("assistant: <think>\n")
        else:
            parts.append("assistant:")

    return "".join([p for p in parts if p != ""])


class YasaTokenizer(PreTrainedTokenizer):
    """Tokenizer with Yasa-style chat template and tool/multimodal support."""

    vocab_files_names = {
        "tiktoken_special_tokens": "tiktoken_special_tokens.json"
    }
    pretrained_vocab_files_map: dict[str, str] = {}
    max_model_input_sizes = {"yasa-model": 2048}
    model_input_names = ["input_ids", "attention_mask"]
    strip_leading_whitespace = True

    def __init__(self, tiktoken_special_tokens=None, **kwargs):
        tiktoken.registry._find_constructors()
        tiktoken_model_name = kwargs.get("tiktoken_model_name")
        if not tiktoken_model_name:
            raise ValueError("'tiktoken_model_name' is required "
                             "to initialize YasaTokenizer.")
        kwargs.setdefault("model_max_length", 8192)
        # tiktoken.registry is a module; stubs may type it as Optional
        base_kwargs = (
            tiktoken.registry.
            ENCODING_CONSTRUCTORS[  # type: ignore[union-attr]
                tiktoken_model_name]())

        if isinstance(tiktoken_special_tokens, str):
            with open(tiktoken_special_tokens) as f:
                special_tokens = json.load(f)
        elif isinstance(tiktoken_special_tokens, dict):
            special_tokens = tiktoken_special_tokens
        else:
            special_tokens = TIKTOKEN_SPECIAL_TOKENS

        special_tokens = dict(special_tokens)
        used_token_ids = set(base_kwargs["special_tokens"].values()).union(
            set(base_kwargs["mergeable_ranks"].values()))
        collision = used_token_ids.intersection(special_tokens.values())
        if collision:
            raise ValueError(
                f"special token overlapping with tiktoken builtin {collision}")

        self.tiktoken_special_tokens = dict(special_tokens)
        for i in range(100256, 100352):
            if i not in special_tokens.values():
                special_tokens[f"<|special_token_{i}|>"] = i

        base_kwargs["special_tokens"].update({
            token: token_id
            for token, token_id in special_tokens.items()
        })
        self.tiktoken = tiktoken.Encoding(**base_kwargs)

        kwargs.pop("add_prefix_space", None)
        super().__init__(add_prefix_space=False, **kwargs)
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.utf8_decoding_strategy = kwargs.get("utf8_decoding_strategy",
                                                 "replace")
        self.allowed_special_tokens = set(special_tokens.keys())
        self.clean_up_tokenization_spaces = False

        chat_template_value: Optional[str] = None
        chat_template_override = kwargs.get("chat_template")
        if isinstance(chat_template_override, str) and chat_template_override:
            if os.path.isfile(chat_template_override):
                with open(chat_template_override, encoding="utf-8") as handle:
                    chat_template_value = handle.read()
            else:
                chat_template_value = chat_template_override
        else:
            model_path = kwargs.get("name_or_path") or kwargs.get("_name_or_path")
            if isinstance(model_path, str):
                chat_template_path = os.path.join(model_path, "chat_template.jinja")
                if os.path.isfile(chat_template_path):
                    with open(chat_template_path, encoding="utf-8") as handle:
                        chat_template_value = handle.read()

        self.chat_template = chat_template_value or DEFAULT_CHAT_TEMPLATE

    @property
    def max_token_id(self) -> int:
        """Get the maximum token ID in the vocabulary."""
        return self.tiktoken.max_token_value

    def apply_chat_template(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        num_img_tokens: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the chat template to the messages.

        Args:
            messages: list of messages in the conversation.
            chat_template: Optional chat template to use.
                If None, uses the default.
            tokenize: Whether to tokenize the formatted prompt.
            add_generation_prompt: Whether to add the generation prompt.
            continue_final_message: Whether to continue the final message.
            return_tensors: Tensor type for outputs (e.g. "pt").
            return_dict: Whether to return a dict payload.
            tools: Optional list of tool specifications.
            num_img_tokens: Optional image token repeat
                count per image placeholder.
            num_video_frames: Optional frame count
               for video placeholder repetition.
            enable_thinking: When True, insert <think>
                blocks for assistant turns
                that contain reasoning content and for the generation prompt.
            **kwargs:
                Additional arguments to pass to the template.

        Returns:
            Prompt string if tokenize is False. Otherwise token IDs or a dict
            payload when return_dict is True.
        """
        if messages is None:
            if "conversation" in kwargs:
                messages = kwargs.pop("conversation")
            else:
                messages = []
        elif not isinstance(messages, list):
            messages = (messages.tolist()
                        if hasattr(messages, "tolist") else list(messages))
        # Ensure list for type checker (messages may have been set from kwargs)
        messages = list(messages) if messages is not None else []

        if continue_final_message and (not messages
                                       or messages[-1]["role"] != "assistant"):
            raise ValueError("'continue_final_message' requires "
                             "the last message to be from the assistant.")

        if continue_final_message and add_generation_prompt:
            logger.warning("'add_generation_prompt' is ignored when "
                           "'continue_final_message' is set.")

        num_img_tokens = 64 if num_img_tokens is None else num_img_tokens
        num_video_frames = 6 if num_video_frames is None else num_video_frames

        def image_builder(_: dict[str, Any]) -> list[str]:
            return ["<image>"
                    ] + ["<REKA_IMG_TOKEN>"] * num_img_tokens + ["</image>"]

        def video_builder(_: dict[str, Any]) -> list[str]:
            return (["<video>"] + ["<REKA_IMG_TOKEN>"] + ["</video>"])

        template_source = chat_template or getattr(self, "chat_template", None)
        if template_source:
            if os.path.isfile(template_source):
                with open(template_source, encoding="utf-8") as handle:
                    template_source = handle.read()
            prompt = _render_chat_template(
                template_source,
                {
                    "messages": messages,
                    "add_generation_prompt": add_generation_prompt,
                    "continue_final_message": continue_final_message,
                    "tools": tools,
                    "enable_thinking": enable_thinking,
                    "num_img_tokens": num_img_tokens,
                    "num_video_frames": num_video_frames,
                },
            )
        else:
            prompt = build_chat_prompt(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tools=tools,
                image_token_builder=image_builder,
                video_token_builder=video_builder,
                enable_thinking=enable_thinking,
            )
        if not tokenize:
            return prompt

        text_input = [prompt] if return_tensors is not None else prompt
        encoded = self(
            text_input,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **kwargs,
        )
        if return_dict:
            return encoded
        return encoded["input_ids"]

    def build_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool,
        tools: Optional[list[dict[str, Any]]],
        image_token_builder: TokenBuilder,
        video_token_builder: TokenBuilder,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Build a Yasa prompt using tokenizer-shared formatting helpers."""
        return build_chat_prompt(
            messages,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            image_token_builder=image_token_builder,
            video_token_builder=video_token_builder,
            enable_thinking=enable_thinking,
        )

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None) -> list[int]:
        return [0] * len(token_ids_0)

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.tiktoken.max_token_value + 1

    def get_vocab(self) -> dict[bytes, int]:
        ret = {}
        for i in range(self.tiktoken.max_token_value + 1):
            try:
                ret[self.tiktoken.decode_single_token_bytes(i)] = i
            except Exception as e:
                raise ValueError(f"Error decoding token {i}: {e}") from e
        return ret

    def _tokenize(self, text: str, **kwargs: Any) -> list[bytes]:
        """Convert a string into a sequence of tokens (bytes)."""
        return [
            self._convert_id_to_token(t)
            for t in self.tiktoken.encode(text, allowed_special="all")
        ]

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        return self.tiktoken.encode_single_token(token)

    def _convert_id_to_token(self, index: int) -> bytes:
        return self.tiktoken.decode_single_token_bytes(index)

    def convert_tokens_to_string(self, tokens: list[Union[bytes, str]]) -> str:
        # Convert all tokens to bytes first
        bytes_tokens = [
            t.encode('utf-8') if isinstance(t, str) else t for t in tokens
        ]
        return b"".join(bytes_tokens).decode(
            "utf8", errors=self.utf8_decoding_strategy)

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> tuple[str]:
        """Save the tokenizer vocabulary to files."""
        os.makedirs(save_directory, exist_ok=True)
        filename = "tiktoken_special_tokens.json"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        save_path = os.path.join(save_directory, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.tiktoken_special_tokens, f, indent=4)
        return (save_path, )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        return super().save_pretrained(save_directory, legacy_format,
                                       filename_prefix, push_to_hub, **kwargs)


Yasa2Tokenizer = YasaTokenizer
