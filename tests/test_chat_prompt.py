# ABOUTME: Tests for build_chat_prompt and its helper functions.
# ABOUTME: Covers system messages, tool calls, thinking blocks, multi-turn, and error cases.

import json

import pytest

from vllm_reka.tokenizer import (
    build_chat_prompt,
    normalize_message_content,
    _build_tools_block,
)


def _image_builder(_):
    return ["<REKA_IMG_TOKEN>"]


def _video_builder(_):
    return ["<video>", "</video>"]


def _build(messages, *, add_gen=True, tools=None, thinking=None,
           continue_final=False):
    """Shorthand for calling build_chat_prompt with default builders."""
    return build_chat_prompt(
        messages,
        add_generation_prompt=add_gen,
        continue_final_message=continue_final,
        tools=tools,
        image_token_builder=_image_builder,
        video_token_builder=_video_builder,
        enable_thinking=thinking,
    )


# ── normalize_message_content ─────────────────────────────────────────


class TestNormalizeMessageContent:

    def test_string_input(self):
        result = normalize_message_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_list_passthrough(self):
        items = [{"type": "text", "text": "hi"}]
        result = normalize_message_content(items)
        assert result is items

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="string or list"):
            normalize_message_content(42)


# ── _build_tools_block ────────────────────────────────────────────────


class TestBuildToolsBlock:

    def test_single_tool(self):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        block = _build_tools_block(tools)
        assert "<tools>" in block
        assert "</tools>" in block
        assert "get_weather" in block
        assert "<tool_call>" in block

    def test_multiple_tools(self):
        tools = [
            {"function": {"name": "a"}},
            {"function": {"name": "b"}},
        ]
        block = _build_tools_block(tools)
        assert block.count(json.dumps(tools[0])) == 1
        assert block.count(json.dumps(tools[1])) == 1


# ── Basic message formatting ─────────────────────────────────────────


class TestBasicMessages:

    def test_simple_user_message(self):
        result = _build([{"role": "user", "content": "hello"}])
        assert result == "human: hello<sep>assistant:"

    def test_user_assistant_turn(self):
        result = _build([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ], add_gen=False)
        assert result == "human: hi<sep>assistant: hey\n\n<sep>"

    def test_multi_turn(self):
        result = _build([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ])
        assert "human: q1<sep>" in result
        assert "assistant: a1\n\n<sep>" in result
        assert "human: q2<sep>" in result
        assert result.endswith("assistant:")

    def test_no_generation_prompt(self):
        result = _build(
            [{"role": "user", "content": "hi"}], add_gen=False)
        assert result == "human: hi<sep>"
        assert "assistant:" not in result

    def test_empty_messages(self):
        result = _build([], add_gen=True)
        assert result == "assistant:"

    def test_empty_messages_no_gen(self):
        result = _build([], add_gen=False)
        assert result == ""


# ── System messages ───────────────────────────────────────────────────


class TestSystemMessages:

    def test_system_message(self):
        result = _build([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ])
        assert result.startswith("system: You are helpful.\n\n<sep>")
        assert "human: hi<sep>" in result

    def test_developer_role(self):
        result = _build([
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "hi"},
        ])
        assert result.startswith("system: Be concise.\n\n<sep>")

    def test_system_not_first_raises(self):
        with pytest.raises(ValueError, match="first message"):
            _build([
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late system"},
            ])


# ── Tool calls ────────────────────────────────────────────────────────


class TestToolCalls:

    def test_tools_in_system(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        result = _build(
            [{"role": "user", "content": "find it"}],
            tools=tools,
        )
        assert "system: # Tools" in result
        assert "<tools>" in result
        assert "search" in result

    def test_system_with_tools(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        result = _build([
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "find it"},
        ], tools=tools)
        assert "system: You are an assistant.\n\n# Tools" in result

    def test_assistant_tool_call(self):
        result = _build([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": {"city": "SF"}}}
            ]},
        ], add_gen=False)
        assert "<tool_call>" in result
        assert '"name": "get_weather"' in result
        assert '"city": "SF"' in result
        assert "</tool_call>" in result

    def test_assistant_tool_call_string_arguments(self):
        result = _build([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": '{"a": 1}'}}
            ]},
        ], add_gen=False)
        assert '"arguments": {"a": 1}' in result

    def test_tool_response(self):
        result = _build([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": {"city": "SF"}}}
            ]},
            {"role": "tool", "content": "72F and sunny"},
        ])
        assert "<tool_response>\n72F and sunny\n</tool_response>" in result


# ── Multimodal content ────────────────────────────────────────────────


class TestMultimodalContent:

    def test_image_in_user_message(self):
        result = _build([{
            "role": "user",
            "content": [
                {"type": "image", "image": "dummy"},
                {"type": "text", "text": "describe"},
            ],
        }])
        assert "human: <REKA_IMG_TOKEN>describe<sep>" in result

    def test_video_in_user_message(self):
        result = _build([{
            "role": "user",
            "content": [
                {"type": "video", "video": "dummy"},
                {"type": "text", "text": "describe"},
            ],
        }])
        assert "human: <video></video>describe<sep>" in result

    def test_unsupported_content_type(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            _build([{
                "role": "user",
                "content": [{"type": "audio", "audio": "dummy"}],
            }])


# ── Thinking / reasoning ─────────────────────────────────────────────


class TestThinking:

    def test_thinking_generation_prompt(self):
        result = _build(
            [{"role": "user", "content": "think hard"}],
            thinking=True,
        )
        assert result.endswith("assistant: <think>\n")

    def test_thinking_disabled_generation_prompt(self):
        result = _build(
            [{"role": "user", "content": "think hard"}],
            thinking=False,
        )
        assert result.endswith("assistant:")

    def test_thinking_in_assistant_reply(self):
        result = _build([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer",
             "reasoning_content": "I thought about it"},
        ], add_gen=False, thinking=True)
        assert "<think>" in result
        assert "I thought about it" in result
        assert "</think>" in result
        assert "answer" in result

    def test_thinking_extracted_from_content(self):
        result = _build([
            {"role": "user", "content": "q"},
            {"role": "assistant",
             "content": "<think>\nreasoning here\n</think>\nthe answer"},
        ], add_gen=False, thinking=True)
        assert "reasoning here" in result
        assert "the answer" in result


# ── Continue final message ────────────────────────────────────────────


class TestContinueFinalMessage:

    def test_continue_final_assistant(self):
        result = _build([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "I think"},
        ], add_gen=False, continue_final=True)
        # Should NOT have trailing \n\n<sep> for continued message
        assert result.endswith("I think")
        assert not result.endswith("<sep>")


# ── Validation errors ─────────────────────────────────────────────────


class TestValidation:

    def test_missing_role(self):
        with pytest.raises(ValueError, match="missing 'role'"):
            _build([{"content": "hi"}])

    def test_missing_content(self):
        with pytest.raises(ValueError, match="missing 'content'"):
            _build([{"role": "user"}])

    def test_assistant_without_content_ok_with_tool_calls(self):
        # Should not raise — content=None is OK when tool_calls present
        result = _build([
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": {}}}
            ]},
        ], add_gen=False)
        assert "<tool_call>" in result

    def test_unsupported_role(self):
        with pytest.raises(ValueError, match="Unsupported message role"):
            _build([{"role": "observer", "content": "hi"}])

    def test_non_string_system_content(self):
        with pytest.raises(ValueError, match="System message content"):
            _build([
                {"role": "system", "content": ["not", "a", "string"]},
                {"role": "user", "content": "hi"},
            ])
