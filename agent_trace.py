"""Turn a deepagents `agent.stream()` result into a readable Markdown trace
of this turn's tool calls, subagent invocations, and their results.

Used by the "🔍 這一輪做了什麼" debug expander in streamlit_ui/app.py. The
raw material (`result["messages"]`, a LangChain message list; `result`'s
other state keys) is already there — this module just re-formats it so a
human can actually parse it, instead of dumping LangChain's own
`.pretty_print()` output.
"""

import json
import pprint

# Only these state keys (besides "messages") are worth surfacing in the
# trace — the rest is either internal bookkeeping or not populated yet.
_STATE_FIELDS_OF_INTEREST = ["step", "metadata", "pipeline_reports", "samples"]

# deepagents launches subagents through a single tool literally named
# "task" (see deepagents/middleware/subagents.py: TaskToolSchema), whose
# args are {description, subagent_type} — worth a distinct icon/label from
# a regular tool call.
_TASK_TOOL_NAME = "task"

_MAX_CONTENT_CHARS = 2000
_MAX_STATE_CHARS = 800


def _truncate(text, limit: int = _MAX_CONTENT_CHARS) -> str:
    text = text if isinstance(text, str) else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（已截斷，完整長度 {len(text)} 字元）"


def _format_args(args: dict) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, indent=2)
    except TypeError:
        return str(args)


def _format_tool_call(tool_call: dict) -> str:
    name = tool_call.get("name", "?")
    args = tool_call.get("args", {}) or {}
    if name == _TASK_TOOL_NAME:
        subagent_type = args.get("subagent_type", "?")
        description = args.get("description", "")
        return f"🤖 **呼叫 subagent**：`{subagent_type}`\n\n> {description}"
    return f"🔧 **呼叫工具**：`{name}`\n```json\n{_format_args(args)}\n```"


def _format_tool_result(tool_message, tool_call_name) -> str:
    content = _truncate(getattr(tool_message, "content", ""))
    if tool_call_name == _TASK_TOOL_NAME:
        return f"↩️ **subagent 回報**\n\n{content}"
    name = tool_call_name or getattr(tool_message, "name", None) or "?"
    return f"↩️ **工具回傳**（`{name}`）\n```\n{content}\n```"


def _format_state(state: dict) -> str:
    interesting = {k: v for k, v in state.items() if k in _STATE_FIELDS_OF_INTEREST and v}
    if not interesting:
        return ""
    lines = ["**狀態**"]
    for key, value in interesting.items():
        formatted = _truncate(pprint.pformat(value, width=100, sort_dicts=False), _MAX_STATE_CHARS)
        lines.append(f"- `{key}`：\n```\n{formatted}\n```")
    return "\n".join(lines)


def _this_turn_messages(messages: list) -> list:
    """Slice out only the messages produced since the latest user turn.

    `result["messages"]` accumulates over the whole checkpointed thread, not
    just this call — without slicing, the trace would re-show every past
    turn's tool calls again on every new message.
    """
    last_human_idx = None
    for i, msg in enumerate(messages):
        if getattr(msg, "type", "") == "human":
            last_human_idx = i
    turn_messages = messages[last_human_idx + 1 :] if last_human_idx is not None else messages

    # Drop the trailing plain-text reply — it's already shown as the main
    # chat bubble, no need to repeat it inside the trace.
    if turn_messages and getattr(turn_messages[-1], "type", "") == "ai" and not getattr(
        turn_messages[-1], "tool_calls", None
    ):
        turn_messages = turn_messages[:-1]
    return turn_messages


def format_agent_trace(result: dict) -> str:
    """Build the Markdown body for the debug expander from one turn's result."""
    turn_messages = _this_turn_messages(result.get("messages", []))

    tool_call_names = {}  # tool_call_id -> tool name, so ToolMessages can match their call
    sections = []
    for msg in turn_messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tool_call_names[tc.get("id")] = tc.get("name")
                sections.append(_format_tool_call(tc))
            content = getattr(msg, "content", "")
            if content and content.strip() and not tool_calls:
                sections.append(f"💬 **agent**：{content}")
        elif msg_type == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_name = tool_call_names.get(tool_call_id)
            sections.append(_format_tool_result(msg, tool_name))
        # human/system messages are skipped — already shown as chat bubbles

    other_state = {k: v for k, v in result.items() if k != "messages"}
    state_section = _format_state(other_state)
    if state_section:
        sections.append(state_section)

    if not sections:
        return "_這一輪沒有工具呼叫或額外狀態。_"

    return "\n\n---\n\n".join(sections)
