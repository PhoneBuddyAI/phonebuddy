#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone end-to-end test runner for the full AndroidWorld 116-task benchmark.

This script drives a locally-deployed AndroidWorld FastAPI server (server.android_server)
on a local Android device/emulator, using a vLLM-served model (Qwen3.5 SFT, toolcall format)
as the GUI agent.

It is SELF-CONTAINED: it has no external agent-framework dependency. Everything it
needs (the AndroidWorld HTTP client, the Qwen3.5-SFT-EN prompt + response parser, the
action -> JSONAction mapping with 0-1000 -> pixel scaling, and the vLLM
OpenAI-compatible client) is reimplemented inline here.

Design notes (verified against the android_world env code):

  * Local server endpoints (see server/android_server.py):
      GET  /health
      GET  /suite/task_list?max_index=N
      GET  /suite/reinitialize?n_task_combinations=&seed=&task_family=
      POST /reset?go_home=bool
      GET  /screenshot?wait_to_stabilize=bool       -> {"pixels": [[[r,g,b],...]]}
      POST /execute_action            (JSON body = a json_action.JSONAction dict)
      POST /task/initialize?task_type=&task_idx=
      POST /task/tear_down?task_type=&task_idx=
      GET  /task/score?task_type=&task_idx=         -> {"score": float}
      GET  /task/goal?task_type=&task_idx=          -> {"goal": str}

  * The local server's JSONAction (android_world/env/json_action.py) ONLY accepts these
    action_type values:
        click, double_tap, scroll, swipe, input_text, navigate_home, navigate_back,
        keyboard_enter, open_app, status, wait, long_press, answer, unknown
    and ONLY these fields:
        action_type, index, x, y, text, direction, goal_status, app_name, keycode, clear_text
    Sending unknown fields, or a non-cardinal `direction`, raises on the server side.
    Therefore we map the model's rich action space into this strict schema, and convert
    scroll/swipe `points` into a cardinal direction (up/down/left/right).

  * The model (Qwen3_5_SFT_EN_Prompt) emits:
        <think>...</think> <tool_call>{"name": <fn>, "arguments": {...}}</tool_call>
    with coordinates normalized to 0-1000. We scale them to device pixels.

Usage:
    python3 run_android_world.py                 # run all 116 tasks
    python3 run_android_world.py --limit 3       # smoke-test first 3 tasks
    python3 run_android_world.py --tasks ContactsAddContact,MarkorCreateNote
    python3 run_android_world.py --max-steps 30 --seed 30

Environment overrides:
    AW_SERVER_URL   (default http://127.0.0.1:5001)
    VLLM_BASE_URL   (default http://YOUR_VLLM_HOST:PORT/v1 — set this to your vLLM server)
    VLLM_API_KEY    (default EMPTY)
    VLLM_MODEL      (default: auto-discovered from /v1/models)
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: the `openai` package is required. Install with: pip install openai")
    raise


# =============================================================================
# Configuration (overridable via CLI flags / env vars)
# =============================================================================
DEFAULT_AW_SERVER_URL = os.environ.get("AW_SERVER_URL", "http://127.0.0.1:5001")
DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://YOUR_VLLM_HOST:PORT/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_VLLM_MODEL = os.environ.get("VLLM_MODEL", "")  # "" => auto-discover

# Sampling params (match the toolcall serving config)
SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "max_completion_tokens": 8192,
}

# Normalization range the model is trained on (0-1000 per the Qwen3.5 SFT EN system prompt)
COORD_NORM = 1000

logger = logging.getLogger("aw116")


# =============================================================================
# Qwen3.5 SFT EN system prompt (the toolcall-format system prompt)
# =============================================================================
QWEN3_5_SFT_EN_SYSTEM_PROMPT = """You are a GUI Agent. Given an instruction, the current screenshot, and the history of operations, you need to predict how to fulfill the user's request and provide the accurate invocation command. Please note that coordinate values must be scaled to a range of 0 to 1000.

# Tools

You can call one or more of the following functions to complete the user's request.

Below is the complete list of tools supported by the system:
<tools>
{"type": "function", "function": {"name": "click", "description": "Click on a specified coordinate position on the screen (coordinate range 0-1000)", "parameters": {"type": "object", "properties": {"points": {"description": "A list of click coordinates, formatted as [[x, y]]", "type": "array"}}, "required": ["points"]}}}
{"type": "function", "function": {"name": "double_click", "description": "Double-click on a specified coordinate position on the screen", "parameters": {"type": "object", "properties": {"points": {"description": "Click coordinates [[x, y]]", "type": "array"}, "interval": {"description": "Interval between two clicks (milliseconds)", "type": "integer"}}, "required": ["points"]}}}
{"type": "function", "function": {"name": "long_press", "description": "Long press on a specified coordinate position on the screen", "parameters": {"type": "object", "properties": {"points": {"description": "Long press coordinates [[x, y]]", "type": "array"}, "duration": {"description": "Long press duration (milliseconds)", "type": "integer"}}, "required": ["points"]}}}
{"type": "function", "function": {"name": "type", "description": "Type text in the currently focused input field", "parameters": {"type": "object", "properties": {"text": {"description": "The text content to type", "type": "string"}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "scroll", "description": "Scroll from start coordinates to target coordinates (for scrolling pages)", "parameters": {"type": "object", "properties": {"points": {"description": "Start and end coordinates for scrolling [[x1, y1], [x2, y2]]", "type": "array"}, "duration": {"description": "Scroll duration (milliseconds)", "type": "integer"}}, "required": ["points"]}}}
{"type": "function", "function": {"name": "drag", "description": "Drag an element from start coordinates to target coordinates", "parameters": {"type": "object", "properties": {"points": {"description": "Start and end coordinates for dragging [[x1, y1], [x2, y2]]", "type": "array"}, "duration": {"description": "Drag duration (milliseconds)", "type": "integer"}}, "required": ["points"]}}}
{"type": "function", "function": {"name": "button_press", "description": "Press a phone physical/virtual button", "parameters": {"type": "object", "properties": {"type": {"description": "Button type: back/home/menu/enter", "type": "string", "enum": ["back", "home", "menu", "enter"]}}, "required": ["type"]}}}
{"type": "function", "function": {"name": "open_app", "description": "Open an app by package name", "parameters": {"type": "object", "properties": {"package": {"description": "App package name", "type": "string"}}, "required": ["package"]}}}
{"type": "function", "function": {"name": "close_app", "description": "Close an app by package name", "parameters": {"type": "object", "properties": {"package": {"description": "App package name", "type": "string"}}, "required": ["package"]}}}
{"type": "function", "function": {"name": "wait", "description": "Wait for a specified duration", "parameters": {"type": "object", "properties": {"time": {"description": "Wait duration (milliseconds)", "type": "integer"}}, "required": ["time"]}}}
{"type": "function", "function": {"name": "output", "description": "Output information to the user", "parameters": {"type": "object", "properties": {"text": {"description": "The text content to output", "type": "string"}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "finish", "description": "Mark the task as complete and output the final result", "parameters": {"type": "object", "properties": {"text": {"description": "Description or result of the completed task", "type": "string"}}, "required": ["text"]}}}
</tools>

When making a function call, first output <tool_call>
The format for each function call is as follows:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
At the end of the function call(s), output </tool_call>"""


# =============================================================================
# Lenient JSON parsing (json_repair is not installed; this covers the common
# malformations vLLM models produce: trailing commas, single quotes, smart
# quotes, stray text around the object).
# =============================================================================
def lenient_json_loads(s: str) -> Any:
    """Best-effort JSON object parse. Returns parsed object or raises ValueError."""
    s = s.strip()
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass
    # Extract the outermost {...} if there is surrounding noise.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    # Normalize smart quotes.
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        return json.loads(s)
    except Exception:
        pass
    # Remove trailing commas before } or ].
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except Exception:
        pass
    # Last resort: swap single quotes for double quotes (only when no double quotes present).
    if '"' not in s2:
        try:
            return json.loads(s2.replace("'", '"'))
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON from: {s[:200]!r}")


# =============================================================================
# AndroidWorld HTTP client — talks to the local FastAPI server
# (server.android_server) over HTTP.
# =============================================================================
class AndroidWorldClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, *, params=None, json_body=None, timeout=None) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = requests.request(
            method, url, params=params, json=json_body, timeout=timeout or self.timeout
        )
        if not resp.ok:
            detail = ""
            try:
                detail = json.dumps(resp.json())[:500]
            except Exception:
                detail = resp.text[:500]
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {method} {path}. Server said: {detail}",
                response=resp,
            )
        return resp

    # --- health / suite -------------------------------------------------------
    def health(self) -> bool:
        try:
            self._request("GET", "/health", timeout=15)
            return True
        except Exception as e:
            logger.warning(f"health check failed: {e}")
            return False

    def get_suite_task_list(self, max_index: int = 100000) -> List[str]:
        return self._request("GET", "/suite/task_list", params={"max_index": max_index}, timeout=60).json()["task_list"]

    def reinitialize_suite(self, n_task_combinations: int = 1, seed: int = 42, task_family: str = "android_world"):
        return self._request(
            "GET",
            "/suite/reinitialize",
            params={"n_task_combinations": n_task_combinations, "seed": seed, "task_family": task_family},
            timeout=120,
        ).json()

    # --- env ------------------------------------------------------------------
    def reset(self, go_home: bool = True):
        return self._request("POST", "/reset", params={"go_home": go_home}, timeout=120).json()

    def get_screenshot(self, wait_to_stabilize: bool = False) -> np.ndarray:
        """Local server returns {'pixels': [[[r,g,b], ...], ...]} as JSON."""
        resp = self._request(
            "GET", "/screenshot", params={"wait_to_stabilize": wait_to_stabilize}, timeout=180
        )
        ctype = resp.headers.get("Content-Type", "").lower()
        if "image/" in ctype:
            return np.array(Image.open(io.BytesIO(resp.content)), dtype=np.uint8)
        data = resp.json()
        if "image_base64" in data:
            return np.array(Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))), dtype=np.uint8)
        return np.array(data["pixels"], dtype=np.uint8)

    def execute_action(self, action_dict: Dict[str, Any]):
        return self._request("POST", "/execute_action", json_body=action_dict, timeout=120).json()

    # --- task -----------------------------------------------------------------
    def initialize_task(self, task_type: str, task_idx: int = 0):
        return self._request(
            "POST", "/task/initialize", params={"task_type": task_type, "task_idx": task_idx}, timeout=180
        ).json()

    def tear_down_task(self, task_type: str, task_idx: int = 0):
        return self._request(
            "POST", "/task/tear_down", params={"task_type": task_type, "task_idx": task_idx}, timeout=120
        ).json()

    def get_task_score(self, task_type: str, task_idx: int = 0) -> float:
        return self._request(
            "GET", "/task/score", params={"task_type": task_type, "task_idx": task_idx}, timeout=120
        ).json()["score"]

    def get_task_goal(self, task_type: str, task_idx: int = 0) -> str:
        return self._request(
            "GET", "/task/goal", params={"task_type": task_type, "task_idx": task_idx}, timeout=60
        ).json()["goal"]


# =============================================================================
# vLLM OpenAI-compatible client (chat.completions + model auto-discovery).
# =============================================================================
class VLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str = "", max_try: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_try = max_try
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.model = model or self._discover_model()
        logger.info(f"vLLM model id: {self.model}")

    def _discover_model(self) -> str:
        models = self.client.models.list()
        if not models.data:
            raise RuntimeError(f"No models served at {self.base_url}")
        return models.data[0].id

    @staticmethod
    def _extract_text(message) -> str:
        """
        Pull the model's text out of a chat message.

        This server is a *reasoning* model: it emits both its thinking AND the
        final <tool_call> into `message.reasoning` (a.k.a. `reasoning_content`),
        leaving `message.content` as null. So we concatenate content + reasoning
        and let the parser find the <tool_call> wherever it landed.
        """
        parts = []
        content = getattr(message, "content", None)
        if content:
            parts.append(content)
        for attr in ("reasoning", "reasoning_content"):
            val = getattr(message, attr, None)
            if val:
                parts.append(val)
        if not parts:
            # Some SDK versions surface unknown fields only via model_dump().
            try:
                dumped = message.model_dump()
                for attr in ("content", "reasoning", "reasoning_content"):
                    if dumped.get(attr):
                        parts.append(dumped[attr])
            except Exception:
                pass
        return "\n".join(parts)

    def request(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        last_err = None
        for attempt in range(1, self.max_try + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=SAMPLING["temperature"],
                    top_p=SAMPLING["top_p"],
                    stream=False,
                    max_completion_tokens=SAMPLING["max_completion_tokens"],
                    extra_body={
                        "top_k": SAMPLING["top_k"],
                        "repetition_penalty": SAMPLING["repetition_penalty"],
                        "skip_special_tokens": False,
                    },
                )
                choice = completion.choices[0]
                text = self._extract_text(choice.message)
                if not text and choice.finish_reason == "length":
                    logger.warning(
                        "vLLM hit max_completion_tokens before producing a tool_call; "
                        "consider raising SAMPLING['max_completion_tokens']."
                    )
                return text
            except Exception as e:
                last_err = e
                logger.warning(f"vLLM request attempt {attempt}/{self.max_try} failed: {e}")
                time.sleep(min(2 * attempt, 10))
        logger.error(f"vLLM request exhausted retries: {last_err}")
        return None


# =============================================================================
# Prompt building + response parsing (Qwen3_5_SFT_EN_Prompt, reimplemented).
# =============================================================================
def encode_image_to_data_url(pixels: np.ndarray) -> str:
    """Encode an RGB numpy array as a JPEG base64 data URL."""
    img = Image.fromarray(pixels).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(instruction: str, image_data_url: str, context: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build the chat messages exactly like Qwen3_5_SFT_EN_Prompt._build_prompt."""
    content_history = ""
    for step_idx, content in enumerate(context):
        thought = content.get("thought", "")
        action_str = content.get("action_str", "")
        content_history += f"- Step {step_idx + 1}:\nThought: {thought}\nAction: {action_str}\n"
    content_history = content_history.strip()
    history_section = f"\n\n# History of Previous Actions\n{content_history}" if content_history else ""

    user_content = [
        {"type": "text", "text": QWEN3_5_SFT_EN_SYSTEM_PROMPT},
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": f"# Instruction\n{instruction}{history_section}"},
    ]
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]


def _extract_point(args: Dict[str, Any], index: int = 0) -> Optional[List[int]]:
    """Extract [x, y] from points/coordinate fields. Supports [[x,y],...] and [x,y]."""
    coord = None
    for key in ("points", "coordinate", "point", "coordinates"):
        if key in args and args[key] is not None:
            coord = args[key]
            break
    if coord is None:
        return None
    if isinstance(coord, list) and coord:
        if isinstance(coord[0], list):
            if index < len(coord) and len(coord[index]) >= 2:
                return [int(coord[index][0]), int(coord[index][1])]
            return None
        if len(coord) >= 2:
            return [int(coord[0]), int(coord[1])]
    return None


def parse_model_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse <think>..</think> + <tool_call>{json}</tool_call> into a normalized action dict:
        {"action": <fn>, "cot": <thought>, "args": {<func args, with 'points' as [x,y] pairs>}}
    Coordinates remain in the 0-1000 space. Returns None on failure.
    """
    if not response:
        return None
    try:
        think = ""
        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if m:
            think = m.group(1).strip()
        remaining = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        tc = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", remaining, re.DOTALL)
        if not tc:
            tc = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)
        if not tc:
            # Some models omit the closing tag; grab everything after the opening tag.
            tc = re.search(r"<tool_call>\s*(\{.*\})", response, re.DOTALL)
        if not tc:
            logger.warning("response missing <tool_call> block")
            return None

        thought_text = remaining.split("<tool_call>")[0].strip() if "<tool_call>" in remaining else ""
        full_thought = "\n".join(x for x in (think, thought_text) if x).strip()

        obj = lenient_json_loads(tc.group(1).strip())
        if not isinstance(obj, dict):
            return None
        name = (obj.get("name") or "").strip()
        args = obj.get("arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}
        # open_app/close_app use "package" -> normalize to "app".
        if name in ("open_app", "close_app") and "package" in args:
            args["app"] = args.pop("package")
        return {"action": name.lower(), "cot": full_thought, "args": args}
    except Exception as e:
        logger.error(f"parse_model_response error: {e}\n{traceback.format_exc()}")
        return None


def action_to_history_str(action: str, args: Dict[str, Any]) -> str:
    """Render the action as func_name({...}) for the prompt history."""
    params: Dict[str, Any] = {}
    a = action
    if action in ("click", "double_click", "long_press"):
        if "points" in args:
            params["points"] = args["points"]
        if action == "long_press" and "duration" in args:
            params["duration"] = args["duration"]
    elif action == "type":
        params["text"] = args.get("text", "")
    elif action in ("scroll", "swipe", "drag"):
        if "points" in args:
            params["points"] = args["points"]
        if "duration" in args:
            params["duration"] = args["duration"]
    elif action == "button_press":
        params["type"] = args.get("type", "")
    elif action in ("open_app", "close_app"):
        params["app"] = args.get("app", "")
    elif action == "wait":
        params["time"] = int(args.get("time", 1000))
    elif action in ("finish", "answer", "output"):
        a = "finish"
        params["text"] = args.get("text", "")
    return f"{a}({json.dumps(params, ensure_ascii=False)})"


# =============================================================================
# Action -> AndroidWorld JSONAction mapping.
#
# The local server's JSONAction is strict: action_type must be one of the
# allowed verbs, scroll/swipe need a CARDINAL direction, and only known fields
# may be present. We translate the model's 0-1000 action into that schema,
# scaling coordinates to device pixels.
#
# Returns (json_action_dict_or_None, is_terminal, terminal_text).
# =============================================================================
TERMINAL_ACTIONS = ("finish", "answer", "output", "terminate", "call_user")


def _scale_xy(point_norm: List[int], screen_w: int, screen_h: int) -> Tuple[int, int]:
    x = int(point_norm[0] * screen_w / COORD_NORM)
    y = int(point_norm[1] * screen_h / COORD_NORM)
    x = max(0, min(x, screen_w - 1))
    y = max(0, min(y, screen_h - 1))
    return x, y


def _points_to_gesture_direction(p1: List[int], p2: List[int]) -> str:
    """
    Convert a scroll/swipe from start point p1 to end point p2 (the finger's
    movement) into the cardinal `direction` the local server's `scroll` action
    expects.

    The android_world `scroll` action moves CONTENT in `direction`; the finger
    moves opposite to the content, so we reverse the gesture to get the API
    direction:
        finger up    (dy<0) -> content scrolls down  -> direction="down"
        finger down  (dy>0) -> content scrolls up    -> direction="up"
        finger left  (dx<0) -> content scrolls right -> direction="right"
        finger right (dx>0) -> content scrolls left  -> direction="left"
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dy) >= abs(dx):
        gesture = "up" if dy < 0 else "down"
    else:
        gesture = "left" if dx < 0 else "right"
    reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}
    return reverse[gesture]


def map_action_to_jsonaction(
    action: str, args: Dict[str, Any], screen_w: int, screen_h: int
) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """Translate a parsed model action into a JSONAction dict for /execute_action."""
    if action in ("click",):
        pt = _extract_point(args)
        if pt is None:
            return None, False, ""
        x, y = _scale_xy(pt, screen_w, screen_h)
        return {"action_type": "click", "x": x, "y": y}, False, ""

    if action == "double_click":
        pt = _extract_point(args)
        if pt is None:
            return None, False, ""
        x, y = _scale_xy(pt, screen_w, screen_h)
        return {"action_type": "double_tap", "x": x, "y": y}, False, ""

    if action in ("long_press", "longpress"):
        pt = _extract_point(args)
        if pt is None:
            return None, False, ""
        x, y = _scale_xy(pt, screen_w, screen_h)
        return {"action_type": "long_press", "x": x, "y": y}, False, ""

    if action == "type":
        return {"action_type": "input_text", "text": str(args.get("text", ""))}, False, ""

    if action in ("scroll", "swipe", "drag"):
        # Preferred path: the model gave two explicit points -> send a literal
        # coordinate-based swipe (start -> end) using the model-predicted pixels.
        # No direction derivation, no precision loss. The patched local server
        # (json_action.x2/y2 + actuation.py) executes this as a finger drag.
        p1 = _extract_point(args, 0)
        p2 = _extract_point(args, 1)
        if p1 is not None and p2 is not None:
            x1, y1 = _scale_xy(p1, screen_w, screen_h)
            x2, y2 = _scale_xy(p2, screen_w, screen_h)
            ja = {"action_type": "swipe", "x": x1, "y": y1, "x2": x2, "y2": y2}
            dur = args.get("duration")
            if isinstance(dur, (int, float)) and dur > 0:
                ja["duration"] = int(dur)
            return ja, False, ""

        # Fallback: only a bare cardinal `direction` (no usable coordinates).
        direction = args.get("direction")
        if direction in ("up", "down", "left", "right"):
            # Model gesture direction -> reverse to content-scroll direction.
            reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}
            return {"action_type": "scroll", "direction": reverse[direction]}, False, ""

        # Single point or unusable args: cannot form a gesture.
        return None, False, ""

    if action == "button_press":
        btn = str(args.get("type", "")).lower()
        if btn == "back":
            return {"action_type": "navigate_back"}, False, ""
        if btn == "home":
            return {"action_type": "navigate_home"}, False, ""
        if btn == "enter":
            return {"action_type": "keyboard_enter"}, False, ""
        # 'menu' has no direct JSONAction; fall back to keyboard_enter? No — skip.
        logger.warning(f"unsupported button_press type: {btn}")
        return None, False, ""

    if action in ("open_app", "start_app"):
        app = str(args.get("app", "") or args.get("package", ""))
        if app in ("File Manager", "file_management"):
            app = "Files"
        if not app:
            return None, False, ""
        return {"action_type": "open_app", "app_name": app}, False, ""

    if action == "close_app":
        # No close_app in JSONAction; go home as a benign approximation.
        return {"action_type": "navigate_home"}, False, ""

    if action == "wait":
        return {"action_type": "wait"}, False, ""

    if action in TERMINAL_ACTIONS:
        text = str(args.get("text", ""))
        # Submit the answer so info-retrieval tasks can be scored, then end episode.
        return {"action_type": "answer", "text": text}, True, text

    logger.warning(f"unknown action '{action}' -> treating as no-op")
    return None, False, ""


# =============================================================================
# Single-task runner (the agent loop).
# =============================================================================
def run_one_task(
    aw: AndroidWorldClient,
    vllm: VLLMClient,
    task_type: str,
    task_idx: int,
    seed: int,
    max_steps: int,
    step_delay: float,
    images_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize a task, run the agent loop, score it, tear it down. Returns a result dict."""
    record: Dict[str, Any] = {
        "task_type": task_type,
        "task_idx": task_idx,
        "seed": seed,
        "goal": "",
        "score": 0.0,
        "steps": 0,
        "success": False,
        "terminated_by": None,
        "error": None,
        "start_time": datetime.now().isoformat(),
        "trajectory": [],
    }
    screen_w = screen_h = None

    try:
        # --- setup: reset -> reinitialize -> tear_down(ignore) -> initialize ---
        aw.reset(go_home=True)
        aw.reinitialize_suite(n_task_combinations=1, seed=seed)
        try:
            aw.tear_down_task(task_type, task_idx)
        except Exception:
            pass
        aw.initialize_task(task_type, task_idx)
        goal = aw.get_task_goal(task_type, task_idx)
        record["goal"] = goal
        logger.info(f"[{task_type}] goal: {goal}")

        context: List[Dict[str, str]] = []
        terminal_text = ""

        for step in range(max_steps):
            record["steps"] = step + 1
            # Give the UI a moment to settle (3s, matching android_world cadence).
            time.sleep(3)

            # 1. screenshot
            try:
                pixels = aw.get_screenshot(wait_to_stabilize=False)
            except Exception as e:
                logger.error(f"[{task_type}] screenshot failed at step {step}: {e}")
                record["error"] = f"screenshot_failed: {e}"
                break
            if screen_w is None:
                screen_h, screen_w = int(pixels.shape[0]), int(pixels.shape[1])
                logger.info(f"[{task_type}] screen size = {screen_w}x{screen_h}")

            if images_dir:
                try:
                    os.makedirs(images_dir, exist_ok=True)
                    Image.fromarray(pixels).convert("RGB").save(
                        os.path.join(images_dir, f"{task_type}_step{step}.jpg"), "JPEG", quality=85
                    )
                except Exception:
                    pass

            image_url = encode_image_to_data_url(pixels)

            # 2. build prompt + call model
            messages = build_messages(goal, image_url, context)
            response = vllm.request(messages)
            if not response:
                logger.error(f"[{task_type}] empty model response at step {step}")
                record["error"] = "empty_model_response"
                break

            # 3. parse
            parsed = parse_model_response(response)
            if parsed is None:
                logger.error(f"[{task_type}] parse failed at step {step}; raw: {response[:300]!r}")
                record["error"] = "parse_failed"
                record["trajectory"].append(
                    {"step": step, "raw_response": response[:2000], "parsed": None}
                )
                break

            action = parsed["action"]
            args = parsed["args"]
            cot = parsed["cot"]

            # 4. map to JSONAction
            json_action, is_terminal, term_text = map_action_to_jsonaction(action, args, screen_w, screen_h)

            step_rec = {
                "step": step,
                "action": action,
                "args": args,
                "cot": cot[:500],
                "json_action": json_action,
                "is_terminal": is_terminal,
            }

            # 5. execute
            exec_ok = True
            exec_err = None
            if json_action is not None:
                try:
                    aw.execute_action(json_action)
                except Exception as e:
                    exec_ok = False
                    exec_err = str(e)
                    logger.warning(f"[{task_type}] execute_action failed at step {step}: {e}")
            step_rec["exec_ok"] = exec_ok
            step_rec["exec_error"] = exec_err
            record["trajectory"].append(step_rec)

            # 6. update history context
            context.append(
                {"thought": cot, "action_str": action_to_history_str(action, args)}
            )

            # 7. terminal?
            if is_terminal:
                terminal_text = term_text
                record["terminated_by"] = action
                logger.info(f"[{task_type}] agent finished at step {step}: {term_text[:120]!r}")
                break

            if step < max_steps - 1:
                time.sleep(step_delay)
        else:
            record["terminated_by"] = "max_steps"

        # --- score ---
        try:
            score = float(aw.get_task_score(task_type, task_idx))
        except Exception as e:
            logger.error(f"[{task_type}] scoring failed: {e}")
            score = 0.0
            record["error"] = (record["error"] or "") + f" | score_failed: {e}"
        record["score"] = score
        record["success"] = score >= 1.0
        record["answer_text"] = terminal_text
        logger.info(f"[{task_type}] SCORE={score}  success={record['success']}  steps={record['steps']}")

    except Exception as e:
        logger.error(f"[{task_type}] fatal error: {e}\n{traceback.format_exc()}")
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        # --- always tear down ---
        try:
            aw.tear_down_task(task_type, task_idx)
        except Exception:
            pass
        record["end_time"] = datetime.now().isoformat()

    return record


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run the full AndroidWorld 116-task benchmark with a vLLM agent.")
    parser.add_argument("--aw-server", default=DEFAULT_AW_SERVER_URL, help="AndroidWorld FastAPI server base URL")
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL, help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--vllm-api-key", default=DEFAULT_VLLM_API_KEY, help="vLLM API key")
    parser.add_argument("--vllm-model", default=DEFAULT_VLLM_MODEL, help="vLLM model id (default: auto-discover)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per task")
    parser.add_argument("--step-delay", type=float, default=2.0, help="Delay between steps (seconds)")
    parser.add_argument("--seed", type=int, default=30, help="Suite seed (matches seed30 config naming)")
    parser.add_argument("--task-idx", type=int, default=0, help="Task combination index within each task type")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke test)")
    parser.add_argument("--tasks", default=None, help="Comma-separated subset of task_type names to run")
    parser.add_argument("--output-dir", default="./aw116_output", help="Directory for results + screenshots")
    parser.add_argument("--save-images", action="store_true", help="Save per-step screenshots")
    parser.add_argument("--resume", action="store_true", help="Skip task_types already present in the results file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    images_root = os.path.join(args.output_dir, "images") if args.save_images else None

    # --- connect to both servers --------------------------------------------
    aw = AndroidWorldClient(args.aw_server)
    if not aw.health():
        print(f"FATAL: AndroidWorld server not healthy at {args.aw_server}")
        sys.exit(1)
    logger.info(f"AndroidWorld server OK: {args.aw_server}")

    try:
        vllm = VLLMClient(args.vllm_base_url, args.vllm_api_key, args.vllm_model)
    except Exception as e:
        print(f"FATAL: could not connect to vLLM at {args.vllm_base_url}: {e}")
        sys.exit(1)

    # --- task list (from the live suite) -------------------------------------
    all_tasks = aw.get_suite_task_list(100000)
    logger.info(f"suite reports {len(all_tasks)} tasks")

    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in all_tasks]
        if unknown:
            print(f"WARNING: these requested tasks are not in the suite and will be skipped: {unknown}")
        tasks = [t for t in all_tasks if t in wanted]
    else:
        tasks = list(all_tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    # --- resume support -------------------------------------------------------
    already_done = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["task_type"])
                except Exception:
                    continue
        tasks = [t for t in tasks if t not in already_done]
        logger.info(f"resume: skipping {len(already_done)} already-done tasks; {len(tasks)} remain")

    logger.info(f"Running {len(tasks)} task(s): seed={args.seed}, max_steps={args.max_steps}")

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    with open(results_path, "a", encoding="utf-8") as out_f:
        for i, task_type in enumerate(tasks):
            logger.info("=" * 78)
            logger.info(f"[{i + 1}/{len(tasks)}] {task_type}")
            logger.info("=" * 78)
            images_dir = os.path.join(images_root, task_type) if images_root else None
            rec = run_one_task(
                aw, vllm, task_type, args.task_idx, args.seed,
                args.max_steps, args.step_delay, images_dir,
            )
            results.append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            done = i + 1
            succ = sum(1 for r in results if r["success"])
            logger.info(f"progress: {done}/{len(tasks)}  running success={succ} ({succ/done:.1%})")

    # --- summary --------------------------------------------------------------
    elapsed = time.time() - t_start
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_error = sum(1 for r in results if r.get("error"))
    summary = {
        "total": n,
        "success": n_success,
        "success_rate": (n_success / n) if n else 0.0,
        "errors": n_error,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "vllm_model": vllm.model,
        "elapsed_sec": round(elapsed, 1),
        "per_task": {r["task_type"]: {"score": r["score"], "success": r["success"],
                                      "steps": r["steps"], "error": r.get("error")} for r in results},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("AndroidWorld 116-task run complete")
    print("=" * 78)
    print(f"  tasks run:     {n}")
    print(f"  successes:     {n_success}")
    print(f"  success rate:  {summary['success_rate']:.1%}")
    print(f"  errored tasks: {n_error}")
    print(f"  elapsed:       {elapsed/60:.1f} min")
    print(f"  results:       {results_path}")
    print(f"  summary:       {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
