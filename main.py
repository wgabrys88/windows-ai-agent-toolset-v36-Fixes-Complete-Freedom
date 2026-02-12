"""
FRANZ -- Autonomous screen agent.
"""

import base64
import ctypes
import ctypes.wintypes
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Final

import drawing
import screenshot

API: Final = "http://localhost:1234/v1/chat/completions"
MODEL: Final = "qwen3-vl-2b-instruct-1m"
WIDTH: Final = 1536
HEIGHT: Final = 864
WIDTH: Final = 736
HEIGHT: Final = 464
SAMPLING: Final = {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1500}
MAX_ACTIONS: Final = 3
VISUAL_MARKS: Final = True

MOVE_STEPS: Final = 20
STEP_DELAY: Final = 0.01
CLICK_DELAY: Final = 0.15
CHAR_DELAY: Final = 0.08
WORD_DELAY: Final = 0.15
ACTION_GAP: Final = 0.3
LOOP_DELAY: Final = 1.0

SYSTEM: Final = """\
You are a Python expert assistant. The user gives you a goal and a screenshot of their screen.

Magenta marks on the screenshot show user actions from the past. They are not part of any application.

You have these Python functions already defined and ready to call:

def left_click(x: int, y: int) -> None:
    \"\"\"Click left mouse button. Coordinates 0-1000.\"\"\"

def right_click(x: int, y: int) -> None:
    \"\"\"Click right mouse button. Coordinates 0-1000.\"\"\"

def double_left_click(x: int, y: int) -> None:
    \"\"\"Double click left mouse button. Coordinates 0-1000.\"\"\"

def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    \"\"\"Drag from (x1,y1) to (x2,y2). Coordinates 0-1000.\"\"\"

def type(text: str) -> None:
    \"\"\"Type text on keyboard.\"\"\"

def explanation(text: str) -> None:
    \"\"\"Mandatory. Explain what you see, what you did and why.\"\"\"

Top-left is 0,0. Bottom-right is 1000,1000.

Respond only with Python function calls, one per line. No imports, no variables, no comments, no other text. The last call must be explanation().\
"""

USER_TEMPLATE: Final = """\
{story}

Look at the screenshot, analyze my markings and help me with next steps towards my ultimate goal.\ 
Provide me python function calls with a mandatory explanation function at the END of the list \ 
I need to know why I have do the calls and what will be the outcome, I dont want to perform actions blindly. \
"""

KNOWN_FUNCTIONS: Final = frozenset({
    "left_click", "right_click", "double_left_click", "drag", "type", "explanation",
})

_user32 = ctypes.windll.user32
_screen_w = _user32.GetSystemMetrics(0)
_screen_h = _user32.GetSystemMetrics(1)


def _to_px(val: int, dim: int) -> int:
    return int(val / 1000 * dim)


def _cursor() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _move(tx: int, ty: int) -> None:
    sx, sy = _cursor()
    for i in range(MOVE_STEPS + 1):
        t = i / MOVE_STEPS
        t = t * t * (3 - 2 * t)
        _user32.SetCursorPos(int(sx + (tx - sx) * t), int(sy + (ty - sy) * t))
        time.sleep(STEP_DELAY)


def _click(down: int, up: int) -> None:
    _user32.mouse_event(down, 0, 0, 0, 0)
    time.sleep(0.05)
    _user32.mouse_event(up, 0, 0, 0, 0)


def _key(vk: int) -> None:
    _user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(vk, 0, 2, 0)


def _type(text: str) -> None:
    for ch in text:
        if ch == " ":
            _key(0x20)
            time.sleep(WORD_DELAY)
        elif ch == "\n":
            _key(0x0D)
            time.sleep(WORD_DELAY)
        else:
            vk = _user32.VkKeyScanW(ord(ch))
            if vk == -1:
                continue
            shift = (vk >> 8) & 1
            if shift:
                _user32.keybd_event(0x10, 0, 0, 0)
                time.sleep(0.01)
            _key(vk & 0xFF)
            if shift:
                _user32.keybd_event(0x10, 0, 2, 0)
            time.sleep(CHAR_DELAY)


story = ""


def left_click(x: int, y: int) -> None:
    _move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _click(0x0002, 0x0004)


def right_click(x: int, y: int) -> None:
    _move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _click(0x0008, 0x0010)


def double_left_click(x: int, y: int) -> None:
    _move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _click(0x0002, 0x0004)
    time.sleep(0.08)
    _click(0x0002, 0x0004)


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    _move(_to_px(x1, _screen_w), _to_px(y1, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.1)
    _move(_to_px(x2, _screen_w), _to_px(y2, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(0x0004, 0, 0, 0, 0)


def type(text: str) -> None:
    _type(text)


def explanation(text: str) -> None:
    global story
    story = text


DISPATCH: Final = {
    "left_click": left_click,
    "right_click": right_click,
    "double_left_click": double_left_click,
    "drag": drag,
    "type": type,
    "explanation": explanation,
}


def run_output(raw: str) -> list[str]:
    executed = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        paren = line.find("(")
        if paren == -1:
            continue
        name = line[:paren].strip()
        if name not in KNOWN_FUNCTIONS:
            continue
        try:
            eval(line, {"__builtins__": {}}, DISPATCH)
            executed.append(line)
        except Exception:
            pass
    return executed


def format_overlay(name: str, args_str: str, params) -> None:
    pass


def make_overlay(actions: list[str]):
    def draw(rgba: bytes, w: int, h: int) -> bytes:
        if not VISUAL_MARKS:
            return rgba
        buf = bytearray(rgba)
        px, py = None, None
        for line in actions:
            paren = line.find("(")
            if paren == -1:
                continue
            name = line[:paren].strip()
            try:
                args = eval(f"[{line[paren+1:line.rfind(')')]}]", {"__builtins__": {}}, {})
            except Exception:
                continue
            match name:
                case "left_click" if len(args) >= 2:
                    x, y = drawing.normalize_coord(int(args[0]), w), drawing.normalize_coord(int(args[1]), h)
                    drawing.annotate_left_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "right_click" if len(args) >= 2:
                    x, y = drawing.normalize_coord(int(args[0]), w), drawing.normalize_coord(int(args[1]), h)
                    drawing.annotate_right_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "double_left_click" if len(args) >= 2:
                    x, y = drawing.normalize_coord(int(args[0]), w), drawing.normalize_coord(int(args[1]), h)
                    drawing.annotate_double_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "drag" if len(args) >= 4:
                    x1, y1 = drawing.normalize_coord(int(args[0]), w), drawing.normalize_coord(int(args[1]), h)
                    x2, y2 = drawing.normalize_coord(int(args[2]), w), drawing.normalize_coord(int(args[3]), h)
                    drawing.annotate_drag(buf, w, h, x1, y1, x2, y2, px, py)
                    px, py = x2, y2
                case "type":
                    if px is not None:
                        drawing.annotate_type(buf, w, h, px, py)
        return bytes(buf)
    return draw


def infer(png: bytes, story: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEMPLATE.format(story=story)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,"
                            f"{base64.b64encode(png).decode()}"
                        },
                    },
                ],
            },
        ],
        **SAMPLING,
    }

    req = urllib.request.Request(
        API, json.dumps(payload).encode(), {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as f:
        return json.load(f)["choices"][0]["message"]["content"]


def main() -> None:
    global story

    time.sleep(3)

    dump = Path("dump") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    dump.mkdir(parents=True, exist_ok=True)

    prev: list[str] = []
    turn = 0

    print(f"\n{'=' * 60}")
    print("FRANZ")
    print(f"{'=' * 60}\n")

    while True:
        turn += 1

        img = screenshot.capture_screen_png(WIDTH, HEIGHT, draw_func=make_overlay(prev))
        (dump / f"{int(time.time() * 1000)}.png").write_bytes(img)

        user_text = USER_TEMPLATE.format(story=story)

        print(f"\n{'=' * 60}")
        print(f"  turn {turn}")
        print(f"{'=' * 60}")
        print(f"  sending:")
        for line in user_text.splitlines():
            print(f"    | {line}")
        print(f"  thinking...")

        raw = infer(img, story)

        print(f"  vlm ({len(raw)} chars):")
        print(f"{'-' * 40}")
        print(raw)
        print(f"{'-' * 40}")

        old_story = story
        executed = run_output(raw)

        print(f"  executed: {len(executed)}")
        for i, e in enumerate(executed):
            print(f"    [{i}] {e}")
        print(f"  story: {story}")

        prev = [e for e in executed if not e.startswith("explanation")] or prev

        state = {
            "turn": turn,
            "story": story,
            "vlm_raw": raw,
            "executed": executed,
            "timestamp": datetime.now().isoformat(),
        }
        (dump / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        (dump / "story.txt").write_text(story, encoding="utf-8")
        Path("story.txt").write_text(story, encoding="utf-8")

        print(f"  rest {LOOP_DELAY}s...")
        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    main()