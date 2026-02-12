"""
FRANZ - Autonomous screen agent.

Single-file implementation: screen capture (GDI), drawing annotations,
and the main agent loop.  Windows 11 / Python 3.12+.
"""

import base64
import ctypes
import ctypes.wintypes
import json
import math
import struct
import sys
import time
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Final

# =============================================================================
# Type aliases
# =============================================================================

type Color = tuple[int, int, int, int]

# =============================================================================
# Configuration
# =============================================================================

API: Final[str] = "http://localhost:1234/v1/chat/completions"
MODEL: Final[str] = "qwen3-vl-2b-instruct-1m"
WIDTH: Final[int] = 736
HEIGHT: Final[int] = 464
SAMPLING: Final[dict[str, float | int]] = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1500,
}
VISUAL_MARKS: Final[bool] = True

MOVE_STEPS: Final[int] = 20
STEP_DELAY: Final[float] = 0.01
CLICK_DELAY: Final[float] = 0.15
CHAR_DELAY: Final[float] = 0.08
WORD_DELAY: Final[float] = 0.15
LOOP_DELAY: Final[float] = 1.0

SYSTEM_PROMPT: Final[str] = """\
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

USER_TEMPLATE: Final[str] = """\
{story}

Look at the screenshot, analyze my markings and help me with next steps towards my ultimate goal.\
 Provide me python function calls with a mandatory explanation function at the END of the list\
 I need to know why I have do the calls and what will be the outcome, I dont want to perform actions blindly.\
"""

KNOWN_FUNCTIONS: Final[frozenset[str]] = frozenset({
    "left_click", "right_click", "double_left_click", "drag", "type", "explanation",
})

# =============================================================================
# Win32 constants
# =============================================================================

_SM_CXSCREEN: Final[int] = 0
_SM_CYSCREEN: Final[int] = 1
_BI_RGB: Final[int] = 0
_DIB_RGB_COLORS: Final[int] = 0
_SRCCOPY: Final[int] = 0x00CC0020
_CAPTUREBLT: Final[int] = 0x40000000
_HALFTONE: Final[int] = 4
_PROCESS_PER_MONITOR_DPI_AWARE: Final[int] = 2

_MOUSEEVENTF_LEFTDOWN: Final[int] = 0x0002
_MOUSEEVENTF_LEFTUP: Final[int] = 0x0004
_MOUSEEVENTF_RIGHTDOWN: Final[int] = 0x0008
_MOUSEEVENTF_RIGHTUP: Final[int] = 0x0010

_VK_RETURN: Final[int] = 0x0D
_VK_SHIFT: Final[int] = 0x10
_VK_SPACE: Final[int] = 0x20

_KEYEVENTF_KEYUP: Final[int] = 2


# =============================================================================
# Win32 GDI structures
# =============================================================================

class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.wintypes.DWORD),
        ("biWidth", ctypes.wintypes.LONG),
        ("biHeight", ctypes.wintypes.LONG),
        ("biPlanes", ctypes.wintypes.WORD),
        ("biBitCount", ctypes.wintypes.WORD),
        ("biCompression", ctypes.wintypes.DWORD),
        ("biSizeImage", ctypes.wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.wintypes.LONG),
        ("biYPelsPerMeter", ctypes.wintypes.LONG),
        ("biClrUsed", ctypes.wintypes.DWORD),
        ("biClrImportant", ctypes.wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", ctypes.wintypes.DWORD * 3),
    ]


# =============================================================================
# Win32 initialisation (DPI-aware, module-level singletons)
# =============================================================================

_shcore: Final[ctypes.WinDLL] = ctypes.WinDLL("shcore", use_last_error=True)
_shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)

_user32: Final[ctypes.WinDLL] = ctypes.WinDLL("user32", use_last_error=True)
_gdi32: Final[ctypes.WinDLL] = ctypes.WinDLL("gdi32", use_last_error=True)

_screen_w: Final[int] = _user32.GetSystemMetrics(_SM_CXSCREEN)
_screen_h: Final[int] = _user32.GetSystemMetrics(_SM_CYSCREEN)


# =============================================================================
# Screen capture helpers
# =============================================================================

def _make_bmi(w: int, h: int) -> _BITMAPINFO:
    """Create a top-down 32-bit BITMAPINFO."""
    bmi = _BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = _BI_RGB
    return bmi


def _capture_bgra(sw: int, sh: int) -> bytes:
    """Capture the full screen as a BGRA buffer."""
    sdc = _user32.GetDC(0)
    memdc = _gdi32.CreateCompatibleDC(sdc)

    bmi = _make_bmi(sw, sh)
    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(bmi), _DIB_RGB_COLORS, ctypes.byref(bits), None, 0,
    )
    old = _gdi32.SelectObject(memdc, hbmp)
    _gdi32.BitBlt(memdc, 0, 0, sw, sh, sdc, 0, 0, _SRCCOPY | _CAPTUREBLT)

    raw = bytes((ctypes.c_ubyte * (sw * sh * 4)).from_address(bits.value))

    _gdi32.SelectObject(memdc, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(memdc)
    _user32.ReleaseDC(0, sdc)
    return raw


def _downsample_bgra(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Downsample a BGRA bitmap via GDI StretchBlt (HALFTONE)."""
    sdc = _user32.GetDC(0)
    src_dc = _gdi32.CreateCompatibleDC(sdc)
    dst_dc = _gdi32.CreateCompatibleDC(sdc)

    src_bmp = _gdi32.CreateCompatibleBitmap(sdc, sw, sh)
    old_src = _gdi32.SelectObject(src_dc, src_bmp)
    _gdi32.SetDIBits(
        sdc, src_bmp, 0, sh, src, ctypes.byref(_make_bmi(sw, sh)), _DIB_RGB_COLORS,
    )

    dst_bits = ctypes.c_void_p()
    dst_bmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(_make_bmi(dw, dh)), _DIB_RGB_COLORS,
        ctypes.byref(dst_bits), None, 0,
    )
    old_dst = _gdi32.SelectObject(dst_dc, dst_bmp)

    _gdi32.SetStretchBltMode(dst_dc, _HALFTONE)
    _gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)
    _gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, _SRCCOPY)

    raw = bytearray((ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value))
    raw[3::4] = b"\xff" * (dw * dh)
    out = bytes(raw)

    _gdi32.SelectObject(dst_dc, old_dst)
    _gdi32.SelectObject(src_dc, old_src)
    _gdi32.DeleteObject(dst_bmp)
    _gdi32.DeleteObject(src_bmp)
    _gdi32.DeleteDC(dst_dc)
    _gdi32.DeleteDC(src_dc)
    _user32.ReleaseDC(0, sdc)
    return out


def _bgra_to_rgba(bgra: bytes) -> bytes:
    """Convert BGRA to RGBA with alpha forced to 255."""
    n = len(bgra)
    out = bytearray(n)
    out[0::4] = bgra[2::4]
    out[1::4] = bgra[1::4]
    out[2::4] = bgra[0::4]
    out[3::4] = b"\xff" * (n // 4)
    return bytes(out)


def _encode_png(rgba: bytes, w: int, h: int) -> bytes:
    """Minimal PNG encoder for an RGBA buffer."""
    stride = w * 4
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgba[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)

    def _chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def capture_screen_png(
    target_width: int | None = None,
    target_height: int | None = None,
    draw_func: Callable[[bytes, int, int], bytes] | None = None,
) -> bytes:
    """Capture the screen and return PNG bytes.

    Args:
        target_width:  Desired width (``None`` keeps native).
        target_height: Desired height (``None`` keeps native).
        draw_func:     Optional overlay callback ``(rgba, w, h) -> rgba``.
    """
    sw, sh = _screen_w, _screen_h
    bgra = _capture_bgra(sw, sh)

    if (
        target_width is not None
        and target_height is not None
        and (sw, sh) != (target_width, target_height)
    ):
        bgra = _downsample_bgra(bgra, sw, sh, target_width, target_height)
        sw, sh = target_width, target_height

    rgba = _bgra_to_rgba(bgra)
    if draw_func is not None:
        rgba = draw_func(rgba, sw, sh)
    return _encode_png(rgba, sw, sh)


# =============================================================================
# Drawing -- colour palette
# =============================================================================

ACTION_PRIMARY: Final[Color] = (255, 50, 200, 255)
ACTION_SECONDARY: Final[Color] = (255, 180, 240, 255)
ACTION_OUTLINE: Final[Color] = (40, 0, 30, 200)


# =============================================================================
# Drawing -- pixel helpers
# =============================================================================

def _set_pixel(
    data: bytearray, w: int, h: int, x: int, y: int, c: Color,
) -> None:
    if 0 <= x < w and 0 <= y < h:
        i = (y * w + x) << 2
        data[i] = c[0]
        data[i + 1] = c[1]
        data[i + 2] = c[2]
        data[i + 3] = c[3]


def _set_pixel_thick(
    data: bytearray, w: int, h: int, x: int, y: int, c: Color, t: int = 1,
) -> None:
    half = t >> 1
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            _set_pixel(data, w, h, x + dx, y + dy, c)


# =============================================================================
# Drawing -- primitives
# =============================================================================

def _draw_line(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int,
    c: Color, t: int = 3,
) -> None:
    """Bresenham line, in-place."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1
    while True:
        _set_pixel_thick(d, w, h, x, y, c, t)
        if x == x2 and y == y2:
            break
        e2 = err << 1
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _draw_dashed_line(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int,
    c: Color, t: int = 2, dash: int = 8, gap: int = 5,
) -> None:
    dx, dy = x2 - x1, y2 - y1
    dist = max(1, int(math.hypot(dx, dy)))
    cycle = dash + gap
    for i in range(dist + 1):
        if (i % cycle) < dash:
            frac = i / dist
            _set_pixel_thick(d, w, h, int(x1 + dx * frac), int(y1 + dy * frac), c, t)


def _draw_circle(
    d: bytearray, w: int, h: int,
    cx: int, cy: int, r: int, c: Color, filled: bool = False,
) -> None:
    r2 = r * r
    inner2 = (r - 2) ** 2
    for oy in range(-r, r + 1):
        for ox in range(-r, r + 1):
            dist2 = ox * ox + oy * oy
            if filled:
                if dist2 <= r2:
                    _set_pixel(d, w, h, cx + ox, cy + oy, c)
            elif inner2 <= dist2 <= r2:
                _set_pixel(d, w, h, cx + ox, cy + oy, c)


def _fill_triangle(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, c: Color,
) -> None:
    lo_x = max(0, min(x1, x2, x3))
    hi_x = min(w - 1, max(x1, x2, x3))
    lo_y = max(0, min(y1, y2, y3))
    hi_y = min(h - 1, max(y1, y2, y3))

    def _edge(px: int, py: int, ax: int, ay: int, bx: int, by: int) -> int:
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    for py in range(lo_y, hi_y + 1):
        for px in range(lo_x, hi_x + 1):
            d1 = _edge(px, py, x1, y1, x2, y2)
            d2 = _edge(px, py, x2, y2, x3, y3)
            d3 = _edge(px, py, x3, y3, x1, y1)
            if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                _set_pixel(d, w, h, px, py, c)


def _draw_arrowhead(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int,
    c: Color, t: int = 3, length: int = 15, angle_deg: float = 30.0,
) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    ha = math.radians(angle_deg)
    lx = int(x2 - length * math.cos(angle - ha))
    ly = int(y2 - length * math.sin(angle - ha))
    rx = int(x2 - length * math.cos(angle + ha))
    ry = int(y2 - length * math.sin(angle + ha))
    _draw_line(d, w, h, x2, y2, lx, ly, c, t)
    _draw_line(d, w, h, x2, y2, rx, ry, c, t)
    _fill_triangle(d, w, h, x2, y2, lx, ly, rx, ry, c)


def _draw_dashed_arrow(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int,
    c: Color, t: int = 2, dash: int = 8, gap: int = 5,
    head_len: int = 15, head_deg: float = 30.0,
) -> None:
    _draw_dashed_line(d, w, h, x1, y1, x2, y2, c, t, dash, gap)
    _draw_arrowhead(d, w, h, x1, y1, x2, y2, c, max(t, 3), head_len, head_deg)


# =============================================================================
# Drawing -- glyph data
# =============================================================================

_GLYPH_CURSOR: Final[list[str]] = [
    "#           ",
    "##          ",
    "#.#         ",
    "#..#        ",
    "#...#       ",
    "#....#      ",
    "#.....#     ",
    "#......#    ",
    "#.......#   ",
    "#........#  ",
    "#.....#####",
    "#..#..#     ",
    "#.# #..#    ",
    "##  #..#    ",
    "#    #..#   ",
    "     ###    ",
]

_GLYPH_CURSOR_RIGHT: Final[list[str]] = [
    "#           ",
    "##          ",
    "#.#         ",
    "#..#        ",
    "#...#       ",
    "#....#      ",
    "#.....#     ",
    "#......#    ",
    "#.......#   ",
    "#........#  ",
    "#.....#####",
    "#..#..# ##  ",
    "#.# #..##.# ",
    "##  #..### ",
    "#    #..#   ",
    "     ###    ",
]

_GLYPH_IBEAM: Final[list[str]] = [
    " ### ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    " ### ",
]

_IBEAM_W: Final[int] = len(_GLYPH_IBEAM[0])
_IBEAM_H: Final[int] = len(_GLYPH_IBEAM)


# =============================================================================
# Drawing -- glyph / burst renderers
# =============================================================================

def _draw_glyph(
    d: bytearray, w: int, h: int,
    x: int, y: int, glyph: list[str],
    primary: Color, outline: Color, scale: int = 1,
) -> None:
    for ri, row in enumerate(glyph):
        for ci, ch in enumerate(row):
            if ch == " ":
                continue
            clr = primary if ch == "#" else outline
            for sy in range(scale):
                for sx in range(scale):
                    _set_pixel(d, w, h, x + ci * scale + sx, y + ri * scale + sy, clr)


def _draw_burst(
    d: bytearray, w: int, h: int,
    x: int, y: int, c: Color,
    r_in: int = 12, r_out: int = 22, rays: int = 8, t: int = 2,
) -> None:
    for i in range(rays):
        a = (2.0 * math.pi * i) / rays
        cos_a, sin_a = math.cos(a), math.sin(a)
        _draw_line(
            d, w, h,
            int(x + r_in * cos_a), int(y + r_in * sin_a),
            int(x + r_out * cos_a), int(y + r_out * sin_a),
            c, t,
        )


# =============================================================================
# Drawing -- annotation composites
# =============================================================================

def _movement_trail(
    d: bytearray, w: int, h: int,
    x: int, y: int, px: int | None, py: int | None,
) -> None:
    if px is not None and py is not None and math.hypot(x - px, y - py) > 20:
        _draw_dashed_arrow(
            d, w, h, px, py, x, y,
            ACTION_SECONDARY, t=2, dash=6, gap=4, head_len=12,
        )


def _annotate_left_click(
    d: bytearray, w: int, h: int,
    x: int, y: int, px: int | None, py: int | None,
) -> None:
    _movement_trail(d, w, h, x, y, px, py)
    _draw_burst(d, w, h, x, y, ACTION_PRIMARY, 14, 24, 8, 2)
    _draw_glyph(d, w, h, x, y, _GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE)


def _annotate_right_click(
    d: bytearray, w: int, h: int,
    x: int, y: int, px: int | None, py: int | None,
) -> None:
    _movement_trail(d, w, h, x, y, px, py)
    p = 20
    _draw_line(d, w, h, x - p, y - p, x + p, y - p, ACTION_PRIMARY, 2)
    _draw_line(d, w, h, x + p, y - p, x + p, y + p, ACTION_PRIMARY, 2)
    _draw_line(d, w, h, x + p, y + p, x - p, y + p, ACTION_PRIMARY, 2)
    _draw_line(d, w, h, x - p, y + p, x - p, y - p, ACTION_PRIMARY, 2)
    _draw_glyph(d, w, h, x, y, _GLYPH_CURSOR_RIGHT, ACTION_PRIMARY, ACTION_OUTLINE)


def _annotate_double_click(
    d: bytearray, w: int, h: int,
    x: int, y: int, px: int | None, py: int | None,
) -> None:
    _movement_trail(d, w, h, x, y, px, py)
    _draw_circle(d, w, h, x, y, 18, ACTION_PRIMARY)
    _draw_circle(d, w, h, x, y, 28, ACTION_PRIMARY)
    _draw_burst(d, w, h, x, y, ACTION_PRIMARY, 30, 38, 8, 2)
    _draw_glyph(d, w, h, x, y, _GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE)


def _annotate_drag(
    d: bytearray, w: int, h: int,
    x1: int, y1: int, x2: int, y2: int,
    px: int | None, py: int | None,
) -> None:
    if px is not None and py is not None and math.hypot(x1 - px, y1 - py) > 20:
        _draw_dashed_arrow(
            d, w, h, px, py, x1, y1,
            ACTION_SECONDARY, t=1, dash=4, gap=4, head_len=8,
        )
    _draw_circle(d, w, h, x1, y1, 8, ACTION_PRIMARY, filled=True)
    _draw_dashed_arrow(
        d, w, h, x1, y1, x2, y2,
        ACTION_PRIMARY, t=3, dash=10, gap=6, head_len=18, head_deg=25.0,
    )
    _draw_circle(d, w, h, x2, y2, 10, ACTION_PRIMARY)


def _annotate_type(d: bytearray, w: int, h: int, x: int, y: int) -> None:
    _draw_glyph(
        d, w, h,
        x - (_IBEAM_W * 2) // 2, y - (_IBEAM_H * 2) // 2,
        _GLYPH_IBEAM, ACTION_PRIMARY, ACTION_OUTLINE, scale=2,
    )
    _draw_line(d, w, h, x - 15, y + _IBEAM_H + 4, x + 15, y + _IBEAM_H + 4, ACTION_PRIMARY, 2)


def _norm(coord: int, extent: int) -> int:
    """Convert normalised coordinate (0-1000) to pixel coordinate."""
    return int((coord / 1000.0) * extent)


# =============================================================================
# Input simulation
# =============================================================================

def _to_px(val: int, dim: int) -> int:
    return int(val / 1000 * dim)


def _cursor_pos() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _smooth_move(tx: int, ty: int) -> None:
    sx, sy = _cursor_pos()
    dx, dy = tx - sx, ty - sy
    for i in range(MOVE_STEPS + 1):
        t = i / MOVE_STEPS
        t = t * t * (3.0 - 2.0 * t)
        _user32.SetCursorPos(int(sx + dx * t), int(sy + dy * t))
        time.sleep(STEP_DELAY)


def _mouse_click(down: int, up: int) -> None:
    _user32.mouse_event(down, 0, 0, 0, 0)
    time.sleep(0.05)
    _user32.mouse_event(up, 0, 0, 0, 0)


def _key_tap(vk: int) -> None:
    _user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


def _type_text(text: str) -> None:
    for ch in text:
        if ch == " ":
            _key_tap(_VK_SPACE)
            time.sleep(WORD_DELAY)
        elif ch == "\n":
            _key_tap(_VK_RETURN)
            time.sleep(WORD_DELAY)
        else:
            vk = _user32.VkKeyScanW(ord(ch))
            if vk == -1:
                continue
            need_shift = bool((vk >> 8) & 1)
            if need_shift:
                _user32.keybd_event(_VK_SHIFT, 0, 0, 0)
                time.sleep(0.01)
            _key_tap(vk & 0xFF)
            if need_shift:
                _user32.keybd_event(_VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
            time.sleep(CHAR_DELAY)


# =============================================================================
# Agent action functions
# =============================================================================

story: str = ""


def left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_click(_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP)


def right_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_click(_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP)


def double_left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_click(_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP)
    time.sleep(0.08)
    _mouse_click(_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP)


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    _smooth_move(_to_px(x1, _screen_w), _to_px(y1, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.1)
    _smooth_move(_to_px(x2, _screen_w), _to_px(y2, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def type_text_action(text: str) -> None:
    """VLM-facing ``type()`` implementation."""
    _type_text(text)


def explanation(text: str) -> None:
    global story
    story = text


DISPATCH: Final[dict[str, Callable[..., None]]] = {
    "left_click": left_click,
    "right_click": right_click,
    "double_left_click": double_left_click,
    "drag": drag,
    "type": type_text_action,
    "explanation": explanation,
}


# =============================================================================
# VLM output parsing / execution
# =============================================================================

def _run_output(raw: str) -> list[str]:
    """Parse VLM response and execute recognised function calls."""
    executed: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        paren = stripped.find("(")
        if paren == -1:
            continue
        name = stripped[:paren].strip()
        if name not in KNOWN_FUNCTIONS:
            continue
        try:
            eval(stripped, {"__builtins__": {}}, DISPATCH)  # noqa: S307
            executed.append(stripped)
        except Exception:
            pass
    return executed


# =============================================================================
# Overlay builder
# =============================================================================

def _make_overlay(actions: list[str]) -> Callable[[bytes, int, int], bytes]:
    """Return a draw callback that annotates RGBA data with *actions*."""

    def _draw(rgba: bytes, w: int, h: int) -> bytes:
        if not VISUAL_MARKS:
            return rgba

        buf = bytearray(rgba)
        px: int | None = None
        py: int | None = None

        for line in actions:
            paren = line.find("(")
            if paren == -1:
                continue
            name = line[:paren].strip()
            try:
                args: list[object] = eval(  # noqa: S307
                    f"[{line[paren + 1 : line.rfind(')')]}]",
                    {"__builtins__": {}},
                    {},
                )
            except Exception:
                continue

            match name:
                case "left_click" if len(args) >= 2:
                    x, y = _norm(int(args[0]), w), _norm(int(args[1]), h)  # type: ignore[arg-type]
                    _annotate_left_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "right_click" if len(args) >= 2:
                    x, y = _norm(int(args[0]), w), _norm(int(args[1]), h)  # type: ignore[arg-type]
                    _annotate_right_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "double_left_click" if len(args) >= 2:
                    x, y = _norm(int(args[0]), w), _norm(int(args[1]), h)  # type: ignore[arg-type]
                    _annotate_double_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "drag" if len(args) >= 4:
                    x1 = _norm(int(args[0]), w)  # type: ignore[arg-type]
                    y1 = _norm(int(args[1]), h)  # type: ignore[arg-type]
                    x2 = _norm(int(args[2]), w)  # type: ignore[arg-type]
                    y2 = _norm(int(args[3]), h)  # type: ignore[arg-type]
                    _annotate_drag(buf, w, h, x1, y1, x2, y2, px, py)
                    px, py = x2, y2
                case "type":
                    if px is not None and py is not None:
                        _annotate_type(buf, w, h, px, py)

        return bytes(buf)

    return _draw


# =============================================================================
# VLM response source (live API or test fixture iterator)
# =============================================================================

def _load_test_responses(paths: list[Path]) -> Iterator[str]:
    """Yield the VLM content string from each JSON fixture file.

    Each file is expected to have the exact same structure as a real API
    response, so the content is extracted from the same JSON path:
    ``["choices"][0]["message"]["content"]``
    """
    for path in paths:
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        yield data["choices"][0]["message"]["content"]  # type: ignore[index,return-value]


def _infer_live(png: bytes, current_story: str) -> str:
    """Send screenshot + story to the VLM; return the raw text response."""
    b64 = base64.b64encode(png).decode("ascii")
    payload: dict[str, object] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_TEMPLATE.format(story=current_story),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        **SAMPLING,
    }

    req = urllib.request.Request(
        API, json.dumps(payload).encode(), {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        body: dict[str, object] = json.load(resp)
        return body["choices"][0]["message"]["content"]  # type: ignore[index,return-value]


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:
    global story

    # ---- determine mode: test (JSON fixtures) or live (API) -----------------
    test_files = [Path(arg) for arg in sys.argv[1:]]
    test_mode = len(test_files) > 0

    if test_mode:
        for path in test_files:
            if not path.is_file():
                print(f"  ERROR: test fixture not found: {path}", file=sys.stderr)
                sys.exit(1)
        test_responses: Iterator[str] = _load_test_responses(test_files)
        print(f"  TEST MODE: {len(test_files)} fixture(s) loaded")
    else:
        test_responses = iter([])  # exhausted immediately, never used

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

        img = capture_screen_png(WIDTH, HEIGHT, draw_func=_make_overlay(prev))
        (dump / f"{int(time.time() * 1000)}.png").write_bytes(img)

        user_text = USER_TEMPLATE.format(story=story)

        print(f"\n{'=' * 60}")
        print(f"  turn {turn}")
        print(f"{'=' * 60}")
        print("  sending:")
        for line in user_text.splitlines():
            print(f"    | {line}")
        print("  thinking...")

        # -- THIS IS THE SINGLE INJECTION POINT ------------------------------
        # In live mode: call the API and extract ["choices"][0]["message"]["content"]
        # In test mode: read the next JSON fixture and extract the SAME path
        # Both produce the identical str that the rest of the pipeline consumes.
        if test_mode:
            raw_or_none = next(test_responses, None)
            if raw_or_none is None:
                print(f"  TEST MODE: all {len(test_files)} fixture(s) consumed -- stopping.")
                break
            raw: str = raw_or_none
            print(f"  TEST MODE: using fixture {turn}/{len(test_files)}")
        else:
            raw = _infer_live(img, story)
        # ---------------------------------------------------------------------

        print(f"  vlm ({len(raw)} chars):")
        print(f"{'-' * 40}")
        print(raw)
        print(f"{'-' * 40}")

        executed = _run_output(raw)

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
            "test_mode": test_mode,
        }
        (dump / "state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (dump / "story.txt").write_text(story, encoding="utf-8")
        Path("story.txt").write_text(story, encoding="utf-8")

        print(f"  rest {LOOP_DELAY}s...")
        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    main()
