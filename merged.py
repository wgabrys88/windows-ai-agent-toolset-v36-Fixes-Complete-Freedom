"""
FRANZ - Autonomous screen agent.

Single-file implementation combining screen capture (GDI), drawing annotations,
and the main agent loop. Windows 11 / Python 3.12+.
"""

import base64
import ctypes
import ctypes.wintypes
import json
import math
import struct
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
from typing import Callable, Final


# =============================================================================
# Configuration
# =============================================================================

API: Final = "http://localhost:1234/v1/chat/completions"
MODEL: Final = "qwen3-vl-2b-instruct-1m"
WIDTH: Final = 736
HEIGHT: Final = 464
SAMPLING: Final[dict[str, float | int]] = {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1500}
MAX_ACTIONS: Final = 3
VISUAL_MARKS: Final = True

MOVE_STEPS: Final = 20
STEP_DELAY: Final = 0.01
CLICK_DELAY: Final = 0.15
CHAR_DELAY: Final = 0.08
WORD_DELAY: Final = 0.15
ACTION_GAP: Final = 0.3
LOOP_DELAY: Final = 1.0

SYSTEM_PROMPT: Final = """\
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
 Provide me python function calls with a mandatory explanation function at the END of the list\
 I need to know why I have do the calls and what will be the outcome, I dont want to perform actions blindly.\
"""

KNOWN_FUNCTIONS: Final = frozenset({
    "left_click", "right_click", "double_left_click", "drag", "type", "explanation",
})


# =============================================================================
# Win32 GDI constants and structures for screen capture
# =============================================================================

SM_CXSCREEN: Final = 0
SM_CYSCREEN: Final = 1
BI_RGB: Final = 0
DIB_RGB_COLORS: Final = 0
SRCCOPY: Final = 0x00CC0020
CAPTUREBLT: Final = 0x40000000
HALFTONE: Final = 4
PROCESS_PER_MONITOR_DPI_AWARE: Final = 2


class BITMAPINFOHEADER(ctypes.Structure):
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


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.wintypes.DWORD * 3),
    ]


# =============================================================================
# Win32 initialization (DPI-aware, single instance)
# =============================================================================

_shcore = ctypes.WinDLL("shcore", use_last_error=True)
_shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

_screen_w: Final = _user32.GetSystemMetrics(SM_CXSCREEN)
_screen_h: Final = _user32.GetSystemMetrics(SM_CYSCREEN)


# =============================================================================
# Screen capture
# =============================================================================

def _capture_bgra(sw: int, sh: int) -> bytes:
    """Capture screen bitmap in BGRA format."""
    sdc = _user32.GetDC(0)
    memdc = _gdi32.CreateCompatibleDC(sdc)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0,
    )
    old = _gdi32.SelectObject(memdc, hbmp)
    _gdi32.BitBlt(memdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)

    raw = bytes((ctypes.c_ubyte * (sw * sh * 4)).from_address(bits.value))

    _gdi32.SelectObject(memdc, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(memdc)
    _user32.ReleaseDC(0, sdc)
    return raw


def _downsample_bgra(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Downsample BGRA bitmap using GDI StretchBlt with HALFTONE filtering."""
    sdc = _user32.GetDC(0)
    src_dc = _gdi32.CreateCompatibleDC(sdc)
    dst_dc = _gdi32.CreateCompatibleDC(sdc)

    src_bmp = _gdi32.CreateCompatibleBitmap(sdc, sw, sh)
    old_src = _gdi32.SelectObject(src_dc, src_bmp)

    bmi_temp = BITMAPINFO()
    bmi_temp.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_temp.bmiHeader.biWidth = sw
    bmi_temp.bmiHeader.biHeight = -sh
    bmi_temp.bmiHeader.biPlanes = 1
    bmi_temp.bmiHeader.biBitCount = 32
    bmi_temp.bmiHeader.biCompression = BI_RGB
    _gdi32.SetDIBits(sdc, src_bmp, 0, sh, src, ctypes.byref(bmi_temp), DIB_RGB_COLORS)

    bmi_dst = BITMAPINFO()
    bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_dst.bmiHeader.biWidth = dw
    bmi_dst.bmiHeader.biHeight = -dh
    bmi_dst.bmiHeader.biPlanes = 1
    bmi_dst.bmiHeader.biBitCount = 32
    bmi_dst.bmiHeader.biCompression = BI_RGB

    dst_bits = ctypes.c_void_p()
    dst_bmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(bmi_dst), DIB_RGB_COLORS, ctypes.byref(dst_bits), None, 0,
    )
    old_dst = _gdi32.SelectObject(dst_dc, dst_bmp)

    _gdi32.SetStretchBltMode(dst_dc, HALFTONE)
    _gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)
    _gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY)

    raw = bytearray((ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value))
    raw[3::4] = bytes([255] * (dw * dh))
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
    out = bytearray(len(bgra))
    out[0::4] = bgra[2::4]
    out[1::4] = bgra[1::4]
    out[2::4] = bgra[0::4]
    out[3::4] = bytes([255] * (len(bgra) // 4))
    return bytes(out)


def _encode_png(rgba: bytes, w: int, h: int) -> bytes:
    """Encode RGBA buffer to PNG."""
    raw = bytearray()
    stride = w * 4
    for y in range(h):
        raw.append(0)
        raw.extend(rgba[y * stride:(y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def capture_screen_png(
    target_width: int | None = None,
    target_height: int | None = None,
    draw_func: Callable[[bytes, int, int], bytes] | None = None,
) -> bytes:
    """Capture the screen and return a PNG image.

    Args:
        target_width:  Desired width (None keeps native).
        target_height: Desired height (None keeps native).
        draw_func:     Optional overlay callback ``(rgba, w, h) -> rgba``.

    Returns:
        PNG file bytes.
    """
    sw, sh = _screen_w, _screen_h
    bgra = _capture_bgra(sw, sh)

    if target_width is not None and target_height is not None and (sw, sh) != (target_width, target_height):
        bgra = _downsample_bgra(bgra, sw, sh, target_width, target_height)
        sw, sh = target_width, target_height

    rgba = _bgra_to_rgba(bgra)

    if draw_func is not None:
        rgba = draw_func(rgba, sw, sh)

    return _encode_png(rgba, sw, sh)


# =============================================================================
# Drawing -- color constants
# =============================================================================

type Color = tuple[int, int, int, int]

ACTION_PRIMARY: Final[Color] = (255, 50, 200, 255)
ACTION_SECONDARY: Final[Color] = (255, 180, 240, 255)
ACTION_OUTLINE: Final[Color] = (40, 0, 30, 200)


# =============================================================================
# Drawing -- pixel helpers
# =============================================================================

def _set_pixel(
    data: bytearray, width: int, height: int,
    x: int, y: int, color: Color,
) -> None:
    """Set a single RGBA pixel (bounds-checked)."""
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 4
        data[idx] = color[0]
        data[idx + 1] = color[1]
        data[idx + 2] = color[2]
        data[idx + 3] = color[3]


def _set_pixel_thick(
    data: bytearray, width: int, height: int,
    x: int, y: int, color: Color,
    thickness: int = 1,
) -> None:
    """Set a thick pixel (square brush)."""
    half = thickness // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            _set_pixel(data, width, height, x + dx, y + dy, color)


# =============================================================================
# Drawing -- core primitives
# =============================================================================

def _draw_line_mut(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int,
    color: Color, thickness: int = 3,
) -> None:
    """Bresenham line, in-place."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1

    while True:
        _set_pixel_thick(data, width, height, x, y, color, thickness)
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _draw_dashed_line_mut(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int,
    color: Color, thickness: int = 2,
    dash_len: int = 8, gap_len: int = 5,
) -> None:
    """Dashed line, in-place."""
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1, int(math.hypot(dx, dy)))
    cycle = dash_len + gap_len
    for i in range(dist + 1):
        if (i % cycle) < dash_len:
            t = i / dist
            _set_pixel_thick(
                data, width, height,
                int(x1 + dx * t), int(y1 + dy * t),
                color, thickness,
            )


def _draw_circle_mut(
    data: bytearray, width: int, height: int,
    x: int, y: int, radius: int,
    color: Color, filled: bool = False,
) -> None:
    """Circle (filled or hollow), in-place."""
    for cy in range(-radius, radius + 1):
        for cx in range(-radius, radius + 1):
            dist_sq = cx * cx + cy * cy
            if filled:
                if dist_sq <= radius * radius:
                    _set_pixel(data, width, height, x + cx, y + cy, color)
            else:
                if (radius - 2) ** 2 <= dist_sq <= radius * radius:
                    _set_pixel(data, width, height, x + cx, y + cy, color)


def _fill_triangle_mut(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int, x3: int, y3: int,
    color: Color,
) -> None:
    """Scanline triangle fill, in-place."""
    min_x = max(0, min(x1, x2, x3))
    max_x = min(width - 1, max(x1, x2, x3))
    min_y = max(0, min(y1, y2, y3))
    max_y = min(height - 1, max(y1, y2, y3))

    def _sign(px: int, py: int, ax: int, ay: int, bx: int, by: int) -> int:
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            d1 = _sign(px, py, x1, y1, x2, y2)
            d2 = _sign(px, py, x2, y2, x3, y3)
            d3 = _sign(px, py, x3, y3, x1, y1)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            if not (has_neg and has_pos):
                _set_pixel(data, width, height, px, py, color)


def _draw_arrowhead_mut(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int,
    color: Color, thickness: int = 3,
    head_length: int = 15, head_angle_deg: float = 30.0,
) -> None:
    """Solid filled arrowhead at (x2, y2)."""
    angle = math.atan2(y2 - y1, x2 - x1)
    head_angle = math.radians(head_angle_deg)

    lx = int(x2 - head_length * math.cos(angle - head_angle))
    ly = int(y2 - head_length * math.sin(angle - head_angle))
    _draw_line_mut(data, width, height, x2, y2, lx, ly, color, thickness)

    rx = int(x2 - head_length * math.cos(angle + head_angle))
    ry = int(y2 - head_length * math.sin(angle + head_angle))
    _draw_line_mut(data, width, height, x2, y2, rx, ry, color, thickness)

    _fill_triangle_mut(data, width, height, x2, y2, lx, ly, rx, ry, color)


def _draw_dashed_arrow_mut(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int,
    color: Color, thickness: int = 2,
    dash_len: int = 8, gap_len: int = 5,
    head_length: int = 15, head_angle_deg: float = 30.0,
) -> None:
    """Dashed line + solid arrowhead, in-place."""
    _draw_dashed_line_mut(data, width, height, x1, y1, x2, y2, color, thickness, dash_len, gap_len)
    _draw_arrowhead_mut(
        data, width, height, x1, y1, x2, y2,
        color, max(thickness, 3), head_length, head_angle_deg,
    )


# =============================================================================
# Drawing -- glyph rendering
# =============================================================================

GLYPH_CURSOR: Final[list[str]] = [
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

GLYPH_CURSOR_RIGHT: Final[list[str]] = [
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

GLYPH_IBEAM: Final[list[str]] = [
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


def _draw_glyph_mut(
    data: bytearray, width: int, height: int,
    x: int, y: int, glyph: list[str],
    primary: Color, outline: Color, scale: int = 1,
) -> None:
    """Render a pixel-art glyph. '#' = primary, '.' = outline, ' ' = skip."""
    for row_idx, row in enumerate(glyph):
        for col_idx, ch in enumerate(row):
            if ch == " ":
                continue
            color = primary if ch == "#" else outline
            for sy in range(scale):
                for sx in range(scale):
                    _set_pixel(
                        data, width, height,
                        x + col_idx * scale + sx,
                        y + row_idx * scale + sy,
                        color,
                    )


def _draw_burst_mut(
    data: bytearray, width: int, height: int,
    x: int, y: int, color: Color,
    inner_radius: int = 12, outer_radius: int = 22,
    num_rays: int = 8, thickness: int = 2,
) -> None:
    """Radiating burst lines around a point (impact indicator)."""
    for i in range(num_rays):
        angle = (2 * math.pi * i) / num_rays
        ix = int(x + inner_radius * math.cos(angle))
        iy = int(y + inner_radius * math.sin(angle))
        ox = int(x + outer_radius * math.cos(angle))
        oy = int(y + outer_radius * math.sin(angle))
        _draw_line_mut(data, width, height, ix, iy, ox, oy, color, thickness)


# =============================================================================
# Drawing -- high-level annotation composites
# =============================================================================

def _draw_movement_trail(
    data: bytearray, width: int, height: int,
    x: int, y: int,
    prev_x: int | None, prev_y: int | None,
) -> None:
    """Draw a dashed movement trail from the previous position if far enough."""
    if prev_x is not None and prev_y is not None:
        if math.hypot(x - prev_x, y - prev_y) > 20:
            _draw_dashed_arrow_mut(
                data, width, height, prev_x, prev_y, x, y,
                ACTION_SECONDARY, thickness=2,
                dash_len=6, gap_len=4, head_length=12,
            )


def annotate_left_click(
    data: bytearray, width: int, height: int,
    x: int, y: int,
    prev_x: int | None = None, prev_y: int | None = None,
) -> None:
    """Left-click: movement trail + burst + cursor pointer."""
    _draw_movement_trail(data, width, height, x, y, prev_x, prev_y)
    _draw_burst_mut(data, width, height, x, y, ACTION_PRIMARY, 14, 24, 8, 2)
    _draw_glyph_mut(data, width, height, x, y, GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE)


def annotate_right_click(
    data: bytearray, width: int, height: int,
    x: int, y: int,
    prev_x: int | None = None, prev_y: int | None = None,
) -> None:
    """Right-click: movement trail + square burst + right-cursor pointer."""
    _draw_movement_trail(data, width, height, x, y, prev_x, prev_y)
    pad = 20
    _draw_line_mut(data, width, height, x - pad, y - pad, x + pad, y - pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height, x + pad, y - pad, x + pad, y + pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height, x + pad, y + pad, x - pad, y + pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height, x - pad, y + pad, x - pad, y - pad, ACTION_PRIMARY, 2)
    _draw_glyph_mut(data, width, height, x, y, GLYPH_CURSOR_RIGHT, ACTION_PRIMARY, ACTION_OUTLINE)


def annotate_double_click(
    data: bytearray, width: int, height: int,
    x: int, y: int,
    prev_x: int | None = None, prev_y: int | None = None,
) -> None:
    """Double-click: movement trail + concentric rings + burst + cursor pointer."""
    _draw_movement_trail(data, width, height, x, y, prev_x, prev_y)
    _draw_circle_mut(data, width, height, x, y, 18, ACTION_PRIMARY, filled=False)
    _draw_circle_mut(data, width, height, x, y, 28, ACTION_PRIMARY, filled=False)
    _draw_burst_mut(data, width, height, x, y, ACTION_PRIMARY, 30, 38, 8, 2)
    _draw_glyph_mut(data, width, height, x, y, GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE)


def annotate_drag(
    data: bytearray, width: int, height: int,
    x1: int, y1: int, x2: int, y2: int,
    prev_x: int | None = None, prev_y: int | None = None,
) -> None:
    """Drag: approach trail + grip dot + dashed drag arrow + release ring."""
    if prev_x is not None and prev_y is not None:
        if math.hypot(x1 - prev_x, y1 - prev_y) > 20:
            _draw_dashed_arrow_mut(
                data, width, height, prev_x, prev_y, x1, y1,
                ACTION_SECONDARY, thickness=1,
                dash_len=4, gap_len=4, head_length=8,
            )
    _draw_circle_mut(data, width, height, x1, y1, 8, ACTION_PRIMARY, filled=True)
    _draw_dashed_arrow_mut(
        data, width, height, x1, y1, x2, y2,
        ACTION_PRIMARY, thickness=3,
        dash_len=10, gap_len=6, head_length=18, head_angle_deg=25.0,
    )
    _draw_circle_mut(data, width, height, x2, y2, 10, ACTION_PRIMARY, filled=False)


def annotate_type(
    data: bytearray, width: int, height: int,
    x: int, y: int,
) -> None:
    """Typing: I-beam glyph + underline bar."""
    glyph_w = len(GLYPH_IBEAM[0])
    glyph_h = len(GLYPH_IBEAM)
    _draw_glyph_mut(
        data, width, height,
        x - (glyph_w * 2) // 2, y - (glyph_h * 2) // 2,
        GLYPH_IBEAM, ACTION_PRIMARY, ACTION_OUTLINE, scale=2,
    )
    bar_y = y + glyph_h + 4
    _draw_line_mut(data, width, height, x - 15, bar_y, x + 15, bar_y, ACTION_PRIMARY, 2)


def normalize_coord(coord: int, max_val: int) -> int:
    """Convert normalised coordinate (0-1000) to pixel coordinate."""
    return int((coord / 1000.0) * max_val)


# =============================================================================
# Input simulation (Windows)
# =============================================================================

def _to_px(val: int, dim: int) -> int:
    return int(val / 1000 * dim)


def _cursor() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _smooth_move(tx: int, ty: int) -> None:
    sx, sy = _cursor()
    for i in range(MOVE_STEPS + 1):
        t = i / MOVE_STEPS
        t = t * t * (3 - 2 * t)
        _user32.SetCursorPos(int(sx + (tx - sx) * t), int(sy + (ty - sy) * t))
        time.sleep(STEP_DELAY)


def _mouse_event(down: int, up: int) -> None:
    _user32.mouse_event(down, 0, 0, 0, 0)
    time.sleep(0.05)
    _user32.mouse_event(up, 0, 0, 0, 0)


def _key_press(vk: int) -> None:
    _user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(vk, 0, 2, 0)


def _type_text(text: str) -> None:
    for ch in text:
        if ch == " ":
            _key_press(0x20)
            time.sleep(WORD_DELAY)
        elif ch == "\n":
            _key_press(0x0D)
            time.sleep(WORD_DELAY)
        else:
            vk = _user32.VkKeyScanW(ord(ch))
            if vk == -1:
                continue
            shift = (vk >> 8) & 1
            if shift:
                _user32.keybd_event(0x10, 0, 0, 0)
                time.sleep(0.01)
            _key_press(vk & 0xFF)
            if shift:
                _user32.keybd_event(0x10, 0, 2, 0)
            time.sleep(CHAR_DELAY)


# =============================================================================
# Agent action functions
# =============================================================================

story: str = ""


def left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_event(0x0002, 0x0004)


def right_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_event(0x0008, 0x0010)


def double_left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(CLICK_DELAY)
    _mouse_event(0x0002, 0x0004)
    time.sleep(0.08)
    _mouse_event(0x0002, 0x0004)


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    _smooth_move(_to_px(x1, _screen_w), _to_px(y1, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.1)
    _smooth_move(_to_px(x2, _screen_w), _to_px(y2, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(0x0004, 0, 0, 0, 0)


def type_text_action(text: str) -> None:
    """Wrapper exposed to the VLM as ``type``."""
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
# VLM output parsing and execution
# =============================================================================

def run_output(raw: str) -> list[str]:
    """Parse VLM response lines and execute recognised function calls."""
    executed: list[str] = []
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
            eval(line, {"__builtins__": {}}, DISPATCH)  # noqa: S307
            executed.append(line)
        except Exception:
            pass
    return executed


# =============================================================================
# Overlay builder
# =============================================================================

def make_overlay(actions: list[str]) -> Callable[[bytes, int, int], bytes]:
    """Return a draw callback that annotates RGBA data with *actions*."""

    def draw(rgba: bytes, w: int, h: int) -> bytes:
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
                args = eval(  # noqa: S307
                    f"[{line[paren + 1:line.rfind(')')]}]",
                    {"__builtins__": {}}, {},
                )
            except Exception:
                continue

            match name:
                case "left_click" if len(args) >= 2:
                    x = normalize_coord(int(args[0]), w)
                    y = normalize_coord(int(args[1]), h)
                    annotate_left_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "right_click" if len(args) >= 2:
                    x = normalize_coord(int(args[0]), w)
                    y = normalize_coord(int(args[1]), h)
                    annotate_right_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "double_left_click" if len(args) >= 2:
                    x = normalize_coord(int(args[0]), w)
                    y = normalize_coord(int(args[1]), h)
                    annotate_double_click(buf, w, h, x, y, px, py)
                    px, py = x, y
                case "drag" if len(args) >= 4:
                    x1 = normalize_coord(int(args[0]), w)
                    y1 = normalize_coord(int(args[1]), h)
                    x2 = normalize_coord(int(args[2]), w)
                    y2 = normalize_coord(int(args[3]), h)
                    annotate_drag(buf, w, h, x1, y1, x2, y2, px, py)
                    px, py = x2, y2
                case "type":
                    if px is not None:
                        annotate_type(buf, w, h, px, py)  # type: ignore[arg-type]
        return bytes(buf)

    return draw


# =============================================================================
# LLM inference
# =============================================================================

def infer(png: bytes, current_story: str) -> str:
    """Send screenshot + story to the VLM and return the raw text response."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEMPLATE.format(story=current_story)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(png).decode()}",
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
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.load(resp)["choices"][0]["message"]["content"]


# =============================================================================
# Main loop
# =============================================================================

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

        img = capture_screen_png(WIDTH, HEIGHT, draw_func=make_overlay(prev))
        (dump / f"{int(time.time() * 1000)}.png").write_bytes(img)

        user_text = USER_TEMPLATE.format(story=story)

        print(f"\n{'=' * 60}")
        print(f"  turn {turn}")
        print(f"{'=' * 60}")
        print("  sending:")
        for line in user_text.splitlines():
            print(f"    | {line}")
        print("  thinking...")

        raw = infer(img, story)

        print(f"  vlm ({len(raw)} chars):")
        print(f"{'-' * 40}")
        print(raw)
        print(f"{'-' * 40}")

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
        (dump / "state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (dump / "story.txt").write_text(story, encoding="utf-8")
        Path("story.txt").write_text(story, encoding="utf-8")

        print(f"  rest {LOOP_DELAY}s...")
        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    main()
