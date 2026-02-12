# drawing.py
"""Drawing utilities for annotating screenshots with AI actions.

Provides self-explanatory visual annotations that are immediately readable
by both humans and vision models. Design principles:
- Annotations must NOT look like native UI elements
- Each action type has a distinct, universally recognizable visual metaphor
- Cursor pointer glyph = "click happened here"
- Radiating burst lines = "impact/event at this point"  
- Dashed arrow trail = "cursor moved from here to there"
- I-beam glyph = "typing happened here"
- All marks use a consistent accent color (magenta) to be distinguishable from UI
"""

import math
from typing import Final

# --- Color constants (RGBA) --------------------------------------------------
# Primary action color: magenta/pink — stands out against virtually any UI
# and is rarely used by Windows/apps as a UI accent color
ACTION_PRIMARY: Final = (255, 50, 200, 255)
ACTION_SECONDARY: Final = (255, 180, 240, 255)
ACTION_OUTLINE: Final = (40, 0, 30, 200)

# Legacy color constants (kept for backward compatibility)
RED: Final = (255, 0, 0, 255)
GREEN: Final = (0, 255, 0, 255)
BLUE: Final = (0, 150, 255, 255)
YELLOW: Final = (255, 255, 0, 255)
WHITE: Final = (255, 255, 255, 255)
BLACK: Final = (0, 0, 0, 255)


# --- Pixel helpers -----------------------------------------------------------

def _set_pixel(data: bytearray, width: int, height: int,
               x: int, y: int, color: tuple[int, int, int, int]) -> None:
    """Set a single pixel in an RGBA bytearray (bounds-checked)."""
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 4
        data[idx] = color[0]
        data[idx + 1] = color[1]
        data[idx + 2] = color[2]
        data[idx + 3] = color[3]


def _set_pixel_thick(data: bytearray, width: int, height: int,
                     x: int, y: int, color: tuple[int, int, int, int],
                     thickness: int = 1) -> None:
    """Set a thick pixel (square brush) in an RGBA bytearray."""
    half = thickness // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            _set_pixel(data, width, height, x + dx, y + dy, color)


# --- Core drawing primitives -------------------------------------------------

def draw_line(rgba: bytes, width: int, height: int,
              x1: int, y1: int, x2: int, y2: int,
              color: tuple[int, int, int, int] = BLUE,
              thickness: int = 3) -> bytes:
    """Draw a solid line between two points using Bresenham's algorithm."""
    data = bytearray(rgba)
    _draw_line_mut(data, width, height, x1, y1, x2, y2, color, thickness)
    return bytes(data)


def _draw_line_mut(data: bytearray, width: int, height: int,
                   x1: int, y1: int, x2: int, y2: int,
                   color: tuple[int, int, int, int],
                   thickness: int = 3) -> None:
    """Draw a solid line in-place on a mutable buffer."""
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


def _draw_dashed_line_mut(data: bytearray, width: int, height: int,
                          x1: int, y1: int, x2: int, y2: int,
                          color: tuple[int, int, int, int],
                          thickness: int = 2,
                          dash_len: int = 8, gap_len: int = 5) -> None:
    """Draw a dashed line in-place on a mutable buffer."""
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1, int(math.hypot(dx, dy)))
    cycle = dash_len + gap_len

    for i in range(dist + 1):
        if (i % cycle) < dash_len:
            t = i / dist
            x = int(x1 + dx * t)
            y = int(y1 + dy * t)
            _set_pixel_thick(data, width, height, x, y, color, thickness)


def draw_crosshair(rgba: bytes, width: int, height: int,
                   x: int, y: int, size: int = 20,
                   color: tuple[int, int, int, int] = RED,
                   thickness: int = 2) -> bytes:
    """Draw a crosshair at specified coordinates."""
    data = bytearray(rgba)
    _draw_crosshair_mut(data, width, height, x, y, size, color, thickness)
    return bytes(data)


def _draw_crosshair_mut(data: bytearray, width: int, height: int,
                        x: int, y: int, size: int = 20,
                        color: tuple[int, int, int, int] = RED,
                        thickness: int = 2) -> None:
    """Draw a crosshair in-place."""
    for dx in range(-size, size + 1):
        _set_pixel_thick(data, width, height, x + dx, y, color, thickness)
    for dy in range(-size, size + 1):
        _set_pixel_thick(data, width, height, x, y + dy, color, thickness)
    # Center dot
    for cy in range(-3, 4):
        for cx in range(-3, 4):
            if cx * cx + cy * cy <= 9:
                _set_pixel(data, width, height, x + cx, y + cy, color)


def draw_circle(rgba: bytes, width: int, height: int,
                x: int, y: int, radius: int,
                color: tuple[int, int, int, int] = GREEN,
                filled: bool = False) -> bytes:
    """Draw a circle at specified coordinates."""
    data = bytearray(rgba)
    _draw_circle_mut(data, width, height, x, y, radius, color, filled)
    return bytes(data)


def _draw_circle_mut(data: bytearray, width: int, height: int,
                     x: int, y: int, radius: int,
                     color: tuple[int, int, int, int],
                     filled: bool = False) -> None:
    """Draw a circle in-place."""
    for cy in range(-radius, radius + 1):
        for cx in range(-radius, radius + 1):
            dist_sq = cx * cx + cy * cy
            if filled:
                if dist_sq <= radius * radius:
                    _set_pixel(data, width, height, x + cx, y + cy, color)
            else:
                if (radius - 2) * (radius - 2) <= dist_sq <= radius * radius:
                    _set_pixel(data, width, height, x + cx, y + cy, color)


def draw_arrow(rgba: bytes, width: int, height: int,
               x1: int, y1: int, x2: int, y2: int,
               color: tuple[int, int, int, int] = BLUE,
               thickness: int = 3) -> bytes:
    """Draw a solid arrow from point 1 to point 2."""
    data = bytearray(rgba)
    _draw_arrow_mut(data, width, height, x1, y1, x2, y2, color, thickness)
    return bytes(data)


def _draw_arrow_mut(data: bytearray, width: int, height: int,
                    x1: int, y1: int, x2: int, y2: int,
                    color: tuple[int, int, int, int],
                    thickness: int = 3,
                    head_length: int = 15,
                    head_angle_deg: float = 30.0) -> None:
    """Draw a solid arrow in-place."""
    _draw_line_mut(data, width, height, x1, y1, x2, y2, color, thickness)
    _draw_arrowhead_mut(data, width, height, x1, y1, x2, y2,
                        color, thickness, head_length, head_angle_deg)


def _draw_dashed_arrow_mut(data: bytearray, width: int, height: int,
                           x1: int, y1: int, x2: int, y2: int,
                           color: tuple[int, int, int, int],
                           thickness: int = 2,
                           dash_len: int = 8, gap_len: int = 5,
                           head_length: int = 15,
                           head_angle_deg: float = 30.0) -> None:
    """Draw a dashed arrow in-place (dashed line + solid arrowhead)."""
    _draw_dashed_line_mut(data, width, height, x1, y1, x2, y2,
                          color, thickness, dash_len, gap_len)
    _draw_arrowhead_mut(data, width, height, x1, y1, x2, y2,
                        color, max(thickness, 3), head_length, head_angle_deg)


def _draw_arrowhead_mut(data: bytearray, width: int, height: int,
                        x1: int, y1: int, x2: int, y2: int,
                        color: tuple[int, int, int, int],
                        thickness: int = 3,
                        head_length: int = 15,
                        head_angle_deg: float = 30.0) -> None:
    """Draw just the arrowhead (the 'spear tip') at the end point."""
    angle = math.atan2(y2 - y1, x2 - x1)
    head_angle = math.radians(head_angle_deg)

    lx = int(x2 - head_length * math.cos(angle - head_angle))
    ly = int(y2 - head_length * math.sin(angle - head_angle))
    _draw_line_mut(data, width, height, x2, y2, lx, ly, color, thickness)

    rx = int(x2 - head_length * math.cos(angle + head_angle))
    ry = int(y2 - head_length * math.sin(angle + head_angle))
    _draw_line_mut(data, width, height, x2, y2, rx, ry, color, thickness)

    # Fill the arrowhead triangle for a solid "spear" look
    _fill_triangle_mut(data, width, height, x2, y2, lx, ly, rx, ry, color)


def _fill_triangle_mut(data: bytearray, width: int, height: int,
                       x1: int, y1: int, x2: int, y2: int,
                       x3: int, y3: int,
                       color: tuple[int, int, int, int]) -> None:
    """Fill a triangle defined by three vertices using scanline."""
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


def draw_rectangle(rgba: bytes, width: int, height: int,
                   x1: int, y1: int, x2: int, y2: int,
                   color: tuple[int, int, int, int] = YELLOW,
                   thickness: int = 2) -> bytes:
    """Draw a rectangle between two corners."""
    data = bytearray(rgba)
    _draw_line_mut(data, width, height, x1, y1, x2, y1, color, thickness)
    _draw_line_mut(data, width, height, x2, y1, x2, y2, color, thickness)
    _draw_line_mut(data, width, height, x2, y2, x1, y2, color, thickness)
    _draw_line_mut(data, width, height, x1, y2, x1, y1, color, thickness)
    return bytes(data)


def normalize_coord(coord: int, max_val: int) -> int:
    """Convert normalized coordinate (0-1000) to actual pixel coordinate."""
    return int((coord / 1000.0) * max_val)


# --- Glyph definitions (pixel art for action indicators) --------------------

# Mouse cursor pointer (classic arrow shape) - 12x16 pixels
# Each row is a string; '#' = primary color, '.' = outline, ' ' = transparent
GLYPH_CURSOR: Final = [
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

# Right-click cursor (pointer with small 'R' indicator) - 12x16
GLYPH_CURSOR_RIGHT: Final = [
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

# Double-click indicator: two concentric impact rings (drawn programmatically)

# I-beam text cursor glyph - 7x14 pixels
GLYPH_IBEAM: Final = [
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


def _draw_glyph_mut(data: bytearray, width: int, height: int,
                    x: int, y: int,
                    glyph: list[str],
                    primary: tuple[int, int, int, int],
                    outline: tuple[int, int, int, int],
                    scale: int = 1) -> None:
    """Draw a pixel-art glyph at the given position.
    
    '#' pixels get the primary color.
    '.' pixels get the outline color.
    ' ' pixels are transparent.
    """
    for row_idx, row in enumerate(glyph):
        for col_idx, ch in enumerate(row):
            if ch == ' ':
                continue
            color = primary if ch == '#' else outline
            for sy in range(scale):
                for sx in range(scale):
                    _set_pixel(data, width, height,
                               x + col_idx * scale + sx,
                               y + row_idx * scale + sy,
                               color)


def _draw_burst_mut(data: bytearray, width: int, height: int,
                    x: int, y: int,
                    color: tuple[int, int, int, int],
                    inner_radius: int = 12,
                    outer_radius: int = 22,
                    num_rays: int = 8,
                    thickness: int = 2) -> None:
    """Draw radiating burst lines around a point (impact/event indicator)."""
    for i in range(num_rays):
        angle = (2 * math.pi * i) / num_rays
        ix = int(x + inner_radius * math.cos(angle))
        iy = int(y + inner_radius * math.sin(angle))
        ox = int(x + outer_radius * math.cos(angle))
        oy = int(y + outer_radius * math.sin(angle))
        _draw_line_mut(data, width, height, ix, iy, ox, oy, color, thickness)


def _draw_double_click_rings_mut(data: bytearray, width: int, height: int,
                                  x: int, y: int,
                                  color: tuple[int, int, int, int]) -> None:
    """Draw two concentric rings to indicate double-click."""
    _draw_circle_mut(data, width, height, x, y, 18, color, filled=False)
    _draw_circle_mut(data, width, height, x, y, 28, color, filled=False)


# --- High-level annotation composites ----------------------------------------

def annotate_left_click(data: bytearray, width: int, height: int,
                        x: int, y: int,
                        prev_x: int | None = None,
                        prev_y: int | None = None) -> None:
    """Draw a complete left-click annotation.
    
    Shows: movement trail (if prev position given) + cursor pointer + burst.
    """
    # Movement trail (dashed arrow from previous position)
    if prev_x is not None and prev_y is not None:
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist > 20:  # Only show trail if meaningful movement
            _draw_dashed_arrow_mut(data, width, height,
                                   prev_x, prev_y, x, y,
                                   ACTION_SECONDARY, thickness=2,
                                   dash_len=6, gap_len=4,
                                   head_length=12)

    # Burst lines (impact indicator)
    _draw_burst_mut(data, width, height, x, y,
                    ACTION_PRIMARY, inner_radius=14, outer_radius=24,
                    num_rays=8, thickness=2)

    # Cursor pointer glyph (offset so the "tip" is at x,y)
    _draw_glyph_mut(data, width, height, x, y,
                    GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE, scale=1)


def annotate_right_click(data: bytearray, width: int, height: int,
                         x: int, y: int,
                         prev_x: int | None = None,
                         prev_y: int | None = None) -> None:
    """Draw a complete right-click annotation.
    
    Shows: movement trail + cursor pointer (right variant) + square burst.
    """
    if prev_x is not None and prev_y is not None:
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist > 20:
            _draw_dashed_arrow_mut(data, width, height,
                                   prev_x, prev_y, x, y,
                                   ACTION_SECONDARY, thickness=2,
                                   dash_len=6, gap_len=4,
                                   head_length=12)

    # Square burst pattern for right-click (distinguishes from left-click's radial burst)
    pad = 20
    _draw_line_mut(data, width, height,
                   x - pad, y - pad, x + pad, y - pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height,
                   x + pad, y - pad, x + pad, y + pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height,
                   x + pad, y + pad, x - pad, y + pad, ACTION_PRIMARY, 2)
    _draw_line_mut(data, width, height,
                   x - pad, y + pad, x - pad, y - pad, ACTION_PRIMARY, 2)

    _draw_glyph_mut(data, width, height, x, y,
                    GLYPH_CURSOR_RIGHT, ACTION_PRIMARY, ACTION_OUTLINE, scale=1)


def annotate_double_click(data: bytearray, width: int, height: int,
                          x: int, y: int,
                          prev_x: int | None = None,
                          prev_y: int | None = None) -> None:
    """Draw a complete double-click annotation.
    
    Shows: movement trail + cursor pointer + double concentric rings.
    """
    if prev_x is not None and prev_y is not None:
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist > 20:
            _draw_dashed_arrow_mut(data, width, height,
                                   prev_x, prev_y, x, y,
                                   ACTION_SECONDARY, thickness=2,
                                   dash_len=6, gap_len=4,
                                   head_length=12)

    # Double rings (the universal "double" indicator)
    _draw_double_click_rings_mut(data, width, height, x, y, ACTION_PRIMARY)

    # Burst
    _draw_burst_mut(data, width, height, x, y,
                    ACTION_PRIMARY, inner_radius=30, outer_radius=38,
                    num_rays=8, thickness=2)

    _draw_glyph_mut(data, width, height, x, y,
                    GLYPH_CURSOR, ACTION_PRIMARY, ACTION_OUTLINE, scale=1)


def annotate_drag(data: bytearray, width: int, height: int,
                  x1: int, y1: int, x2: int, y2: int,
                  prev_x: int | None = None,
                  prev_y: int | None = None) -> None:
    """Draw a complete drag annotation.
    
    Shows: optional approach trail + grip dot at start + dashed drag arrow + 
    release dot at end.
    """
    # Approach trail to drag start
    if prev_x is not None and prev_y is not None:
        dist = math.hypot(x1 - prev_x, y1 - prev_y)
        if dist > 20:
            _draw_dashed_arrow_mut(data, width, height,
                                   prev_x, prev_y, x1, y1,
                                   ACTION_SECONDARY, thickness=1,
                                   dash_len=4, gap_len=4,
                                   head_length=8)

    # Grip indicator at start: filled circle
    _draw_circle_mut(data, width, height, x1, y1, 8,
                     ACTION_PRIMARY, filled=True)

    # Drag path: thick dashed arrow (the main visual)
    _draw_dashed_arrow_mut(data, width, height, x1, y1, x2, y2,
                           ACTION_PRIMARY, thickness=3,
                           dash_len=10, gap_len=6,
                           head_length=18, head_angle_deg=25.0)

    # Release indicator at end: hollow circle
    _draw_circle_mut(data, width, height, x2, y2, 10,
                     ACTION_PRIMARY, filled=False)


def annotate_type(data: bytearray, width: int, height: int,
                  x: int, y: int) -> None:
    """Draw a typing annotation at the estimated focus position.
    
    Shows: I-beam cursor glyph + subtle underline.
    """
    # I-beam glyph (centered on x, y)
    glyph_w = len(GLYPH_IBEAM[0]) if GLYPH_IBEAM else 0
    glyph_h = len(GLYPH_IBEAM)
    _draw_glyph_mut(data, width, height,
                    x - (glyph_w * 2) // 2, y - (glyph_h * 2) // 2,
                    GLYPH_IBEAM, ACTION_PRIMARY, ACTION_OUTLINE, scale=2)

    # Underline bar below the I-beam
    bar_y = y + glyph_h + 4
    _draw_line_mut(data, width, height,
                   x - 15, bar_y, x + 15, bar_y,
                   ACTION_PRIMARY, thickness=2)