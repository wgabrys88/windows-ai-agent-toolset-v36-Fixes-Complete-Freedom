# screenshot.py
"""Windows screenshot capture utility using GDI with DPI awareness.

Can be run as a standalone script or imported as a module.

Usage:
    python screenshot.py [output.png] [width] [height]
    
Examples:
    python screenshot.py                    # Capture full screen to screenshot.png
    python screenshot.py capture.png        # Capture full screen to capture.png
    python screenshot.py out.png 1920 1080  # Capture and resize to 1920x1080
"""

import ctypes
import ctypes.wintypes as w
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

# Win32 constants
SM_CXSCREEN: Final = 0
SM_CYSCREEN: Final = 1
BI_RGB: Final = 0
DIB_RGB_COLORS: Final = 0
SRCCOPY: Final = 0x00CC0020
CAPTUREBLT: Final = 0x40000000
HALFTONE: Final = 4

# DPI Awareness constants
PROCESS_PER_MONITOR_DPI_AWARE: Final = 2


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", w.DWORD * 3),
    ]


@dataclass(frozen=True, slots=True)
class Win32Context:
    """Windows GDI context wrapper."""
    u32: ctypes.WinDLL
    g32: ctypes.WinDLL
    shcore: ctypes.WinDLL

    @classmethod
    def create(cls) -> "Win32Context":
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        
        return cls(
            u32=ctypes.WinDLL("user32", use_last_error=True),
            g32=ctypes.WinDLL("gdi32", use_last_error=True),
            shcore=shcore,
        )


_ctx = Win32Context.create()


def get_screen_size() -> tuple[int, int]:
    """Get screen dimensions."""
    return (
        _ctx.u32.GetSystemMetrics(SM_CXSCREEN),
        _ctx.u32.GetSystemMetrics(SM_CYSCREEN),
    )


def capture(sw: int, sh: int) -> bytes:
    """Capture screen bitmap in BGRA format."""
    sdc = _ctx.u32.GetDC(0)
    memdc = _ctx.g32.CreateCompatibleDC(sdc)
    
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB
    
    bits = ctypes.c_void_p()
    hbmp = _ctx.g32.CreateDIBSection(sdc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
    old = _ctx.g32.SelectObject(memdc, hbmp)
    
    _ctx.g32.BitBlt(memdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)
    raw = bytes((ctypes.c_ubyte * (sw * sh * 4)).from_address(bits.value))
    
    _ctx.g32.SelectObject(memdc, old)
    _ctx.g32.DeleteObject(hbmp)
    _ctx.g32.DeleteDC(memdc)
    _ctx.u32.ReleaseDC(0, sdc)
    
    return raw


def downsample(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Downsample bitmap using StretchBlt."""
    sdc = _ctx.u32.GetDC(0)
    src_dc = _ctx.g32.CreateCompatibleDC(sdc)
    dst_dc = _ctx.g32.CreateCompatibleDC(sdc)
    
    # Create source bitmap and copy data
    src_bmp = _ctx.g32.CreateCompatibleBitmap(sdc, sw, sh)
    old_src = _ctx.g32.SelectObject(src_dc, src_bmp)
    
    # Create a DIB to write our source data into
    bmi_temp = BITMAPINFO()
    bmi_temp.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_temp.bmiHeader.biWidth = sw
    bmi_temp.bmiHeader.biHeight = -sh
    bmi_temp.bmiHeader.biPlanes = 1
    bmi_temp.bmiHeader.biBitCount = 32
    bmi_temp.bmiHeader.biCompression = BI_RGB
    
    _ctx.g32.SetDIBits(sdc, src_bmp, 0, sh, src, ctypes.byref(bmi_temp), DIB_RGB_COLORS)
    
    # Create destination bitmap
    bmi_dst = BITMAPINFO()
    bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_dst.bmiHeader.biWidth = dw
    bmi_dst.bmiHeader.biHeight = -dh
    bmi_dst.bmiHeader.biPlanes = 1
    bmi_dst.bmiHeader.biBitCount = 32
    bmi_dst.bmiHeader.biCompression = BI_RGB
    
    dst_bits = ctypes.c_void_p()
    dst_bmp = _ctx.g32.CreateDIBSection(sdc, ctypes.byref(bmi_dst), DIB_RGB_COLORS, ctypes.byref(dst_bits), None, 0)
    old_dst = _ctx.g32.SelectObject(dst_dc, dst_bmp)
    
    # Perform the stretch with HALFTONE for better quality
    _ctx.g32.SetStretchBltMode(dst_dc, HALFTONE)
    _ctx.g32.SetBrushOrgEx(dst_dc, 0, 0, None)
    _ctx.g32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY)
    
    # Read the result and fix alpha channel
    raw = bytearray((ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value))
    
    # Set all alpha values to 255 (fully opaque)
    raw[3::4] = bytes([255] * (dw * dh))
    
    out = bytes(raw)
    
    # Cleanup
    _ctx.g32.SelectObject(dst_dc, old_dst)
    _ctx.g32.SelectObject(src_dc, old_src)
    _ctx.g32.DeleteObject(dst_bmp)
    _ctx.g32.DeleteObject(src_bmp)
    _ctx.g32.DeleteDC(dst_dc)
    _ctx.g32.DeleteDC(src_dc)
    _ctx.u32.ReleaseDC(0, sdc)
    
    return out


def bgra_to_rgba(bgra: bytes) -> bytes:
    """Convert BGRA to RGBA and ensure alpha is 255."""
    out = bytearray(len(bgra))
    out[0::4] = bgra[2::4]  # R
    out[1::4] = bgra[1::4]  # G
    out[2::4] = bgra[0::4]  # B
    out[3::4] = bytes([255] * (len(bgra) // 4))  # A = 255 (fully opaque)
    return bytes(out)


def encode_png(rgba: bytes, sw: int, sh: int) -> bytes:
    """Encode RGBA to PNG."""
    raw = bytearray()
    for y in range(sh):
        raw.append(0)
        raw.extend(rgba[y * sw * 4:(y + 1) * sw * 4])
    
    ihdr = struct.pack(">IIBBBBB", sw, sh, 8, 6, 0, 0, 0)
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
    draw_func: Callable[[bytes, int, int], bytes] | None = None
) -> bytes:
    """Capture screen and return PNG image.
    
    Args:
        target_width: Target width for downsampling (None = no downsampling)
        target_height: Target height for downsampling (None = no downsampling)
        draw_func: Optional function to draw annotations on RGBA data
                   Signature: func(rgba: bytes, width: int, height: int) -> bytes
    
    Returns:
        PNG image bytes
    """
    sw, sh = get_screen_size()
    bgra = capture(sw, sh)
    
    if target_width is not None and target_height is not None and (sw, sh) != (target_width, target_height):
        bgra = downsample(bgra, sw, sh, target_width, target_height)
        sw, sh = target_width, target_height
    
    rgba = bgra_to_rgba(bgra)
    
    # Apply drawing function if provided
    if draw_func is not None:
        rgba = draw_func(rgba, sw, sh)
    
    return encode_png(rgba, sw, sh)


def main() -> None:
    """CLI entry point for standalone usage."""
    output_path = Path("screenshot.png")
    target_width = None
    target_height = None
    
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    if len(sys.argv) > 3:
        target_width = int(sys.argv[2])
        target_height = int(sys.argv[3])
    
    sw, sh = get_screen_size()
    print(f"Capturing screen ({sw}x{sh})...", file=sys.stderr)
    
    png_data = capture_screen_png(target_width, target_height)
    
    output_path.write_bytes(png_data)
    
    final_size = f"{target_width}x{target_height}" if target_width and target_height else f"{sw}x{sh}"
    print(f"Screenshot saved to {output_path} ({final_size})", file=sys.stderr)


if __name__ == "__main__":
    main()
