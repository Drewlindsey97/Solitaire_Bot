import struct
from unittest.mock import patch
import pytest
from PIL import Image
import bridge


@pytest.mark.parametrize("header_size", [12, 16])
@pytest.mark.parametrize("pixel_format,channels", [(1, 4), (2, 4), (3, 3)])
def test_raw_screenshot_preserves_rgb_pixels(header_size, pixel_format, channels):
    rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (12, 34, 56)]
    pixels = bytes(v for color in rgb for v in (color + ((255,) if channels == 4 else ())))
    header = struct.pack("<3I", 2, 2, pixel_format)
    if header_size == 16:
        header += struct.pack("<I", 1)
    image = bridge.decode_raw_screenshot(header + pixels)
    assert image.size == (2, 2)
    assert image.tobytes() == bytes(v for color in rgb for v in color)


@pytest.mark.parametrize("data", [b"", struct.pack("<3I", 720, 1600, 1),
    struct.pack("<3I", 2, 2, 99) + bytes(16),
    struct.pack("<3I", 0, 2, 1) + bytes(16)])
def test_malformed_raw_frame_is_rejected(data):
    with pytest.raises(ValueError):
        bridge.decode_raw_screenshot(data)


def test_capture_falls_back_to_png():
    import io
    expected = Image.new("RGB", (2, 2), (12, 34, 56))
    png = io.BytesIO()
    expected.save(png, format="PNG")
    with patch.object(bridge, "RUN_MODE", "PC_ADB"), \
         patch.object(bridge, "adb_capture_raw_image", return_value=None), \
         patch.object(bridge, "adb_capture_png_bytes", return_value=png.getvalue()):
        assert bridge.screenshot().tobytes() == expected.tobytes()
