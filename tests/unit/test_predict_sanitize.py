"""Input-sanitization tests for the /predict boundary (EXIF stripping)."""

from __future__ import annotations

import io

from PIL import Image

from app.api.routes_predict import _strip_exif


def _jpeg_with_exif() -> Image.Image:
    """Build a JPEG-decoded image carrying an EXIF block (incl. GPS)."""
    exif = Image.Exif()
    exif[0x0110] = "TestCamera"  # Model tag.
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[2] = (52.0, 22.0, 0.0)  # GPSLatitude (deg, min, sec).
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    loaded = Image.open(buf)
    loaded.load()
    return loaded


def test_source_image_has_exif():
    # Guard: the fixture actually carries EXIF, so the strip assertion is meaningful.
    img = _jpeg_with_exif()
    assert img.info.get("exif") is not None
    assert len(img.getexif()) > 0


def test_strip_exif_removes_all_metadata():
    stripped = _strip_exif(_jpeg_with_exif())
    assert stripped.info.get("exif") is None
    assert len(stripped.getexif()) == 0
    # GPS IFD is gone too.
    assert stripped.getexif().get_ifd(0x8825) == {}


def test_strip_exif_preserves_pixels():
    src = _jpeg_with_exif()
    stripped = _strip_exif(src)
    assert stripped.size == src.size
    assert stripped.mode == src.mode
    assert list(stripped.getdata()) == list(src.getdata())
