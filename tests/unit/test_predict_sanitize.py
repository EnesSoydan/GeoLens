"""Input-sanitization tests for the /predict boundary (EXIF stripping)."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

from app.api.routes_predict import _strip_exif


def _jpeg_with_orientation(tag: int) -> Image.Image:
    """A landscape JPEG whose EXIF Orientation demands a display transform.

    Asymmetric corner colours make any wrong rotation detectable pixel-wise.
    """
    img = Image.new("RGB", (16, 10), color=(200, 200, 200))
    img.paste((220, 30, 30), (0, 0, 6, 4))  # red top-left marker
    exif = Image.Exif()
    exif[0x0112] = tag  # Orientation.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    loaded = Image.open(buf)
    loaded.load()
    return loaded


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
    src = _jpeg_with_exif()  # Orientation defaults to 1 (upright) -> no transform.
    stripped = _strip_exif(src)
    assert stripped.size == src.size
    assert stripped.mode == src.mode
    assert list(stripped.getdata()) == list(src.getdata())


def test_strip_exif_bakes_orientation_into_pixels():
    # Orientation=6 (rotate 90 deg CW on display): the pipeline must feed the
    # upright pixels, not the raw sideways sensor frame, or the query embedding
    # ends up rotated relative to the upright MSLS reference index.
    src = _jpeg_with_orientation(6)
    expected = ImageOps.exif_transpose(src)  # ground-truth display orientation
    stripped = _strip_exif(src)
    # The strip output is rotated to portrait and matches the viewer orientation.
    assert stripped.size == expected.size != src.size
    assert list(stripped.getdata()) == list(expected.getdata())
    # And the orientation tag itself is gone (rebuilt from raw pixels).
    assert stripped.getexif().get(0x0112) is None
