import pytest

from app.core.exceptions import (
    GeoLensError,
    InvalidImageError,
    LowConfidenceError,
    ModelNotReadyError,
)


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (InvalidImageError, 422),
        (LowConfidenceError, 200),
        (ModelNotReadyError, 503),
    ],
)
def test_status_codes(exc, code):
    assert issubclass(exc, GeoLensError)
    assert exc().status_code == code


def test_default_message_from_docstring():
    assert InvalidImageError().message  # non-empty
    assert ModelNotReadyError("custom").message == "custom"
