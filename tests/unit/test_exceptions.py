import pytest

from app.core.exceptions import (
    FileTooLargeError,
    GeoLensError,
    InvalidImageError,
    LowConfidenceError,
    ModelNotReadyError,
)


@pytest.mark.parametrize(
    ("exc", "status_code", "code"),
    [
        (InvalidImageError, 400, "INVALID_IMAGE"),
        (FileTooLargeError, 413, "FILE_TOO_LARGE"),
        (LowConfidenceError, 200, "LOW_CONFIDENCE"),
        (ModelNotReadyError, 503, "MODEL_NOT_READY"),
    ],
)
def test_status_and_error_codes(exc, status_code, code):
    assert issubclass(exc, GeoLensError)
    assert exc().status_code == status_code
    assert exc().code == code


def test_default_message_from_docstring():
    assert InvalidImageError().message  # non-empty
    assert ModelNotReadyError("custom").message == "custom"
