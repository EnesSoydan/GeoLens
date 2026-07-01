import pytest

from app.ml.aggregator import DESCRIPTOR_DIM, check_descriptor_dim


def test_descriptor_dim_is_8448():
    assert DESCRIPTOR_DIM == 8448


def test_check_descriptor_dim_ok():
    check_descriptor_dim(DESCRIPTOR_DIM)  # no raise


def test_check_descriptor_dim_wrong():
    with pytest.raises(ValueError):
        check_descriptor_dim(512)
