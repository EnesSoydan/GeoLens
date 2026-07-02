from pathlib import Path

from scripts.download_msls import copy_city_subset, find_city_dir, run


def _make_fake_msls(root: Path) -> Path:
    """Build a minimal MSLS-like tree with the two val cities (cph, sf)."""
    cph = root / "train_val" / "cph"
    (cph / "database" / "images").mkdir(parents=True)
    (cph / "query" / "images").mkdir(parents=True)
    (cph / "database" / "images" / "a.jpg").write_bytes(b"a")
    (cph / "query" / "images" / "b.jpg").write_bytes(b"b")
    sf = root / "train_val" / "sf" / "database" / "images"
    sf.mkdir(parents=True)
    (sf / "c.jpg").write_bytes(b"c")
    return cph


def test_find_city_dir(tmp_path):
    cph = _make_fake_msls(tmp_path)
    found = find_city_dir(tmp_path, "cph")
    assert found == cph


def test_find_city_dir_missing(tmp_path):
    _make_fake_msls(tmp_path)
    assert find_city_dir(tmp_path, "paris") is None


def test_copy_city_subset(tmp_path):
    cph = _make_fake_msls(tmp_path)
    dest = tmp_path / "out"
    stats = copy_city_subset(cph, dest, overwrite=False, dry_run=False)
    assert stats.files == 2
    assert stats.roles == {"database": 1, "query": 1}
    assert (dest / "database" / "images" / "a.jpg").exists()


def test_run_copies_all_cities(tmp_path):
    _make_fake_msls(tmp_path)
    dest = tmp_path / "out"
    code = run(tmp_path, ["cph", "sf"], dest, overwrite=False, dry_run=False)
    assert code == 0
    assert (dest / "cph" / "database" / "images" / "a.jpg").exists()
    assert (dest / "sf" / "database" / "images" / "c.jpg").exists()


def test_run_dry_run_copies_nothing(tmp_path):
    _make_fake_msls(tmp_path)
    dest = tmp_path / "out"
    code = run(tmp_path, ["cph"], dest, overwrite=False, dry_run=True)
    assert code == 0
    assert not (dest / "cph" / "database").exists()


def test_run_without_source_returns_2(tmp_path):
    assert run(None, ["cph"], tmp_path / "out", False, False) == 2
