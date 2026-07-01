from pathlib import Path

from scripts.download_msls import copy_city_subset, find_city_dir, run


def _make_fake_msls(root: Path) -> Path:
    """Build a minimal MSLS-like tree with two cities."""
    ams = root / "train_val" / "amsterdam"
    (ams / "database" / "images").mkdir(parents=True)
    (ams / "query" / "images").mkdir(parents=True)
    (ams / "database" / "images" / "a.jpg").write_bytes(b"a")
    (ams / "query" / "images" / "b.jpg").write_bytes(b"b")
    berlin = root / "train_val" / "berlin" / "database" / "images"
    berlin.mkdir(parents=True)
    (berlin / "c.jpg").write_bytes(b"c")
    return ams


def test_find_city_dir(tmp_path):
    ams = _make_fake_msls(tmp_path)
    found = find_city_dir(tmp_path, "amsterdam")
    assert found == ams


def test_find_city_dir_missing(tmp_path):
    _make_fake_msls(tmp_path)
    assert find_city_dir(tmp_path, "paris") is None


def test_copy_city_subset(tmp_path):
    ams = _make_fake_msls(tmp_path)
    dest = tmp_path / "out"
    stats = copy_city_subset(ams, dest, overwrite=False, dry_run=False)
    assert stats.files == 2
    assert stats.roles == {"database": 1, "query": 1}
    assert (dest / "database" / "images" / "a.jpg").exists()


def test_run_dry_run_copies_nothing(tmp_path):
    _make_fake_msls(tmp_path)
    dest = tmp_path / "out"
    code = run(tmp_path, "amsterdam", dest, overwrite=False, dry_run=True)
    assert code == 0
    assert not (dest / "database").exists()


def test_run_without_source_returns_2(tmp_path):
    assert run(None, "amsterdam", tmp_path / "out", False, False) == 2
