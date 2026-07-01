from app.ml.model_registry import ModelManifest, default_manifest


def test_defaults():
    m = default_manifest()
    assert m.model_name == "dinov2_salad"
    assert m.embedding_dim == 8448
    assert m.image_size == 224


def test_json_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    original = default_manifest(index_size=123, weights_sha256="abc")
    original.to_json(path)
    loaded = ModelManifest.from_json(path)
    assert loaded == original
    assert loaded.mean == (0.485, 0.456, 0.406)


def test_is_compatible_ignores_index_size():
    a = default_manifest(index_size=10)
    b = default_manifest(index_size=999)
    assert a.is_compatible(b)


def test_is_incompatible_on_dim_change():
    a = default_manifest()
    b = ModelManifest(embedding_dim=512)
    assert not a.is_compatible(b)
