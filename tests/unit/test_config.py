from app.core.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.app_name == "GeoLens"
    assert s.embedding_dim == 8448
    assert s.default_top_k == 5
    assert s.target_city == "amsterdam"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
