import pytest

from data import cache


@pytest.fixture
def isolated_cache_root(tmp_path, monkeypatch):
    """Point the cache module at a throwaway directory for this test only."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    return tmp_path
