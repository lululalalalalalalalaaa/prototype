"""설정 로더 검증 — rules.yaml 값이 그대로 노출되는지(리팩터 시 값 보존)."""
from rag.config import get_settings


def test_settings_match_rules_yaml():
    s = get_settings()
    assert s.model == "gpt-5.4-nano"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.top_k == 5
    assert s.similarity_floor == 0.40
    assert s.embed_encoding == "cl100k_base"
    assert s.embed_max_tokens == 8000


def test_settings_cached_singleton():
    assert get_settings() is get_settings()
