"""토큰/비용 계측 검증 — UsageTracker + embed_text 캡처. API 불필요."""
from types import SimpleNamespace

from rag.usage import UsageTracker


def _resp(prompt_tokens, completion_tokens=0):
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def test_total_tokens():
    u = UsageTracker()
    u.record("임베딩", "text-embedding-3-small", _resp(100))
    u.record("추천", "gpt-5.4-nano", _resp(2000, 50))
    assert u.total_tokens() == 2150


def test_cost_usd_known_price():
    u = UsageTracker()
    u.record("임베딩", "text-embedding-3-small", _resp(1_000_000))  # 1M input
    assert abs(u.cost_usd() - 0.02) < 1e-9   # small input 0.02/1M


def test_cost_usd_input_output_split():
    u = UsageTracker()
    u.record("추천", "gpt-5.4-nano", _resp(1_000_000, 1_000_000))
    # gpt-5.4-nano: input 0.05 + output 0.40 = 0.45
    assert abs(u.cost_usd() - 0.45) < 1e-9


def test_unknown_model_zero_cost():
    u = UsageTracker()
    u.record("x", "unknown-model", _resp(1000, 1000))
    assert u.total_tokens() == 2000
    assert u.cost_usd() == 0.0


def test_record_ignores_missing_usage():
    u = UsageTracker()
    u.record("x", "m", SimpleNamespace())   # usage 속성 없음 → 무시
    assert u.total_tokens() == 0
    assert u.cost_usd() == 0.0


def test_by_stage():
    u = UsageTracker()
    u.record("리랭커", "gpt-5.4-nano", _resp(10, 5))
    assert u.by_stage() == [{"label": "리랭커", "model": "gpt-5.4-nano", "in": 10, "out": 5}]


def test_embed_text_records_usage():
    from rag.embed import embed_text

    class _Client:
        class embeddings:
            @staticmethod
            def create(model, input):
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0))

    u = UsageTracker()
    embed_text(_Client, "짧은 질의", usage=u)
    assert u.total_tokens() == 7
