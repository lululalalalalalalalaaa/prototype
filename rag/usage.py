"""OpenAI 호출의 토큰·비용 집계.

검색 1회 동안 각 단계(임베딩·리랭커·추천)의 OpenAI usage(토큰)를 모아 총 토큰·USD 비용을 낸다.
RAG의 운영 관측성(비용 추적)을 위한 모듈.
"""

from rag.config import get_settings

# 모델별 단가(USD per 1M tokens)는 config/rules.yaml의 `prices`가 단일 소스다.


class UsageTracker:
    """단계별 (label, model, prompt_tokens, completion_tokens)를 누적한다."""

    def __init__(self):
        self.calls = []  # [(label, model, in_tok, out_tok)]

    def record(self, label, model, response):
        """OpenAI 응답의 usage를 기록한다(usage 없으면 무시)."""
        u = getattr(response, "usage", None)
        if u is None:
            return
        self.calls.append((label, model,
                           getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0))

    def total_tokens(self):
        return sum(i + o for _, _, i, o in self.calls)

    def cost_usd(self):
        prices = get_settings().prices
        total = 0.0
        for _, model, i, o in self.calls:
            p = prices.get(model, {"input": 0.0, "output": 0.0})
            total += i / 1e6 * p["input"] + o / 1e6 * p["output"]
        return total

    def by_stage(self):
        """[{label, model, in, out}] — UI/trace 표시용."""
        return [{"label": label, "model": model, "in": i, "out": o}
                for label, model, i, o in self.calls]
