"""OpenAI 호출의 토큰·비용 집계.

검색 1회 동안 각 단계(임베딩·리랭커·추천)의 OpenAI usage(토큰)를 모아 총 토큰·USD 비용을 낸다.
RAG의 운영 관측성(비용 추적)을 위한 모듈.
"""

# 모델별 단가 (USD per 1M tokens). ⚠️ 실제 청구 단가에 맞춰 조정하세요(추정치 포함).
PRICES = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4": {"input": 1.25, "output": 10.0},   # 이미지 vision용(추정치 — 실제 단가로 조정)
}


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
        total = 0.0
        for _, model, i, o in self.calls:
            p = PRICES.get(model, {"input": 0.0, "output": 0.0})
            total += i / 1e6 * p["input"] + o / 1e6 * p["output"]
        return total

    def by_stage(self):
        """[{label, model, in, out}] — UI/trace 표시용."""
        return [{"label": label, "model": model, "in": i, "out": o}
                for label, model, i, o in self.calls]
