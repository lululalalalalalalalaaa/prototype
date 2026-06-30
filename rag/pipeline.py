"""검색 오케스트레이션 — retrieve + generate 조립.

원본 app.py의 run_search 흐름을 그대로 옮기되, retrieval(retrieve.rank)과
생성(generate.recommend)을 분리해 각 레이어를 단독 검증할 수 있게 했습니다.
반환 dict 형태는 원본과 동일하여 UI 렌더링(render_search_result)을 바꾸지 않습니다.
"""
import logging
import time

from rag.clients import get_client
from rag.config import get_settings
from rag.embed import embed_text
from rag.generate import recommend
from rag.rerank import rerank
from rag.retrieve import best_chunk, hybrid_rank
from rag.usage import UsageTracker

log = logging.getLogger("rag.pipeline")


def _ms(a, b):
    return round((b - a) * 1000)


def visible_others(others, scores, floor, ratio):
    """LLM이 제시한 '다른 유사 후보(others)' 중 화면에 노출할 것만 거릅니다.

    임계 = min(절대 floor, 최고 점수 × ratio). 짧은/제너럴 질의는 코사인이 전반적으로
    낮아 절대 floor에 모두 걸리므로, 추천(최고 점수)에 견줘 비슷한 후보는 노출합니다
    (추천은 floor와 무관하게 항상 표시되므로 일관). 임계는 floor 이하라 기존 동작은 비회귀.
    """
    if not others:
        return []
    top = max(scores.values(), default=0.0)
    threshold = min(floor, top * ratio)
    return [o for o in others
            if scores.get(o.get("db_name", ""), 0.0) >= threshold]


def search(reports, query):
    """검색을 실행하고 렌더링에 필요한 결과 dict를 반환합니다.

    반환:
      {"error": "no_key"|"no_reports"|"no_embeddings"}  또는
      {"data": dict|None, "raw": str, "scores": {db_name: score}, "trace": {...}}
    trace는 각 단계(임베딩·하이브리드·리랭커·추천)의 입출력·타이밍을 담아 관측성을 제공한다.
    """
    client = get_client()
    if client is None:
        return {"error": "no_key"}
    if not reports:
        return {"error": "no_reports"}

    settings = get_settings()
    usage = UsageTracker()

    # 1) 검색어 임베딩
    t0 = time.perf_counter()
    query_embedding = embed_text(client, query, usage=usage)
    t1 = time.perf_counter()
    # 2) Dense+BM25를 RRF로 융합해 넓게(rerank_pool) 후보 추림
    pool = hybrid_rank(query, query_embedding, reports, top_k=settings.rerank_pool)
    t2 = time.perf_counter()
    if not pool:
        return {"error": "no_embeddings"}
    # 3) LLM 리랭커로 관련도 재정렬 → 상위 top_k
    ranked = rerank(client, query, pool, settings.top_k, usage=usage)
    t3 = time.perf_counter()
    # 4) 추천 모델(no_match 포함)
    result = recommend(client, query, ranked, usage=usage)
    t4 = time.perf_counter()

    result["scores"] = {r["db_name"]: s for s, r in ranked}
    # 출처(provenance): 후보별 질의와 가장 잘 맞는 근거 본문 청크
    result["evidence"] = {r["db_name"]: best_chunk(query_embedding, r) for _, r in ranked}

    data = result.get("data") or {}
    rec = ("no_match (적합 DB 없음)" if data.get("no_match")
           else (data.get("recommended") or {}).get("db_name", "(파싱 실패)"))
    result["trace"] = {
        "query": query,
        "stages": [
            {"name": "1. 질의 임베딩", "detail": f"{len(query_embedding)}차원 벡터", "ms": _ms(t0, t1)},
            {"name": "2. 하이브리드 검색 (Dense+BM25 → RRF)",
             "detail": f"후보 {len(pool)}개 추림",
             "top": [(r["db_name"], round(s, 3)) for s, r in pool[:5]], "ms": _ms(t1, t2)},
            {"name": "3. LLM 리랭커",
             "detail": f"top-{len(ranked)} 재정렬",
             "before": [r["db_name"] for _, r in pool[:5]],
             "after": [r["db_name"] for _, r in ranked], "ms": _ms(t2, t3)},
            {"name": "4. LLM 추천", "detail": rec, "ms": _ms(t3, t4)},
        ],
        "total_ms": _ms(t0, t4),
        "tokens": usage.total_tokens(),
        "cost_usd": usage.cost_usd(),
        "usage": usage.by_stage(),
    }
    log.info("[검색] '%s' | 임베딩 %dms · 하이브리드 %dms(후보 %d) · 리랭커 %dms · 추천 %dms · 합 %dms"
             " | 토큰 %d · $%.5f → %s",
             query[:24], _ms(t0, t1), _ms(t1, t2), len(pool), _ms(t2, t3), _ms(t3, t4),
             _ms(t0, t4), usage.total_tokens(), usage.cost_usd(), rec)
    return result
