"""코사인 유사도 기반 검색(retrieval) 레이어.

LLM을 호출하지 않으므로 eval 하니스로 단독 평가(Recall@k/MRR)가 가능합니다.
추후 BM25 하이브리드·리랭커가 이 레이어에 합류합니다.
"""
import math
import re

from rag.config import get_settings

# 토크나이저: 한글 음절 런 → 문자 bigram, 영숫자 런 → 토큰(소문자).
# bigram은 '경남'질의가 '경남권' 문서와 겹치게 해 정확형 변형을 흡수한다.
_HANGUL_RUN = re.compile(r"[가-힣]+")
_ALNUM_RUN = re.compile(r"[a-z0-9]+")


def cosine_similarity(a, b):
    """두 벡터의 코사인 유사도(-1~1)를 계산합니다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def best_chunk(query_embedding, report):
    """문서에서 질의와 가장 잘 맞는 청크를 반환합니다(출처/근거 표시용).

    반환: {"score", "text", "chunk_index"(1-based), "n_chunks", "position_pct"}.
    청크는 문서 순서대로라 chunk_index가 문서 내 위치(앞→뒤)를 나타낸다(HWP는 페이지 평탄화로
    페이지 번호가 없어 청크 위치로 대체). text는 이름 prefix 없는 원문 발췌.
    """
    chunks = report.get("chunks", [])
    n = len(chunks)
    sims = [(cosine_similarity(query_embedding, c["embedding"]), i, c.get("text"))
            for i, c in enumerate(chunks) if c.get("embedding")]
    if not sims:
        return {"score": 0.0, "text": None, "chunk_index": 0, "n_chunks": n, "position_pct": 0}
    score, idx, text = max(sims, key=lambda x: x[0])
    return {"score": round(score, 3), "text": text,
            "chunk_index": idx + 1, "n_chunks": n,
            "position_pct": round(idx / n * 100) if n else 0}


def rank(query_embedding, reports, top_k=None):
    """질의 임베딩과 모든 보고서의 유사도를 구해 상위 top_k를 반환합니다.

    문서 점수는 그 문서 청크들의 코사인 유사도 **최댓값(max)**입니다(가장 잘 맞는 섹션 기준).
    청크가 없는 보고서는 제외합니다. 반환은 [(score, report), ...] (내림차순).
    top_k가 None이면 rules.yaml의 top_k를 사용합니다.
    """
    k = get_settings().top_k if top_k is None else top_k
    scored = []
    for r in reports:
        sims = [cosine_similarity(query_embedding, c["embedding"])
                for c in r.get("chunks", []) if c.get("embedding")]
        if sims:
            scored.append((max(sims), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# BM25 (sparse) + RRF 하이브리드 — 순수 파이썬(외부 의존성 없음)
# ---------------------------------------------------------------------------
def tokenize(text):
    """텍스트를 BM25 토큰 목록으로 변환합니다.

    - 한글 음절 런 → 문자 bigram(길이 1이면 unigram)
    - 영숫자 런 → 토큰 그대로(mdf, ktx, lpg, lci, co2 등)
    모두 소문자화. 순서는 무의미(bag-of-words).
    """
    text = text.lower()
    tokens = list(_ALNUM_RUN.findall(text))
    for run in _HANGUL_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def bm25_scores(query_text, docs_tokens, settings=None):
    """질의에 대한 각 문서의 BM25 점수 리스트(docs_tokens와 같은 순서)를 반환합니다."""
    settings = settings or get_settings()
    q = set(tokenize(query_text))
    n_docs = len(docs_tokens)
    if n_docs == 0 or not q:
        return [0.0] * n_docs

    df = {}
    for toks in docs_tokens:
        for t in set(toks) & q:
            df[t] = df.get(t, 0) + 1
    dls = [len(toks) for toks in docs_tokens]
    avgdl = (sum(dls) / n_docs) or 1
    k1, b = settings.bm25_k1, settings.bm25_b

    scores = []
    for toks, dl in zip(docs_tokens, dls):
        tf = {}
        for t in toks:
            if t in q:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t, f in tf.items():
            n = df.get(t, 0)
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))  # 항상 양수(BM25+)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def hybrid_rank(query_text, query_embedding, reports, top_k=None):
    """Dense(청크-max 코사인) + BM25(sparse)를 RRF로 융합해 상위 top_k를 반환합니다.

    반환 점수는 **dense 코사인**입니다(app의 similarity_floor 필터 의미 보존). 순서만 RRF가 결정.
    """
    settings = get_settings()
    k = settings.top_k if top_k is None else top_k

    # 후보: 임베딩 청크가 있는 문서. dense 점수 = 청크 코사인 max.
    candidates, dense_score = [], {}
    for r in reports:
        sims = [cosine_similarity(query_embedding, c["embedding"])
                for c in r.get("chunks", []) if c.get("embedding")]
        if sims:
            dense_score[len(candidates)] = max(sims)
            candidates.append(r)
    if not candidates:
        return []

    # dense 순위(1-based)
    dense_order = sorted(range(len(candidates)), key=lambda i: dense_score[i], reverse=True)
    dense_rank = {i: pos for pos, i in enumerate(dense_order, start=1)}

    # bm25 순위(점수 0인 문서는 제외 → dense 기여만).
    # 색인은 DB 이름만 사용한다: 변별 신호(경남권·수도권·MDF)는 이름에 있고,
    # 긴 본문을 bigram으로 색인하면 흔한 bigram이 잡음으로 작용해 랭킹을 해친다(실측).
    docs_tokens = [tokenize(r["db_name"]) for r in candidates]
    bm = bm25_scores(query_text, docs_tokens, settings)
    bm_order = [i for i in sorted(range(len(candidates)), key=lambda i: bm[i], reverse=True)
                if bm[i] > 0]
    bm_rank = {i: pos for pos, i in enumerate(bm_order, start=1)}

    # RRF 융합
    rrf_k = settings.rrf_k
    fused = {}
    for i in range(len(candidates)):
        s = 1.0 / (rrf_k + dense_rank[i])
        if i in bm_rank:
            s += 1.0 / (rrf_k + bm_rank[i])
        fused[i] = s

    order = sorted(range(len(candidates)), key=lambda i: fused[i], reverse=True)[:k]
    return [(dense_score[i], candidates[i]) for i in order]
