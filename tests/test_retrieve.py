"""검색 레이어 검증 — 코사인 유사도 값과 랭킹 정렬/필터/top_k."""
import math

from rag.retrieve import (best_chunk, bm25_scores, cosine_similarity,
                          hybrid_rank, rank, tokenize)


def test_best_chunk_returns_position():
    report = {"chunks": [
        {"text": "c0", "embedding": [1.0, 0.0]},
        {"text": "c1", "embedding": [0.0, 1.0]},
        {"text": "c2", "embedding": [0.5, 0.5]},
    ]}
    out = best_chunk([0.0, 1.0], report)   # 질의 → c1(인덱스 1)이 최고
    assert out["text"] == "c1"
    assert out["chunk_index"] == 2          # 1-based
    assert out["n_chunks"] == 3
    assert out["position_pct"] == 33        # round(1/3*100)


def test_best_chunk_no_chunks():
    out = best_chunk([1.0, 0.0], {"chunks": []})
    assert out["text"] is None
    assert out["n_chunks"] == 0
    assert out["chunk_index"] == 0


def test_cosine_identical_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_guard():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_known_value():
    # [1,1] vs [1,0] → 1/sqrt(2)
    assert math.isclose(cosine_similarity([1.0, 1.0], [1.0, 0.0]), 1 / math.sqrt(2))


def _report(name, *chunk_embs):
    """청크 임베딩들을 가진 보고서(데이터 모델: {db_name, body, chunks:[{embedding}]})."""
    return {"db_name": name, "body": name,
            "chunks": [{"text": name, "embedding": e} for e in chunk_embs]}


def test_rank_orders_by_similarity_desc():
    reports = [
        _report("A", [1.0, 0.0]),
        _report("B", [0.0, 1.0]),
        _report("C", [1.0, 1.0]),
    ]
    ranked = rank([1.0, 0.0], reports, top_k=3)
    names = [r["db_name"] for _, r in ranked]
    assert names[0] == "A"            # 완전 일치가 1위
    assert names[-1] == "B"           # 직교가 꼴찌


def test_rank_uses_max_chunk_score():
    # B는 무관 청크 + 완전 일치 청크를 가짐 → max 집계로 무관 단일 청크 A를 이겨야 함.
    a = _report("A", [0.3, 0.3])               # 한 청크, 어중간
    b = _report("B", [0.0, 1.0], [1.0, 0.0])   # 두 청크, 하나가 완전 일치
    ranked = rank([1.0, 0.0], [a, b], top_k=2)
    assert ranked[0][1]["db_name"] == "B"
    assert ranked[0][0] == 1.0                 # 최고 청크 점수 = 1.0


def test_rank_skips_reports_without_chunks():
    reports = [_report("A", [1.0, 0.0]),
               {"db_name": "X", "body": "x"},          # chunks 없음
               {"db_name": "Y", "body": "y", "chunks": []}]  # 빈 청크
    ranked = rank([1.0, 0.0], reports, top_k=5)
    assert [r["db_name"] for _, r in ranked] == ["A"]


def test_rank_respects_top_k():
    reports = [_report(str(i), [float(i), 1.0]) for i in range(10)]
    assert len(rank([1.0, 1.0], reports, top_k=3)) == 3


def test_rank_default_top_k_from_settings():
    reports = [_report(str(i), [float(i), 1.0]) for i in range(10)]
    assert len(rank([1.0, 1.0], reports)) == 5  # rules.yaml top_k


# --- BM25 / 하이브리드 ---
def test_tokenize_hangul_bigram():
    toks = tokenize("경남권")
    assert "경남" in toks and "남권" in toks


def test_tokenize_alnum_lowercase():
    toks = tokenize("MDF 합판")
    assert "mdf" in toks          # 영숫자 토큰 + 소문자화
    assert "합판" in toks          # 한글 bigram


def test_tokenize_variant_overlap():
    # '경남' 질의가 '경남권' 문서와 bigram을 공유(정확형 변형 흡수)
    assert set(tokenize("경남")) & set(tokenize("경남권"))


def test_bm25_prefers_matching_doc():
    docs = [tokenize("경남 산업용수 데이터"), tokenize("전혀 무관한 문서")]
    scores = bm25_scores("경남 산업용수", docs)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0       # 겹치는 토큰 없으면 0


def test_hybrid_rank_boosts_lexical_match():
    # dense로는 A(완전 일치)가 1위지만, B만 정확 토큰('경남')을 가져 RRF가 B를 끌어올림.
    a = {"db_name": "가나다", "body": "전혀 다른 내용",
         "chunks": [{"text": "x", "embedding": [1.0, 0.0]}]}
    b = {"db_name": "경남권", "body": "경남 산업용수 인벤토리",
         "chunks": [{"text": "y", "embedding": [0.6, 0.4]}]}
    out = hybrid_rank("경남 산업용수", [1.0, 0.0], [a, b], top_k=2)
    assert out[0][1]["db_name"] == "경남권"
    # 반환 점수는 dense 코사인(floor 보존)
    assert math.isclose(out[0][0], 0.6 / math.sqrt(0.52), rel_tol=1e-6)


def test_hybrid_rank_skips_no_chunks():
    a = _report("A", [1.0, 0.0])
    no_chunk = {"db_name": "X", "body": "x"}
    out = hybrid_rank("query", [1.0, 0.0], [a, no_chunk], top_k=5)
    assert [r["db_name"] for _, r in out] == ["A"]
