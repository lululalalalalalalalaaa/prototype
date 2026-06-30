"""청킹 검증 — 문단 기반 토큰 윈도우(크기 한도·경계 보존·오버랩·분할). API 불필요."""
from rag.embed import get_encoder
from rag.ingest.chunk import build_chunk_input, chunk_body

enc = get_encoder("cl100k_base")


def _ntok(text):
    return len(enc.encode(text))


def test_build_chunk_input_prefixes_name():
    assert build_chunk_input("철강", "본문 내용") == "철강\n본문 내용"


def test_empty_body_returns_empty():
    assert chunk_body("   \n  \n\t") == []


def test_single_short_paragraph_one_chunk():
    out = chunk_body("첫 문단입니다", max_tokens=1000, overlap=0)
    assert out == ["첫 문단입니다"]


def test_paragraphs_preserved_in_chunk():
    out = chunk_body("첫 문단입니다\n둘째 문단입니다", max_tokens=1000, overlap=0)
    assert len(out) == 1
    assert "첫 문단입니다" in out[0]
    assert "둘째 문단입니다" in out[0]


def test_each_chunk_within_token_limit():
    body = "\n".join(f"문단{i} " + "데이터 " * 25 for i in range(30))
    out = chunk_body(body, max_tokens=80, overlap=15)
    assert len(out) >= 2
    for c in out:
        assert _ntok(c) <= 80


def test_overlap_shares_paragraphs_between_chunks():
    # 문단이 overlap 토큰보다 작으면, 인접 청크가 끝쪽 문단을 공유(겹침)해야 함.
    body = "\n".join(f"문장{i}" for i in range(40))
    out = chunk_body(body, max_tokens=20, overlap=8)
    assert len(out) >= 2
    shared = any(set(a.split("\n")) & set(b.split("\n"))
                 for a, b in zip(out, out[1:]))
    assert shared


def test_no_overlap_has_no_shared_paragraph():
    body = "\n".join(f"문장{i}" for i in range(40))
    out = chunk_body(body, max_tokens=20, overlap=0)
    assert len(out) >= 2
    shared = any(set(a.split("\n")) & set(b.split("\n"))
                 for a, b in zip(out, out[1:]))
    assert not shared


def test_oversized_paragraph_is_split():
    body = "토큰 " * 300  # 한도를 크게 넘는 단일 문단
    out = chunk_body(body, max_tokens=50, overlap=0)
    assert len(out) >= 2
    for c in out:
        assert _ntok(c) <= 50


def test_no_overlap_progress_terminates():
    # overlap=0이어도 무한루프 없이 모든 내용을 청크로 소진해야 함
    body = "\n".join(f"문단{i} 짧다" for i in range(50))
    out = chunk_body(body, max_tokens=20, overlap=0)
    assert len(out) >= 1
    assert "문단0" in out[0]
    assert "문단49" in out[-1]
