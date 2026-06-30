"""청킹 검증 — 문단 기반 토큰 윈도우(크기 한도·경계 보존·오버랩·분할). API 불필요."""
from rag.embed import get_encoder
from rag.ingest.chunk import build_chunk_input, chunk_body, structure_chunks

enc = get_encoder("cl100k_base")


def _ntok(text):
    return len(enc.encode(text))


def test_build_chunk_input_prefixes_name_and_section():
    c = {"text": "본문 내용", "section": "영향평가 결과", "kind": "table"}
    assert build_chunk_input("철강", c) == "철강 | 영향평가 결과\n본문 내용"


def test_build_chunk_input_no_section():
    c = {"text": "본문 내용", "section": "", "kind": "body"}
    assert build_chunk_input("철강", c) == "철강\n본문 내용"


# --- structure_chunks: 섹션 경계 분할 + 표 분리 + 섹션/kind 메타 ---
def test_structure_chunks_splits_sections_and_tables():
    body = ("1. 제품 개요\n무궁화호 디젤기차\n"
            "2. 시스템 경계\ngate to gate 수송\n"
            "표. 영향평가 결과(요약)\nClimate change 4.95E-02 kg CO2 eq")
    chunks = structure_chunks(body, max_tokens=1000, overlap=0)
    by_section = {c["section"]: c for c in chunks}
    assert "제품 개요" in by_section
    assert "시스템 경계" in by_section
    # 표는 캡션을 섹션명으로, kind='table'
    tbl = next(c for c in chunks if c["kind"] == "table")
    assert "영향평가 결과" in tbl["section"]
    assert "4.95E-02" in tbl["text"]
    assert by_section["제품 개요"]["kind"] == "body"


def test_structure_chunks_no_headers_single_segment():
    chunks = structure_chunks("헤더 없는 일반 본문입니다", max_tokens=1000, overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "" and chunks[0]["kind"] == "body"


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
