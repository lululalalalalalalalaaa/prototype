"""인덱스 아티팩트 검증 — save_index → load_index round-trip(reports 구조 복원)."""
import math

from rag.store import body_hash, load_index, save_index


def _reports():
    return [
        {"db_name": "철강", "body": "조강 배출 계수",
         "metadata": {"overview": "철강 LCI", "functional_unit": "1 톤"},
         "chunks": [{"text": "철강\n조강", "embedding": [0.5, 0.25]},
                    {"text": "철강\n계수", "embedding": [0.0, 1.0]}]},
        {"db_name": "생활용수(제주권)", "body": "제주 생활용수",
         "metadata": None,
         "chunks": [{"text": "생활용수\n제주", "embedding": [1.0, 0.0]}]},
    ]


def _close(a, b):
    return all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)
               for x, y in zip(a, b))


def test_round_trip_reconstructs_reports(tmp_path):
    reports = _reports()
    save_index(reports, index_dir=tmp_path)
    loaded = load_index(index_dir=tmp_path)

    assert [r["db_name"] for r in loaded] == [r["db_name"] for r in reports]
    assert [r["body"] for r in loaded] == [r["body"] for r in reports]
    for orig, got in zip(reports, loaded):
        assert [c["text"] for c in got["chunks"]] == [c["text"] for c in orig["chunks"]]
        for oc, gc in zip(orig["chunks"], got["chunks"]):
            assert _close(gc["embedding"], oc["embedding"])  # float32 허용오차


def test_round_trip_preserves_korean(tmp_path):
    save_index(_reports(), index_dir=tmp_path)
    raw = (tmp_path / "docs.jsonl").read_text(encoding="utf-8")
    assert "생활용수" in raw  # ensure_ascii=False 유지


def test_chunk_embedding_alignment(tmp_path):
    # 청크가 여러 문서·여러 개여도 임베딩 행 정렬이 정확한지
    reports = _reports()
    save_index(reports, index_dir=tmp_path)
    loaded = load_index(index_dir=tmp_path)
    assert _close(loaded[0]["chunks"][1]["embedding"], [0.0, 1.0])
    assert _close(loaded[1]["chunks"][0]["embedding"], [1.0, 0.0])


def test_metadata_round_trip(tmp_path):
    save_index(_reports(), index_dir=tmp_path)
    loaded = load_index(index_dir=tmp_path)
    assert loaded[0]["metadata"] == {"overview": "철강 LCI", "functional_unit": "1 톤"}
    assert loaded[1]["metadata"] is None   # metadata 없는 문서도 graceful


def test_missing_index_returns_empty(tmp_path):
    assert load_index(index_dir=tmp_path / "none") == []


def test_body_hash_stable_and_distinct():
    assert body_hash("a", "b") == body_hash("a", "b")
    assert body_hash("a", "b") != body_hash("a", "c")
