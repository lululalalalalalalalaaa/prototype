"""빌드 인덱서 검증 — 아티팩트 생성 + 증분(재빌드 시 재임베딩 0). mock client."""
import importlib.util
from pathlib import Path

from rag.store import load_index

_SPEC = importlib.util.spec_from_file_location(
    "build_index", Path(__file__).resolve().parent.parent / "scripts" / "build_index.py")
build_index = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_index)


def _upload(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    (up / "철강 데이터.txt").write_text("조강 생산 배출 계수 인벤토리", encoding="utf-8")
    (up / "시멘트 클링커.txt").write_text("클링커 소성 공정 CO2 배출", encoding="utf-8")
    return up


def test_build_creates_loadable_index(tmp_path, fake_client):
    up, idx = _upload(tmp_path), tmp_path / "index"
    build_index.build(upload_dir=up, index_dir=idx, client=fake_client())

    reports = load_index(idx)
    assert {r["db_name"] for r in reports} == {"철강 데이터", "시멘트 클링커"}
    assert all(r["chunks"] and r["chunks"][0]["embedding"] for r in reports)


def test_build_incremental_reuses_embeddings(tmp_path, fake_client):
    up, idx = _upload(tmp_path), tmp_path / "index"

    c1 = fake_client()
    build_index.build(upload_dir=up, index_dir=idx, client=c1)
    assert len(c1.embed_calls) >= 1          # 첫 빌드: 임베딩 호출

    c2 = fake_client()
    build_index.build(upload_dir=up, index_dir=idx, client=c2)
    assert c2.embed_calls == []              # 변경 없음 → 전부 재사용, 재임베딩 0


def test_build_precomputes_and_reuses_metadata(tmp_path, fake_client):
    import json
    up, idx = _upload(tmp_path), tmp_path / "index"
    meta = {"overview": "x", "functional_unit": "1kg", "system_boundary": "c2g",
            "process_flow": "p", "climate_change_total": "1.0"}

    c1 = fake_client(chat_content=json.dumps(meta))
    build_index.build(upload_dir=up, index_dir=idx, client=c1)
    assert all(r["metadata"] == meta for r in load_index(idx))   # 빌드시 precompute
    assert len(c1.chat_calls) >= 1                               # summarize 호출됨

    c2 = fake_client(chat_content=json.dumps(meta))
    build_index.build(upload_dir=up, index_dir=idx, client=c2)
    assert c2.chat_calls == []                                   # 재빌드: metadata도 재사용
    assert c2.embed_calls == []


def test_build_incremental_embeds_only_new(tmp_path, fake_client):
    up, idx = _upload(tmp_path), tmp_path / "index"
    build_index.build(upload_dir=up, index_dir=idx, client=fake_client())

    # 새 문서 1개 추가 → 그 문서만 임베딩
    (up / "알루미늄.txt").write_text("알루미늄 제련 전력 소비", encoding="utf-8")
    c = fake_client()
    build_index.build(upload_dir=up, index_dir=idx, client=c)
    assert len(c.embed_calls) == 1           # 신규 1건만
    assert {r["db_name"] for r in load_index(idx)} == {
        "철강 데이터", "시멘트 클링커", "알루미늄"}
