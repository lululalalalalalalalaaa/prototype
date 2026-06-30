"""리랭커 검증 — 재정렬·폴백·top_k (mock client, API 키 불필요)."""
import json

from rag.rerank import rerank


def _cands(*names):
    # [(dense_score, report)] — 점수는 내림차순으로 부여
    return [(1.0 - i * 0.1, {"db_name": n, "body": f"{n} 본문"})
            for i, n in enumerate(names)]


def test_rerank_reorders_by_ranking(fake_client):
    client = fake_client(chat_content=json.dumps({"ranking": [2, 0, 1]}))
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=3)
    assert [r["db_name"] for _, r in out] == ["C", "A", "B"]


def test_rerank_top_k_truncates(fake_client):
    client = fake_client(chat_content=json.dumps({"ranking": [2, 0, 1]}))
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=2)
    assert [r["db_name"] for _, r in out] == ["C", "A"]


def test_rerank_partial_ranking_keeps_missing(fake_client):
    # 1번만 지정 → B 먼저, 누락된 A·C는 원래 순서로 뒤에 보존
    client = fake_client(chat_content=json.dumps({"ranking": [1]}))
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=3)
    assert [r["db_name"] for _, r in out] == ["B", "A", "C"]


def test_rerank_string_indices_coerced(fake_client):
    client = fake_client(chat_content=json.dumps({"ranking": ["2", "0", "1"]}))
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=3)
    assert [r["db_name"] for _, r in out] == ["C", "A", "B"]


def test_rerank_invalid_json_falls_back(fake_client):
    client = fake_client(chat_content="not json")
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=3)
    assert [r["db_name"] for _, r in out] == ["A", "B", "C"]   # 원래 순서


def test_rerank_out_of_range_ignored(fake_client):
    client = fake_client(chat_content=json.dumps({"ranking": [9, 1, 99]}))
    out = rerank(client, "q", _cands("A", "B", "C"), top_k=3)
    assert [r["db_name"] for _, r in out] == ["B", "A", "C"]   # 1만 유효 → B, 나머지 보존


def test_rerank_single_candidate_no_call(fake_client):
    client = fake_client(chat_content="{}")
    out = rerank(client, "q", _cands("A"), top_k=5)
    assert [r["db_name"] for _, r in out] == ["A"]
    assert client.chat_calls == []   # 후보 1개면 LLM 호출 없음
