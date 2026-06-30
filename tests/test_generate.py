"""생성 레이어 검증 — 컨텍스트 포맷, 프롬프트 불변식(스냅샷), JSON 파싱."""
import json

from rag.generate import build_candidates_context, recommend, summarize_detail


def test_build_candidates_context_format():
    ranked = [(0.5, {"db_name": "철강", "body": "조강 배출 계수"})]
    ctx = build_candidates_context(ranked)
    assert "[후보 1] LCI DB 이름: 철강" in ctx
    assert "유사도: 0.500" in ctx
    assert "관련 정보: 조강 배출 계수" in ctx


def test_recommend_parses_json(fake_client):
    content = json.dumps({
        "no_match": False,
        "recommended": {"db_name": "철강", "reasons": ["일치"]},
        "others": [],
    })
    client = fake_client(chat_content=content)
    out = recommend(client, "철강 배출", [(0.9, {"db_name": "철강", "body": "조강"})])
    assert out["data"]["recommended"]["db_name"] == "철강"
    assert out["raw"] == content


def test_recommend_invalid_json_keeps_raw(fake_client):
    client = fake_client(chat_content="not json at all")
    out = recommend(client, "q", [(0.9, {"db_name": "철강", "body": "조강"})])
    assert out["data"] is None
    assert out["raw"] == "not json at all"


def test_recommend_prompt_invariants(fake_client):
    """추천 프롬프트의 핵심 규칙이 우발적으로 사라지지 않았는지 보호(스냅샷)."""
    client = fake_client(chat_content="{}")
    recommend(client, "검색어", [(0.9, {"db_name": "철강", "body": "조강"})])
    call = client.chat_calls[0]
    system = call["messages"][0]["content"]
    user = call["messages"][1]["content"]
    assert call["response_format"] == {"type": "json_object"}
    # 후보 외 환각 방지 / 원문 인용 금지 / no_match 규칙이 유지되어야 함
    assert "주어진 후보 중에서만 고르고" in system
    assert "원문을 그대로 인용/복사하지 마세요" in system
    assert '"no_match"' in system
    assert "대체하지 말고 no_match" in system   # near-domain 거부(그라운딩 강화)
    assert "직접 해당하는" in system            # 일반 질의도 후보 있으면 추천(과잉기권 방지)
    assert "검색어" in user


def test_summarize_detail_none_client_returns_none():
    assert summarize_detail(None, "철강", "본문") is None


def test_summarize_detail_parses_json(fake_client):
    content = json.dumps({
        "overview": "철강 LCI", "functional_unit": "1 톤",
        "system_boundary": "cradle-to-gate", "process_flow": "전기로",
        "climate_change_total": "1.8 tCO2eq",
    })
    client = fake_client(chat_content=content)
    out = summarize_detail(client, "철강", "본문")
    assert out["functional_unit"] == "1 톤"


def test_summarize_detail_invalid_json_returns_none(fake_client):
    client = fake_client(chat_content="oops")
    assert summarize_detail(client, "철강", "본문") is None
