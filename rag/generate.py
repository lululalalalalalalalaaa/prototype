"""LLM 생성 레이어 — 추천 문구 및 세부정보 요약.

system/user 프롬프트는 원본 app.py에서 한 글자도 바꾸지 않고 옮겼습니다.
Streamlit 캐싱(st.cache_data)은 UI 계층(app.py)으로 옮겨 rag 패키지를 UI에서 분리했습니다.
"""
import json

from rag.config import get_settings

# (요약 키, 화면 라벨) — 사용자가 요청한 5개 항목
DETAIL_FIELDS = [
    ("overview", "제품(모듈) 개요"),
    ("functional_unit", "기능단위"),
    ("system_boundary", "시스템 경계"),
    ("process_flow", "공정흐름도"),
    ("climate_change_total", "영향평가 결과 (Climate change_Total)"),
]


def build_candidates_context(ranked):
    """유사도 상위 후보들을 모델에 전달할 텍스트로 만듭니다.

    본문은 추천 사유 판단을 위한 내부 근거로만 전달하며,
    모델에는 원문을 인용/복사하지 말라고 별도로 지시합니다.
    """
    blocks = []
    for rank, (score, report) in enumerate(ranked, start=1):
        blocks.append(
            f"[후보 {rank}] LCI DB 이름: {report['db_name']}\n"
            f"유사도: {score:.3f}\n"
            f"관련 정보: {report['body']}"
        )
    return "\n\n".join(blocks)


def recommend(client, query, ranked, usage=None):
    """유사도 상위 후보를 LLM에 전달해 추천 결과(JSON)를 생성합니다.

    반환: {"data": dict|None, "raw": str}. data는 JSON 파싱 실패 시 None.
    usage(UsageTracker)를 주면 토큰을 기록합니다.
    """
    context = build_candidates_context(ranked)
    system_prompt = (
        "당신은 국가 LCI(전과정 목록분석) 데이터베이스 추천 도우미입니다. "
        "사용자의 검색어와 가장 유사한 'LCI DB'를 추천합니다.\n"
        "아래 후보는 임베딩 유사도로 미리 추려진 LCI DB이며, 각 후보의 db_name과 관련 정보가 주어집니다.\n"
        "규칙:\n"
        "- 반드시 주어진 후보 중에서만 고르고, db_name은 주어진 그대로(변형 없이) 사용하세요.\n"
        "- 관련 정보(본문)는 판단 근거로만 사용하고, 원문을 그대로 인용/복사하지 마세요.\n"
        "- 가장 적합한 1개를 recommended로, 나머지 유사 후보를 others로 제시하세요.\n"
        "- 모든 사유(reasons/reason)는 한국어 개조식(짧은 구절)으로 작성하세요.\n"
        "- 충분히 적합한 후보가 없으면 no_match를 true로 하세요.\n"
        "- 검색어가 가리키는 구체적 대상(수송 수단·연료·용수 종류·제품)이 후보에 없으면, 같은 큰 범주라도 "
        "비슷한 다른 항목으로 대체하지 말고 no_match를 true로 하세요. "
        "(예: 검색어가 지하철·트럭·수소차·농업용수인데 후보에 해당 항목이 없으면 기차·전기차·생활용수 등으로 대체 금지)\n"
        "출력은 다음 JSON 형식만 사용하세요:\n"
        '{"no_match": false, '
        '"recommended": {"db_name": "이름", "reasons": ["사유1", "사유2"]}, '
        '"others": [{"db_name": "이름", "reason": "사유"}]}'
    )
    user_prompt = (
        f"[유사도 상위 LCI DB 후보]\n{context}\n\n"
        f"[사용자 검색어]\n{query}\n\n"
        "위 후보를 바탕으로 JSON으로만 답하세요."
    )
    response = client.chat.completions.create(
        model=get_settings().model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    if usage is not None:
        usage.record("추천", get_settings().model, response)
    answer = response.choices[0].message.content
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        data = None
    return {"data": data, "raw": answer}


def summarize_detail(client, db_name, body):
    """보고서 본문에서 세부정보 항목별 요약을 생성합니다.

    client가 None이거나 파싱 실패 시 None을 반환합니다.
    캐싱(동일 DB 재호출 방지)은 UI 계층에서 감쌉니다.
    """
    if client is None:
        return None
    system_prompt = (
        "당신은 국가 LCI(전과정 목록분석) 보고서를 읽고 핵심 항목을 요약하는 도우미입니다.\n"
        "주어진 보고서 본문에서 아래 다섯 항목을 한국어 개조식(짧은 구절, 항목당 1~4줄)으로 요약하세요.\n"
        "- overview: 제품(모듈) 개요 — 어떤 제품/모듈에 대한 LCI 데이터인지\n"
        "- functional_unit: 기능단위 — 데이터의 기준 단위(예: 1 kg, 1 톤, 1 MJ 등)\n"
        "- system_boundary: 시스템 경계 — 포함/제외 공정 범위(예: cradle-to-gate)\n"
        "- process_flow: 공정흐름도 — 주요 공정 흐름·단계\n"
        "- climate_change_total: 영향평가 결과(Climate change_Total) — 기후변화 총량 수치와 단위(가능하면 숫자 포함)\n"
        "규칙:\n"
        "- 보고서에 근거해 사실만 요약하고, 없는 내용을 지어내지 마세요.\n"
        "- 본문에서 해당 항목을 찾을 수 없으면 값으로 '보고서에서 확인되지 않음'을 쓰세요.\n"
        "출력은 다음 JSON 형식만 사용하세요:\n"
        '{"overview": "...", "functional_unit": "...", "system_boundary": "...", '
        '"process_flow": "...", "climate_change_total": "..."}'
    )
    user_prompt = (
        f"[LCI DB 이름]\n{db_name}\n\n[보고서 본문]\n{body}\n\n"
        "위 항목을 JSON으로만 요약하세요."
    )
    resp = client.chat.completions.create(
        model=get_settings().model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return None
