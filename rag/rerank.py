"""LLM 리랭커 — hybrid 후보를 관련도 순으로 재정렬.

기존 OpenAI 클라이언트를 재사용한다(새 의존성·벤더 없음). 토큰 절감을 위해 후보의
db_name + 본문 짧은 발췌만 전달한다. LLM 출력 파싱 실패/번호 누락 시 **원래 순서로 폴백**해
품질이 급락하거나 크래시하지 않게 한다.

후보·반환 형태는 retrieve.hybrid_rank와 동일한 [(dense_score, report), ...] (점수는 dense 코사인).
"""
import json

from rag.config import get_settings

def _build_candidates_text(candidates):
    snippet_len = get_settings().rerank_snippet
    lines = []
    for i, (_, report) in enumerate(candidates):
        snippet = report["body"].replace("\n", " ")[:snippet_len]
        lines.append(f"[{i}] {report['db_name']} — {snippet}")
    return "\n".join(lines)


def rerank(client, query, candidates, top_k, usage=None):
    """candidates를 LLM 관련도 순으로 재정렬해 상위 top_k를 반환합니다.

    candidates: [(dense_score, report), ...] (hybrid_rank 출력).
    실패 시 입력 순서를 유지합니다(강건). usage를 주면 토큰을 기록합니다.
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    context = _build_candidates_text(candidates)
    system_prompt = (
        "당신은 검색 결과를 관련도 순으로 재정렬하는 도우미입니다.\n"
        "사용자 검색어와 후보 LCI DB 목록(번호: 이름 — 발췌)이 주어집니다.\n"
        "검색어에 가장 적합한 순서로 후보 번호를 나열하세요.\n"
        "규칙:\n"
        "- 주어진 모든 후보 번호를 한 번씩 포함하세요.\n"
        "- 표면 글자뿐 아니라 의미·동의어를 고려하세요(예: '전라남도'='전남', '관광버스'='전세버스').\n"
        '출력은 다음 JSON만 사용하세요: {"ranking": [번호, 번호, ...]}'
    )
    user_prompt = (
        f"[검색어]\n{query}\n\n[후보]\n{context}\n\n관련도 순으로 번호를 JSON으로 답하세요."
    )
    try:
        resp = client.chat.completions.create(
            model=get_settings().model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        if usage is not None:
            usage.record("리랭커", get_settings().model, resp)
        order = json.loads(resp.choices[0].message.content)["ranking"]
    except Exception:
        return candidates[:top_k]  # 폴백: 원래 순서

    # 유효 인덱스만, 중복 제거. 누락된 후보는 원래 순서대로 뒤에 보존.
    seen, reordered = set(), []
    for idx in order:
        try:
            idx = int(idx)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            reordered.append(candidates[idx])
    for i, c in enumerate(candidates):
        if i not in seen:
            reordered.append(c)
    return reordered[:top_k]
