"""파이프라인 보조 로직 검증 — others 노출 필터(상대 임계). API 불필요."""
from rag.pipeline import visible_others


def _o(*names):
    return [{"db_name": n, "reason": "r"} for n in names]


def test_others_above_floor_shown():
    # 일반 질의: 점수 높음 → 절대 floor로 노출(기존 동작 유지)
    vis = visible_others(_o("B"), {"A": 0.7, "B": 0.5}, floor=0.4, ratio=0.85)
    assert [o["db_name"] for o in vis] == ["B"]


def test_general_query_relative_shows_close_other():
    # 짧은/제너럴 질의: 점수 다 낮지만 other가 추천(최고)에 근접 → 노출
    # 임계 = min(0.4, 0.358*0.85=0.304)=0.304, 0.352 >= 0.304
    scores = {"여객디젤": 0.358, "화물디젤": 0.352}
    vis = visible_others(_o("화물디젤"), scores, floor=0.4, ratio=0.85)
    assert [o["db_name"] for o in vis] == ["화물디젤"]


def test_far_other_hidden():
    # 최고 점수에서 멀리 떨어진 무관 후보 → 숨김 (0.3 < min(0.4, 0.51)=0.4)
    vis = visible_others(_o("junk"), {"A": 0.6, "junk": 0.3}, floor=0.4, ratio=0.85)
    assert vis == []


def test_never_tightens_vs_absolute_floor():
    # 임계는 항상 floor 이하 → 절대 floor로 통과하던 후보는 늘 통과(비회귀)
    vis = visible_others(_o("B"), {"A": 0.9, "B": 0.45}, floor=0.4, ratio=0.85)
    assert [o["db_name"] for o in vis] == ["B"]


def test_empty_others():
    assert visible_others([], {"A": 0.5}, floor=0.4, ratio=0.85) == []
