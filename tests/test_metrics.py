"""eval 채점 로직 검증 — recall_at_k(any/all) · reciprocal_rank. API 불필요."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_eval)

rr = run_eval.reciprocal_rank
recall = run_eval.recall_at_k


def test_reciprocal_rank_first_position():
    assert rr(["A", "B", "C"], ["A"]) == 1.0


def test_reciprocal_rank_third_position():
    assert rr(["X", "Y", "A"], ["A"]) == 1.0 / 3


def test_reciprocal_rank_not_found():
    assert rr(["X", "Y"], ["A"]) == 0.0


def test_reciprocal_rank_first_relevant_counts():
    # 여러 정답 중 가장 먼저 나온 것의 순위
    assert rr(["X", "B", "A"], ["A", "B"]) == 1.0 / 2


def test_recall_any_found():
    # any: 기대 중 하나라도 retrieved에 있으면 1.0
    assert recall(["X", "B"], ["A", "B"], "any") == 1.0


def test_recall_any_none():
    assert recall(["X", "Y"], ["A", "B"], "any") == 0.0


def test_recall_all_partial():
    # all: 교집합/expected
    assert recall(["A", "X"], ["A", "B"], "all") == 0.5


def test_recall_all_full():
    assert recall(["A", "B", "C"], ["A", "B"], "all") == 1.0


def test_recall_all_single():
    assert recall(["A"], ["A"], "all") == 1.0


def test_recall_empty_expected():
    # no_match(정답 없음)는 recall 정의 안 됨 → 0.0
    assert recall(["A"], [], "all") == 0.0
