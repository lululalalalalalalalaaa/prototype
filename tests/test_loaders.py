"""로더 검증 — 순수 파싱 로직 + 실제 .hwp 37개에 대한 회귀 안전망(API 불필요)."""
import json

import pytest

from rag.config import SUPPORTED_EXTS, UPLOAD_DIR
from rag.ingest.loaders import clean_db_name, parse_report_file


# --- clean_db_name: 일련번호 제거 / 글자에 붙은 숫자 보존 / 보고서 표기 제거 ---
@pytest.mark.parametrize("stem, expected", [
    ("001 정밀검토보고서_경유 승용차 수송", "경유 승용차 수송"),
    ("철강 전기로 평균 데이터", "철강 전기로 평균 데이터"),
    ("CO2 PM10", "CO2 PM10"),                       # 글자에 붙은 숫자는 보존
    ("121 생활용수(제주권) 001 정밀검토보고서", "생활용수(제주권)"),
])
def test_clean_db_name(stem, expected):
    assert clean_db_name(stem) == expected


def test_clean_db_name_all_digits_falls_back_to_stem():
    # 숫자만 남으면 원본 stem을 유지(빈 이름 방지)
    assert clean_db_name("12345") == "12345"


def test_clean_db_name_normalizes_macos_nfd():
    # macOS 파일명은 NFD(자모 분리형). NFC 리터럴 매칭이 실패하지 않아야 함.
    import unicodedata
    nfd = unicodedata.normalize("NFD", "001 정밀검토보고서_전기 승용차 수송")
    assert nfd != "001 정밀검토보고서_전기 승용차 수송"   # 실제로 NFD라 다름
    assert clean_db_name(nfd) == "전기 승용차 수송"


# --- 텍스트 계열 파싱 ---
def test_parse_txt(tmp_path):
    f = tmp_path / "철강 전기로 평균 데이터.txt"
    f.write_text("전기로 방식 조강 생산의 온실가스 배출 계수", encoding="utf-8")
    recs = parse_report_file(f)
    assert recs == [{"db_name": "철강 전기로 평균 데이터",
                     "body": "전기로 방식 조강 생산의 온실가스 배출 계수"}]


def test_parse_csv_korean_headers(tmp_path):
    f = tmp_path / "list.csv"
    f.write_text("db_name,body\n철강,조강 배출\n시멘트,클링커 배출\n", encoding="utf-8")
    recs = parse_report_file(f)
    assert recs == [
        {"db_name": "철강", "body": "조강 배출"},
        {"db_name": "시멘트", "body": "클링커 배출"},
    ]


def test_parse_json_list_and_alias_keys(tmp_path):
    f = tmp_path / "list.json"
    f.write_text(json.dumps([{"이름": "철강", "내용": "조강 배출"}]), encoding="utf-8")
    recs = parse_report_file(f)
    assert recs == [{"db_name": "철강", "body": "조강 배출"}]


def test_parse_skips_empty_body(tmp_path):
    f = tmp_path / "빈 보고서.txt"
    f.write_text("   \n  ", encoding="utf-8")
    assert parse_report_file(f) == []


# --- 실제 HWP 37개: 동작 보존 회귀 안전망 (API 키 불필요) ---
_HWP_FILES = sorted(
    p for p in UPLOAD_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".hwp"
) if UPLOAD_DIR.exists() else []


@pytest.mark.skipif(not _HWP_FILES, reason="reports_upload/에 .hwp 파일 없음")
@pytest.mark.parametrize("path", _HWP_FILES, ids=lambda p: p.name)
def test_parse_real_hwp_yields_one_nonempty_record(path):
    recs = parse_report_file(path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["db_name"].strip()        # 이름 비어있지 않음
    assert len(rec["body"]) > 50         # 본문이 실제로 추출됨
    assert "\x00" not in rec["body"]     # 제어문자 누수 없음


@pytest.mark.skipif(not _HWP_FILES, reason="reports_upload/에 .hwp 파일 없음")
def test_supported_exts_covers_hwp():
    assert ".hwp" in SUPPORTED_EXTS
