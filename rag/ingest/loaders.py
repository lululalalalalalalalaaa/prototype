"""보고서 파일 로더 — 파일 1개를 [{db_name, body}, ...]로 변환합니다.

지원 형식:
  .hwp       : 한글(HWP 5.0) 파일 1개 = LCI DB 1개 (파일명 = 이름, 본문 = 정보)
  .txt / .md : 파일 1개 = LCI DB 1개 (파일명 = 이름, 내용 = 정보)
  .csv       : 행 1개 = LCI DB 1개 (이름 열 + 내용 열 필요)
  .json      : [{"db_name": ..., "body": ...}, ...] 또는 단일 객체

전부 순수 함수(파일/문자열 in → 레코드 out)라 API 키 없이 단독 테스트가 가능합니다.
원본 app.py에서 로직을 그대로 옮겼습니다.
"""
import csv
import io
import json
import re
import struct
import unicodedata
import zlib

import olefile

_NAME_KEYS = {"db_name", "name", "lci_db", "db", "이름", "lci db 이름", "lci db", "데이터베이스"}
_BODY_KEYS = {"body", "content", "text", "내용", "정보", "보고서", "보고서 내용", "본문"}


def _pick(row, keys):
    """딕셔너리에서 후보 키(대소문자/공백 무시) 중 첫 값을 반환합니다."""
    for k, v in row.items():
        if k is not None and str(k).strip().lower() in keys:
            return v
    return None


def clean_db_name(stem):
    """파일명에서 LCI DB 이름을 추출합니다.

    - '정밀검토보고서' 같은 보고서 종류 표기는 이름에서 제외
    - '001' 같은 일련번호(숫자)는 이름으로 인식하지 않음 (단, CO2·PM10 등
      글자에 붙은 숫자는 보존)
    - 남은 구분자(공백·_·-)를 정리

    macOS는 한글 파일명을 NFD(자모 분리형)로 저장하므로, NFC(완성형)로 정규화한 뒤
    처리한다. 정규화하지 않으면 '정밀검토보고서' 같은 리터럴(NFC) 매칭이 실패한다.
    """
    stem = unicodedata.normalize("NFC", stem)
    name = stem.replace("정밀검토보고서", "")
    # 일련번호: 공백/_/-/문자열경계로 둘러싸인 숫자 토큰만 제거
    # (CO2·PM10처럼 글자에 붙은 숫자, '1차' 같은 표현은 보존)
    name = re.sub(r"(?:^|(?<=[\s_\-]))\d+(?=$|[\s_\-])", "", name)
    name = re.sub(r"[ _\-]+", " ", name).strip(" _-")
    return name or stem


# --- 추출 본문 정제(HWP 폼 노이즈 제거) -------------------------------------
# HWP 보고서는 폼 양식이라 체크박스·수식 자리표시자가 텍스트로 함께 추출된다.
# 미선택 체크박스(□○)는 '고르지 않은 옵션'이라 검색 노이즈(잘못된 매칭 유발)이고,
# 선택 체크박스(■▣●)는 '고른 값'이라 의미가 있다(마커만 떼고 보존).
_EMPTY_BOX = "□○☐"          # 미선택(노이즈)
_FILL_BOX = "■▣●◉☑✓◆"      # 선택(내용 보존)


def clean_body_text(text):
    """추출된 본문에서 HWP 폼 노이즈를 정리합니다(실제 내용·표·수치는 보존).

    - 미선택 체크박스(□○)로 시작하는 줄: 제거(고르지 않은 옵션).
    - 선택 체크박스(■▣●)로 시작: 마커만 떼고 선택된 값 보존.
    - 단독 `(수식)`·빈 괄호·기호만 있는 줄: 제거.
    표 수치·짧은 토큰(디젤·1차 등)은 건드리지 않는다.
    """
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if s[0] in _EMPTY_BOX:            # 미선택 옵션 → 제거
            continue
        if s[0] in _FILL_BOX:             # 선택 옵션 → 마커 제거, 값 보존
            s = s[1:].strip()
            if not s:
                continue
        if s == "(수식)" or s.strip("()=·•◦▶▷-– \t") == "":   # 빈 괄호/단독 기호 → 제거
            continue
        kept.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


# --- .hwp(한글, HWP 5.0) 본문 텍스트 추출 -----------------------------------
# .hwp는 OLE 복합 파일이며, 본문은 BodyText/Section* 스트림에 들어 있습니다.
# (FileHeader의 압축 플래그가 켜져 있으면 zlib raw-deflate로 압축되어 있음)
# 문단 텍스트(HWPTAG_PARA_TEXT)는 UTF-16LE 문자열 + 제어문자로 구성되며,
# 제어문자는 8워드(표·그림 등 인라인/확장 제어) 또는 1워드(단순 제어)를 차지합니다.
_HWPTAG_PARA_TEXT = 67
_HWP_CHAR_CTRL = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}          # 1워드
_HWP_INLINE_CTRL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23}            # 8워드(16바이트)


def _hwp_decode_para(rec):
    """문단 텍스트 레코드에서 제어문자를 걸러 순수 텍스트만 추출합니다."""
    out = []
    j, m = 0, len(rec)
    while j + 2 <= m:
        code = struct.unpack_from("<H", rec, j)[0]
        if code in _HWP_INLINE_CTRL:      # 인라인/확장 제어: 8워드 통째로 건너뜀
            j += 16
        elif code in _HWP_CHAR_CTRL:      # 단순 제어: 1워드
            if code in (10, 13):          # 줄/문단 바꿈
                out.append("\n")
            j += 2
        else:
            out.append(chr(code))
            j += 2
    return "".join(out)


def _hwp_section_text(buf):
    """압축 해제된 BodyText 섹션에서 모든 문단 텍스트를 추출합니다."""
    parts = []
    i, n = 0, len(buf)
    while i + 4 <= n:
        header = struct.unpack_from("<I", buf, i)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4
        if size == 0xFFF:                 # 확장 크기: 다음 4바이트가 실제 크기
            size = struct.unpack_from("<I", buf, i)[0]
            i += 4
        rec = buf[i:i + size]
        i += size
        if tag_id == _HWPTAG_PARA_TEXT:
            parts.append(_hwp_decode_para(rec))
    return "\n".join(p for p in parts if p.strip())


def extract_hwp_text(path):
    """.hwp(HWP 5.0) 파일에서 본문 텍스트를 추출합니다."""
    if not olefile.isOleFile(str(path)):
        raise ValueError("HWP 5.0 형식이 아닙니다(.hwpx 또는 손상된 파일일 수 있음).")
    ole = olefile.OleFileIO(str(path))
    try:
        if not ole.exists("FileHeader"):
            raise ValueError("HWP FileHeader 스트림이 없습니다.")
        header = ole.openstream("FileHeader").read()
        is_compressed = bool(header[36] & 0x01)
        sections = sorted(
            (e for e in ole.listdir()
             if len(e) > 1 and e[0] == "BodyText" and e[1].startswith("Section")),
            key=lambda e: int(e[1][len("Section"):]),
        )
        texts = []
        for entry in sections:
            data = ole.openstream(entry).read()
            if is_compressed:
                data = zlib.decompress(data, -15)
            texts.append(_hwp_section_text(data))
        return clean_body_text("\n".join(t for t in texts if t.strip()))
    finally:
        ole.close()


def parse_report_file(path):
    """보고서 파일 하나를 [{db_name, body}, ...] 목록으로 변환합니다."""
    ext = path.suffix.lower()

    # .hwp(한글)는 바이너리(OLE)이므로 전용 추출기를 사용합니다.
    if ext == ".hwp":
        body = extract_hwp_text(path).strip()
        return [{"db_name": clean_db_name(path.stem), "body": body}] if body else []

    # 텍스트 계열(.txt/.md/.csv/.json)은 인코딩을 추정해 디코딩합니다.
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    records = []
    if ext in (".txt", ".md"):
        body = text.strip()
        if body:
            records.append({"db_name": clean_db_name(path.stem), "body": body})
    elif ext == ".csv":
        for row in csv.DictReader(io.StringIO(text)):
            db_name = _pick(row, _NAME_KEYS)
            body = _pick(row, _BODY_KEYS)
            if db_name and body and str(body).strip():
                records.append({"db_name": str(db_name).strip(), "body": str(body).strip()})
    elif ext == ".json":
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                db_name = _pick(item, _NAME_KEYS)
                body = _pick(item, _BODY_KEYS)
                if db_name and body and str(body).strip():
                    records.append({"db_name": str(db_name).strip(), "body": str(body).strip()})
    return records
