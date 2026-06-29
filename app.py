import csv
import io
import json
import math
import os
import re
import struct
import zlib
from pathlib import Path

import olefile
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
load_dotenv()  # 프로젝트 폴더의 .env 파일을 읽어옵니다.

MODEL = "gpt-5.4-nano"                 # 추천 문구 생성용 모델
EMBEDDING_MODEL = "text-embedding-3-small"  # 유사도 검색용 임베딩 모델
TOP_K = 5                              # 모델에 넘길 유사 후보 보고서 개수

# '다른 유사 후보'를 노출할 최소 코사인 유사도.
# 임베딩 코사인은 무관한 문서끼리도 0이 되지 않으므로(보통 0.2~0.4), 0 대신
# 실측 분포의 빈틈(관련 후보 ≥0.42 vs 무관 후보 ≤0.31)에 기준값을 둡니다.
# 이 값 미만인 후보는 '유사하지 않음'으로 보고 노출하지 않습니다.
SIMILARITY_FLOOR = 0.40

# 임베딩 모델의 입력 한도는 8,191토큰입니다. 한 청크에 담을 토큰 수는
# 여유를 둬 8,000으로 잡고, 이를 넘는 긴 본문은 여러 청크로 나눠 임베딩한 뒤
# 토큰 수로 가중평균합니다(표가 많아 본문이 긴 보고서 대응).
EMBED_ENCODING = "cl100k_base"         # text-embedding-3-small이 쓰는 토크나이저
EMBED_MAX_TOKENS = 8000

# 보고서와 임베딩을 보관할 파일(앱을 재시작해도 유지됨).
# 내용이 비공개 자료이므로 .gitignore에 등록해 git에 올라가지 않게 합니다.
DATA_FILE = Path(__file__).parent / "lci_reports.json"

# 관리자가 보고서 파일을 넣어두는 업로드 폴더.
# 앱 시작 시(또는 '다시 스캔' 시) 이 폴더의 새 파일을 자동으로 학습합니다.
UPLOAD_DIR = Path(__file__).parent / "reports_upload"
UPLOAD_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTS = {".hwp", ".txt", ".md", ".csv", ".json"}

st.set_page_config(page_title="국가 LCI DB 검색", page_icon="🔎", layout="centered")

# ecosq.or.kr(에코스퀘어, KEITI 환경산업 포털)과 유사한 블루 톤 ------------------
st.markdown(
    """
    <style>
      /* 상단 타이틀 배너 */
      .lci-header {
        background: linear-gradient(135deg, #0B4F8A 0%, #2E86C1 100%);
        padding: 26px 30px; border-radius: 14px; margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(11,79,138,0.18);
      }
      .lci-header h1 { color:#ffffff; margin:0; font-size:27px; font-weight:800; letter-spacing:-0.5px; }
      .lci-header p  { color:#E3EEF8; margin:8px 0 0; font-size:14.5px; }
      /* 결과 카드 강조용 테두리 */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:#C5DBF0 !important;
      }
      /* 본문 헤더 색을 블루로 */
      h2, h3 { color:#0E5AA7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# page title 위쪽 공간, 오른쪽에 '국가 LCI DB 보고서 읽기' 버튼 배치
_, _btn_col = st.columns([5, 3])
read_clicked = _btn_col.button(
    "국가 LCI DB 보고서 읽기",
    use_container_width=True,
    type="primary",
    help="새 보고서 파일(.hwp 등)을 reports_upload/ 폴더에 넣은 뒤 누르세요.",
)

# 페이지 타이틀(배너)
st.markdown(
    """
    <div class="lci-header">
      <h1>🔎 국가 LCI DB 검색</h1>
      <p>검색어와 가장 유사한 국가 LCI DB를 추천해 드립니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 보고서 저장 공간 (파일에 영속 저장)
# 각 보고서는 특정 LCI DB에 대한 정보를 담고 있으며,
# 저장 시점에 임베딩을 계산해 함께 보관합니다(개수 제한 없음).
#   report = {"db_name": str, "body": str, "embedding": list[float]}
# ---------------------------------------------------------------------------
def load_reports():
    """저장 파일에서 보고서 목록을 읽어옵니다. 없거나 손상되면 빈 목록."""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            st.warning(f"저장 파일({DATA_FILE.name})을 읽지 못해 빈 목록으로 시작합니다.")
    return []


def save_reports(reports):
    """보고서 목록(임베딩 포함)을 파일에 저장합니다."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"보고서를 파일에 저장하지 못했습니다: {e}")


# ---------------------------------------------------------------------------
# OpenAI / 임베딩 유틸
# ---------------------------------------------------------------------------
def get_client():
    """OPENAI_API_KEY로 OpenAI 클라이언트를 만듭니다. 키가 없으면 None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_encoder():
    """임베딩 모델용 토크나이저(첫 호출 시 1회 로드 후 캐시)."""
    return tiktoken.get_encoding(EMBED_ENCODING)


def _average_vectors(vectors, weights):
    """여러 임베딩 벡터를 가중평균해 하나의 벡터로 합칩니다."""
    dim = len(vectors[0])
    total = sum(weights) or 1
    avg = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for k in range(dim):
            avg[k] += vec[k] * w
    return [x / total for x in avg]


def embed_text(client, text):
    """텍스트를 임베딩 벡터로 변환합니다.

    임베딩 모델의 토큰 한도(8,191)를 넘는 긴 본문은 8,000토큰씩 여러 청크로
    나눠 한 번의 API 호출로 각각 임베딩한 뒤, 청크의 토큰 수로 가중평균하여
    하나의 벡터로 합칩니다(긴 보고서가 한도 초과로 학습 실패하지 않도록).
    """
    enc = get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= EMBED_MAX_TOKENS:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding

    chunks = [tokens[i:i + EMBED_MAX_TOKENS]
              for i in range(0, len(tokens), EMBED_MAX_TOKENS)]
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[enc.decode(c) for c in chunks],
    )
    # 응답 순서를 index로 보장한 뒤, 청크 토큰 수를 가중치로 평균.
    vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    return _average_vectors(vectors, [len(c) for c in chunks])


def cosine_similarity(a, b):
    """두 벡터의 코사인 유사도(-1~1)를 계산합니다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_text(report):
    """임베딩 대상이 되는 텍스트(LCI DB 이름 + 보고서 내용)."""
    return f"{report['db_name']}\n{report['body']}"


# ---------------------------------------------------------------------------
# 업로드 폴더 자동 학습
#  - reports_upload/ 폴더의 파일을 읽어 [{db_name, body}, ...]로 변환
#  - 지원 형식:
#      .hwp       : 한글(HWP 5.0) 파일 1개 = LCI DB 1개 (파일명 = 이름, 본문 = 정보)
#      .txt / .md : 파일 1개 = LCI DB 1개 (파일명 = 이름, 내용 = 정보)
#      .csv       : 행 1개 = LCI DB 1개 (이름 열 + 내용 열 필요)
#      .json      : [{"db_name": ..., "body": ...}, ...] 또는 단일 객체
#  - 이미 학습한 (이름, 내용)은 건너뛰어 재임베딩 비용을 막습니다.
# ---------------------------------------------------------------------------
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
    """
    name = stem.replace("정밀검토보고서", "")
    # 일련번호: 공백/_/-/문자열경계로 둘러싸인 숫자 토큰만 제거
    # (CO2·PM10처럼 글자에 붙은 숫자, '1차' 같은 표현은 보존)
    name = re.sub(r"(?:^|(?<=[\s_\-]))\d+(?=$|[\s_\-])", "", name)
    name = re.sub(r"[ _\-]+", " ", name).strip(" _-")
    return name or stem


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
        return "\n".join(t for t in texts if t.strip()).strip()
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


def ingest_upload_folder(reports):
    """업로드 폴더를 스캔해 아직 학습하지 않은 보고서를 임베딩·추가합니다.

    reports 리스트를 직접 수정하고, 결과 요약 딕셔너리를 반환합니다.
    """
    existing = {(r["db_name"], r["body"]) for r in reports}
    files = sorted(
        p for p in UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        and p.stem.lower() != "readme"  # 폴더 안내용 README는 학습에서 제외
    )
    new_records = []
    errors = []
    for p in files:
        try:
            for rec in parse_report_file(p):
                key = (rec["db_name"], rec["body"])
                if key not in existing:
                    existing.add(key)
                    new_records.append(rec)
        except Exception as e:
            errors.append(f"{p.name}: {e}")

    result = {"scanned": len(files), "new": len(new_records),
              "added": 0, "failed": 0, "errors": errors, "no_key": False}
    if not new_records:
        return result

    client = get_client()
    if client is None:
        result["no_key"] = True
        return result

    for rec in new_records:
        try:
            rec["embedding"] = embed_text(client, search_text(rec))
            reports.append(rec)
            result["added"] += 1
        except Exception as e:
            result["failed"] += 1
            result["errors"].append(f"{rec['db_name']}: 임베딩 실패 ({e})")
    if result["added"]:
        save_reports(reports)
    return result


def show_ingest_result(res, initial=False):
    """학습 결과를 토스트로 알리고, 실패 내역은 세션에 저장해 화면에 남깁니다.

    initial=True는 세션 첫 로드 시의 자동 학습으로, '새로 읽을 보고서가 없음'
    같은 잡음성 안내는 띄우지 않습니다.
    """
    if res["no_key"]:
        st.toast("OPENAI_API_KEY가 없어 보고서를 읽지 못했습니다. .env를 확인하세요.", icon="⚠️")
    elif res["added"]:
        st.toast(f"새 보고서 {res['added']}건을 학습했습니다.", icon="✅")
    elif not initial and res["new"] == 0 and not res["errors"]:
        st.toast("새로 읽을 보고서가 없습니다.", icon="ℹ️")
    if res["failed"]:
        st.toast(f"{res['failed']}건은 학습하지 못했습니다(아래 경고 참고).", icon="⚠️")
    # 토스트는 곧 사라지므로, 실패/오류 내역은 세션에 보관해 화면에 지속 표시합니다.
    st.session_state.ingest_errors = res["errors"]


# 세션 첫 로드 시: 저장 파일을 읽고 업로드 폴더를 자동 학습합니다.
if "reports" not in st.session_state:
    st.session_state.reports = load_reports()
    with st.spinner("국가 LCI DB 보고서를 읽는 중입니다..."):
        res = ingest_upload_folder(st.session_state.reports)
    show_ingest_result(res, initial=True)

# 상단 '국가 LCI DB 보고서 읽기' 버튼 처리 (새로 추가된 파일 학습)
if read_clicked:
    with st.spinner("국가 LCI DB 보고서를 읽는 중입니다..."):
        res = ingest_upload_folder(st.session_state.reports)
    show_ingest_result(res)

# 학습하지 못한 보고서가 있으면(토스트가 사라져도) 화면에 계속 노출합니다.
_ingest_errors = st.session_state.get("ingest_errors")
if _ingest_errors:
    st.warning(f"보고서 {len(_ingest_errors)}건을 학습하지 못했습니다.")
    with st.expander("학습 실패 내역 보기"):
        st.markdown("\n".join(f"- {e}" for e in _ingest_errors))

# ---------------------------------------------------------------------------
# 메인: 검색어 입력
#  - 폼으로 감싸 입력칸에서 Enter를 치면 바로 검색되도록 합니다.
# ---------------------------------------------------------------------------
with st.form("search_form"):
    query = st.text_input(
        "적절한 LCI DB를 찾기위한 정보를 입력해보세요.",
        placeholder="예) 전기로 방식으로 만든 철강의 온실가스 배출 데이터가 필요해",
    )
    search = st.form_submit_button("검색하기", type="primary")


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


# ---------------------------------------------------------------------------
# 추천된 LCI DB의 세부정보 (보고서 본문을 항목별로 요약해 모달 창으로 표시)
# ---------------------------------------------------------------------------
# (요약 키, 화면 라벨) — 사용자가 요청한 5개 항목
DETAIL_FIELDS = [
    ("overview", "제품(모듈) 개요"),
    ("functional_unit", "기능단위"),
    ("system_boundary", "시스템 경계"),
    ("process_flow", "공정흐름도"),
    ("climate_change_total", "영향평가 결과 (Climate change_Total)"),
]


@st.cache_data(show_spinner=False)
def summarize_detail(db_name, body):
    """보고서 본문에서 세부정보 항목별 요약을 생성합니다.

    (db_name, body) 단위로 캐시되어 같은 DB를 다시 열어도 재호출하지 않습니다.
    실패 시 None을 반환합니다.
    """
    client = get_client()
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
        model=MODEL,
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


@st.dialog("LCI DB 세부정보", width="large")
def show_detail_dialog(db_name):
    """추천된 LCI DB의 보고서를 항목별로 요약해 모달 창에 표시합니다."""
    st.markdown(f"### {db_name}")
    body = next(
        (r["body"] for r in st.session_state.reports if r["db_name"] == db_name),
        None,
    )
    if not body:
        st.warning("이 LCI DB의 보고서 본문을 찾을 수 없습니다.")
        return
    with st.spinner("보고서에서 세부정보를 정리하는 중입니다..."):
        detail = summarize_detail(db_name, body)
    if detail is None:
        st.error("세부정보를 생성하지 못했습니다. (OPENAI_API_KEY/네트워크를 확인하세요)")
        return
    for key, label in DETAIL_FIELDS:
        st.markdown(f"**■ {label}**")
        st.markdown(detail.get(key) or "보고서에서 확인되지 않음")
        st.markdown("")  # 항목 간 여백


# ---------------------------------------------------------------------------
# 검색 → 임베딩 유사도 → LCI DB 추천
#  - 결과는 세션에 저장합니다(세부정보 창을 열어 화면이 새로고침돼도 결과 유지).
# ---------------------------------------------------------------------------
def run_search(query):
    """검색을 실행하고 렌더링에 필요한 결과 dict를 반환합니다."""
    client = get_client()
    if client is None:
        return {"error": "no_key"}
    if not st.session_state.reports:
        return {"error": "no_reports"}

    # 1) 검색어 임베딩 → 2) 모든 보고서와 코사인 유사도 → 상위 후보 추림
    query_embedding = embed_text(client, query)
    scored = [
        (cosine_similarity(query_embedding, r["embedding"]), r)
        for r in st.session_state.reports
        if r.get("embedding")
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = scored[:TOP_K]
    if not ranked:
        return {"error": "no_embeddings"}

    # 3) 상위 후보만 모델에 전달하여 추천 결과를 JSON(개조식 렌더링용)으로 생성
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
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    answer = response.choices[0].message.content
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        data = None
    # 후보별 실제 코사인 유사도(유사도 0인 '다른 후보'를 거르는 데 사용)
    return {"data": data, "raw": answer,
            "scores": {r["db_name"]: s for s, r in ranked}}


def render_search_result(result):
    """저장된 검색 결과를 화면에 그립니다(대화형이 아닌 개조식)."""
    err = result.get("error")
    if err == "no_key":
        st.error(
            "OPENAI_API_KEY를 찾을 수 없습니다. "
            "프로젝트 폴더의 .env 파일에 OPENAI_API_KEY를 설정했는지 확인해 주세요."
        )
        return
    if err == "no_reports":
        st.warning(
            f"학습된 보고서가 없습니다. `{UPLOAD_DIR.name}/` 폴더에 보고서 파일을 넣고 "
            "오른쪽 위의 '국가 LCI DB 보고서 읽기' 버튼을 눌러 학습해 주세요. "
            "추천은 학습된 LCI DB 중에서만 이루어집니다."
        )
        return
    if err == "no_embeddings":
        st.info("유사도를 계산할 수 있는 보고서가 없습니다. 보고서를 다시 저장해 주세요.")
        return
    if err == "exception":
        st.error(
            "추천을 생성하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n"
            "확인할 사항:\n"
            "- 인터넷 연결 상태\n"
            "- .env의 OPENAI_API_KEY 값이 올바른지\n"
            f"- 모델명('{MODEL}'), 임베딩 모델명('{EMBEDDING_MODEL}')이 사용 가능한지"
        )
        with st.expander("자세한 오류 내용 (개발자용)"):
            st.code(result.get("detail", ""))
        return

    data = result.get("data")
    scores = result.get("scores", {})

    st.subheader("추천 국가 LCI DB")
    if data is None:
        st.write(result.get("raw", ""))  # JSON 파싱 실패 시 원문 표시
        return
    if data.get("no_match"):
        st.warning("검색어에 충분히 부합하는 국가 LCI DB가 없습니다.")
        return

    rec = data.get("recommended") or {}
    rec_name = rec.get("db_name", "(이름 없음)")

    with st.container(border=True):
        st.markdown("**■ DB명**  (아래 버튼을 누르면 세부정보를 볼 수 있습니다)")
        if st.button(f"🔎 {rec_name}", key="rec_detail_btn",
                     use_container_width=True):
            show_detail_dialog(rec_name)
        st.markdown("**■ 추천 사유**")
        reasons = rec.get("reasons") or []
        st.markdown(
            "\n".join(f"- {r}" for r in reasons) if reasons else "- (사유 정보 없음)"
        )

    # 다른 유사 후보: 유사도가 기준값(SIMILARITY_FLOOR) 미만이면 노출하지 않고,
    # 남는 후보가 없으면 "다른 유사 후보가 없습니다." 안내문을 표시합니다.
    others = data.get("others") or []
    visible_others = [
        o for o in others
        if scores.get(o.get("db_name", ""), 0) >= SIMILARITY_FLOOR
    ]
    st.markdown("**■ 다른 유사 후보**")
    if visible_others:
        with st.container(border=True):
            st.markdown("\n".join(
                f"- **{o.get('db_name', '')}** — {o.get('reason', '')}"
                for o in visible_others
            ))
    else:
        st.info("다른 유사 후보가 없습니다.")


# 검색 실행: 결과를 세션에 저장(세부정보 창을 열어도 결과가 유지되도록)
if search:
    if not query.strip():
        st.warning("검색어를 입력한 뒤 '검색하기'를 눌러 주세요.")
    else:
        try:
            with st.spinner("가장 적합한 국가 LCI DB를 찾는 중입니다..."):
                st.session_state.search_result = run_search(query.strip())
        except Exception as e:
            st.session_state.search_result = {"error": "exception", "detail": str(e)}

# 저장된 검색 결과 표시
if st.session_state.get("search_result"):
    render_search_result(st.session_state.search_result)
