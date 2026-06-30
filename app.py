"""국가 LCI DB 검색 — Streamlit UI (읽기 전용 서빙).

검색·생성 로직은 rag 패키지에 있고, 이 파일은 화면만 담당합니다. 인덱스는 오프라인
빌드(scripts/build_index.py) 산출물(index/)을 읽기 전용으로 로드합니다(서빙 중 쓰기 없음).
"""
import logging

import streamlit as st
from dotenv import load_dotenv

from rag.config import get_settings
from rag.generate import DETAIL_FIELDS
from rag.pipeline import search as run_search
from rag.pipeline import visible_others
from rag.store import load_index

# 서버 콘솔에 각 검색 단계(임베딩→하이브리드→리랭커→추천)의 타이밍·결과를 로깅합니다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logging.getLogger("rag").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
# 프로젝트 폴더의 .env를 읽어옵니다. override=True는 .env 값이 셸 환경변수보다
# 우선하도록 하여, 키를 .env 한 곳에서만 관리할 수 있게 합니다(셸의 오래된 키에 가려지지 않음).
load_dotenv(override=True)

settings = get_settings()

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

# page title 위쪽 공간, 오른쪽에 '인덱스 다시 불러오기' 버튼 배치(읽기 전용 재로드)
_, _btn_col = st.columns([5, 3])
reload_clicked = _btn_col.button(
    "인덱스 다시 불러오기",
    use_container_width=True,
    type="primary",
    help="새 index/ 아티팩트를 업로드한 뒤 누르면 디스크에서 다시 읽어옵니다.",
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


# 세션 첫 로드 시: 인덱스 아티팩트를 읽기 전용으로 로드합니다.
if "reports" not in st.session_state:
    with st.spinner("인덱스를 불러오는 중입니다..."):
        st.session_state.reports = load_index()

# '인덱스 다시 불러오기' 버튼: 디스크에서 다시 읽어옵니다.
if reload_clicked:
    with st.spinner("인덱스를 다시 불러오는 중입니다..."):
        st.session_state.reports = load_index()
    st.toast(f"인덱스를 불러왔습니다(LCI DB {len(st.session_state.reports)}개).", icon="✅")

# ---------------------------------------------------------------------------
# 메인: 검색어 입력
#  - 폼으로 감싸 입력칸에서 Enter를 치면 바로 검색되도록 합니다.
# ---------------------------------------------------------------------------
with st.form("search_form"):
    query = st.text_input(
        "적절한 LCI DB를 찾기위한 정보를 입력해보세요.",
        placeholder="예) 디젤 기차로 화물을 수송할 때 온실가스 배출 데이터가 필요해",
    )
    search = st.form_submit_button("검색하기", type="primary")


# ---------------------------------------------------------------------------
# 추천된 LCI DB의 세부정보 (보고서 본문을 항목별로 요약해 모달 창으로 표시)
#  - 동일 DB를 다시 열어도 재호출하지 않도록 UI 계층에서 캐시합니다.
# ---------------------------------------------------------------------------
@st.dialog("LCI DB 세부정보", width="large")
def show_detail_dialog(db_name):
    """추천된 LCI DB의 세부정보를 모달 창에 표시합니다(빌드 시점 precompute된 metadata 사용)."""
    st.markdown(f"### {db_name}")
    detail = next(
        (r.get("metadata") for r in st.session_state.reports if r["db_name"] == db_name),
        None,
    )
    if not detail:
        st.warning("이 LCI DB의 세부정보가 인덱스에 없습니다(빌드를 다시 실행해 주세요).")
        return
    for key, label in DETAIL_FIELDS:
        st.markdown(f"**■ {label}**")
        st.markdown(detail.get(key) or "보고서에서 확인되지 않음")
        st.markdown("")  # 항목 간 여백


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
            "인덱스가 비어 있습니다. `reports_upload/`에 보고서 파일을 넣고 "
            "`uv run python scripts/build_index.py`로 인덱스를 빌드한 뒤, "
            "오른쪽 위 '인덱스 다시 불러오기'를 눌러 주세요."
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
            f"- 모델명('{settings.model}'), 임베딩 모델명('{settings.embedding_model}')이 사용 가능한지"
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
        # 출처(provenance): 추천 DB의 보고서에서 질의와 가장 일치하는 근거 본문 발췌
        evidence = (result.get("evidence") or {}).get(rec_name)
        if evidence and evidence.get("text"):
            st.markdown("**■ 근거 (출처 본문 발췌)**")
            st.caption(f"추천 DB 보고서에서 검색어와 가장 일치하는 구절 · 코사인 {evidence['score']}")
            snippet = evidence["text"].strip().replace("\n", " ")
            st.markdown(f"> {snippet[:300]}{'…' if len(snippet) > 300 else ''}")

    # 다른 유사 후보: 임계 min(floor, 최고점수×others_ratio) 미만이면 노출하지 않고,
    # 남는 후보가 없으면 "다른 유사 후보가 없습니다." 안내문을 표시합니다.
    # (짧은/제너럴 질의는 코사인이 전반적으로 낮아도 추천에 견줘 비슷한 후보를 보여줌)
    others = data.get("others") or []
    visible = visible_others(others, scores, settings.similarity_floor, settings.others_ratio)
    st.markdown("**■ 다른 유사 후보**")
    if visible:
        with st.container(border=True):
            st.markdown("\n".join(
                f"- **{o.get('db_name', '')}** — {o.get('reason', '')}"
                for o in visible
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
                st.session_state.search_result = run_search(
                    st.session_state.reports, query.strip()
                )
        except Exception as e:
            st.session_state.search_result = {"error": "exception", "detail": str(e)}

def render_trace(trace):
    """검색 각 단계(임베딩→하이브리드→리랭커→추천)의 입출력·타이밍·토큰·비용을 펼침 패널로 표시."""
    tok, cost = trace.get("tokens", 0), trace.get("cost_usd", 0.0)
    with st.expander(f"🔍 검색 과정 로깅 — 총 {trace['total_ms']}ms · 토큰 {tok} · ${cost:.5f}"):
        for s in trace["stages"]:
            st.markdown(f"**{s['name']}** · `{s['ms']}ms` — {s['detail']}")
            if s.get("top"):
                st.markdown("\n".join(
                    f"  - {n}  (코사인 {sc})" for n, sc in s["top"]))
            if s.get("before") and s.get("after"):
                st.markdown(f"  - 리랭크 전(top5): {', '.join(s['before'])}")
                st.markdown(f"  - 리랭크 후(top{len(s['after'])}): {', '.join(s['after'])}")
        usage = trace.get("usage")
        if usage:
            st.markdown("**토큰·비용 (OpenAI usage)**")
            st.markdown("\n".join(
                f"  - {u['label']} ({u['model']}): 입력 {u['in']} · 출력 {u['out']} 토큰"
                for u in usage))
            st.caption(f"합계 {tok} 토큰 · 추정 ${cost:.5f}  (단가: rag/usage.py PRICES — 실제 단가로 조정)")


# 저장된 검색 결과 표시
if st.session_state.get("search_result"):
    render_search_result(st.session_state.search_result)
    _trace = st.session_state.search_result.get("trace")
    if _trace:
        render_trace(_trace)
