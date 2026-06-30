# 국가 LCI DB 검색 (RAG)

[![CI](https://github.com/lululalalalalalalalaaa/prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/lululalalalalalalalaaa/prototype/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?logo=streamlit&logoColor=white)
![rerank MRR](https://img.shields.io/badge/rerank%20MRR-0.972-success)

**자연어로 물어보면 가장 알맞은 국가 LCI(전과정 목록분석) DB를 출처와 함께 찾아주는 검색 도구입니다.**

국가 LCI DB(제품 1단위의 온실가스 등 환경영향 데이터)는 종류가 많고 이름이 비슷해 찾기 어렵습니다.
평범한 말로 입력하면 가장 맞는 DB를 추천하고, 보고서 핵심(기능단위·시스템경계·영향평가 수치)을 요약합니다.

```
입력:  디젤 기차로 사람을 수송할 때 온실가스 배출
──────────────────────────────────────────────────
추천:  여객수송용 디젤기차 수송
출처:  「여객수송용 디젤기차 수송」 보고서 · 📑 공정흐름도(그림)   ← 보고서 그림을 vision으로 읽음
세부:  기능단위 1 person·km · gate-to-gate · Climate change_Total 4.95E-02 kg CO2 eq
```

> 데이터에 없는 주제(항공·철강 등)는 억지로 고르지 않고 **"적합 DB 없음"으로 정직하게** 답합니다.

## 빠른 시작

```bash
uv sync                                   # 1) 의존성 설치
cp .env.example .env                      # 2) .env에 OPENAI_API_KEY= 입력
uv run python scripts/build_index.py      # 3) reports_upload/ 보고서 → index/ 빌드(오프라인, 1회)
uv run streamlit run app.py               # 4) 앱 실행
```

## 핵심

- **하이브리드 검색 + LLM 리랭커·추천** — Dense(임베딩)+BM25를 융합해 LLM이 재정렬·추천. 105문항으로 측정(rerank MRR 0.972, off-domain 기권 1.000).
- **구조를 보는 인제스션** — 섹션·표 단위로 청킹하고, 공정흐름도 **이미지를 vision으로** 텍스트화해 함께 색인.
- **검증 가능한 출처** — *어느 문서 어느 섹션*인지 표시하고, 추천 DB를 누르면 **보고서 원문**까지 펼쳐 확인.
- **build/serve 분리** — 비용은 빌드 1회. 서빙은 `index/` 파일만 읽어 **벡터DB 서버 없이** 배포.

## 운영 (데이터를 바꿀 때)

1. `reports_upload/`에 보고서 추가/교체 (파일명 = DB 이름)
2. `uv run python scripts/build_index.py` — 변경분만 재빌드(증분). 청킹 로직을 바꿨다면 `index/`를 비우고 풀빌드.
3. `git add index/ && git commit && git push` — 배포는 이 `index/`를 사용. 앱은 우측 상단 **"인덱스 다시 불러오기"**.

배포는 [Streamlit Cloud](https://streamlit.io/cloud)에 repo를 연결(`app.py` 지정, `OPENAI_API_KEY`를 Secrets)하면 push마다 자동입니다.

## 더 알아보기

- **설계·아키텍처·평가·프로젝트 구조** → [`CLAUDE.md`](CLAUDE.md)
- 작업 이력·측정값 → [`logging.md`](logging.md) · 다음 할 일 → [`Nextsession.md`](Nextsession.md)
- 보고서 입력 형식(`.hwp`·txt·csv·json) → [`reports_upload/README.md`](reports_upload/README.md)
