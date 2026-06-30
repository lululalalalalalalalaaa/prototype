# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

국가 LCI(전과정 목록분석) DB 검색 Streamlit 앱. 사용자가 자연어 검색어를 입력하면
임베딩 유사도로 후보 LCI DB를 추리고, LLM이 그중 가장 적합한 1개와 다른 유사 후보를
개조식으로 추천한다. 추천된 DB는 모달 창에서 보고서 본문을 5개 항목으로 요약해 보여준다.

모든 UI 텍스트·주석은 한국어. ecosq.or.kr(에코스퀘어/KEITI) 톤의 블루 테마.

## 명령어

```bash
uv sync                              # 의존성 설치 (.venv 생성)
uv run python scripts/build_index.py # 오프라인: reports_upload/ → index/ 아티팩트(임베딩, 증분)
uv run streamlit run app.py          # 앱 실행 — index/를 읽기 전용 로드(진입점)
uv run pytest                        # 전체 테스트 (API 키 불필요, mock 사용)
uv run pytest tests/test_loaders.py  # 단일 파일
uv run python eval/run_eval.py --mode rerank --k 5  # 검색 품질 평가(Recall@k/MRR) — API 키 + index/ 필요
uv run python eval/run_eval.py --mode answer        # 그라운딩 평가(실제 LLM 응답·기권)
```

- `OPENAI_API_KEY`는 프로젝트 루트의 `.env`에 둔다 (`load_dotenv()`로 읽음, git ignore됨).
- 키가 없으면 앱은 뜨지만 학습·검색·요약이 모두 비활성화되고 안내 메시지를 띄운다.
  단위 테스트는 `tests/conftest.py`의 `FakeClient`로 키 없이 결정론적으로 돈다.
- 진입점은 `app.py` 하나다.

## 아키텍처

`app.py`는 **얇은 Streamlit UI 껍데기**(세션·렌더·토스트)이고, 검색·임베딩·인제스션 로직은
`rag/` 패키지에 레이어로 분리되어 있다. 이 분리의 목적은 **각 단계를 UI 없이 단독으로
테스트·평가**할 수 있게 하는 것이다(`rag/`는 Streamlit에 의존하지 않는다).

```
config/rules.yaml      # 모델명·임계치·청킹·RRF·rerank_pool 등 설정의 단일 소스(하드코딩 금지)
scripts/build_index.py # 오프라인 인덱서: reports_upload/ → index/ (증분, 임베딩+세부정보 메타데이터)
index/                 # 불변 아티팩트(gitignore): docs.jsonl(본문+metadata)·chunks.jsonl·embeddings.npz
rag/
  config.py            # rules.yaml 로더(get_settings) + 경로 상수(INDEX_DIR, UPLOAD_DIR)
  clients.py           # get_client() — OpenAI 클라이언트 단일 생성 지점(로컬 교체 시 여기만)
  ingest/loaders.py    # 파일→레코드 파싱(.hwp/.txt/.md/.csv/.json), HWP 추출, clean_db_name
  ingest/chunk.py      # chunk_body(문단 토큰 윈도우) + build_chunk_input(이름 prefix)
  embed.py             # embed_text(질의), embed_texts(배치), get_encoder
  store.py             # save_index/load_index (npz+jsonl 아티팩트). load는 reports 구조 복원
  retrieve.py          # cosine_similarity, BM25(tokenize/bm25_scores), hybrid_rank(RRF), best_chunk(출처)
  rerank.py            # rerank() — hybrid 후보를 LLM 관련도로 재정렬(폴백: 원래 순서)
  generate.py          # recommend(추천), summarize_detail(세부정보, 빌드시 metadata로 precompute)
  usage.py             # UsageTracker — 단계별 OpenAI 토큰·USD 비용 집계(PRICES 단가표)
  pipeline.py          # search() — hybrid_rank → rerank → recommend + trace(출처·토큰·타이밍·로깅)
eval/                  # golden.jsonl(93문항, difficulty/match) + run_eval.py (dense|hybrid|rerank|answer)
tests/                 # 단계별 단위 테스트 + 실제 HWP 37개 회귀 테스트
```

**데이터 흐름 (build/serve 분리):**
1. **빌드(오프라인)**: `reports_upload/`에 보고서를 넣고 `scripts/build_index.py` 실행 → 파싱·청킹·임베딩
   + `summarize_detail`로 세부정보 metadata를 1회 계산해 `index/` 생성(`body_hash` 증분으로 변경분만).
2. **서빙(읽기 전용)**: 앱 시작 시 `store.load_index()`가 아티팩트를 기존과 동일한 `reports`
   구조(`{db_name, body, metadata, chunks}`)로 복원. **서빙 중 쓰기 없음 → 동시성 안전.** 버튼은 디스크 재로드만.
3. 검색 → `pipeline.search()`가 검색어 임베딩 → `retrieve.hybrid_rank()`로 Dense(청크-max 코사인)
   +BM25(이름 색인)를 RRF 융합해 넓게(`rerank_pool`) 추림 → `rerank.rerank()`가 LLM으로 관련도 재정렬해
   `top_k`로 좁힘 → `generate.recommend()`가 후보를 LLM에 넘겨 추천 JSON(no_match 포함) 생성.
   (검색당 OpenAI 호출 3회: 질의 임베딩 → rerank → recommend) `search()`는 결과에
   `evidence`(추천 근거 청크=출처)와 `trace`(단계별 타이밍·토큰·USD 비용)를 함께 담고, `rag.pipeline` 로거로 콘솔에도 남긴다.
4. 추천 DB 버튼 → 모달이 **빌드 시점에 precompute된 `metadata`(5항목)를 읽어 즉시 표시**(질의 시점 LLM 호출 없음).
   추천 카드엔 근거 본문 청크(출처), `🔍 검색 과정 로깅` 패널엔 단계별 타이밍·토큰·비용이 노출된다.

**핵심 설계 결정 (변경 시 주의):**
- **설정은 `config/rules.yaml`이 단일 소스.** 모델명·임계치를 코드에 하드코딩하지 말 것.
- **build/serve 분리, 읽기 전용 서빙:** 저장은 `index/` 불변 아티팩트(npz+jsonl). 빌드는 오프라인
  스크립트에서만, 서빙은 읽기 전용 로드 → 동시성 안전. 배포는 `index/`(+코드)로 끝(DB 서버 불필요).
  `index/`는 GitHub 배포를 위해 **커밋**한다(데이터 소유자 공개 결정). `.env`(키)·`reports_upload/`(원본)·eval 캐시는 git 제외.
- **출처(provenance):** `retrieve.best_chunk()`가 추천 DB에서 질의와 가장 일치하는 본문 청크를 찾아
  `result["evidence"]`로 반환 → 화면에 근거 발췌로 표시. (LLM 사유와 별개의 *결정론적* 출처)
- **토큰/비용·로깅(관측성):** `usage.UsageTracker`가 embed/rerank/recommend의 `response.usage`를 모아
  단계별 토큰·USD 비용 산출(단가는 `usage.PRICES` — 실제 단가로 조정). `pipeline.search`가 `trace`로
  담고 `rag.pipeline` 로거로 콘솔에 남긴다. recommend가 본문 전체를 받아 토큰 대부분을 차지(컨텍스트 다이어트 여지).
- **store.load_index는 동작 보존:** 아티팩트를 기존과 동일한 `reports=[{db_name, body, chunks:[{text,
  embedding}]}]` 구조로 복원(임베딩은 npz 행→list). 그래서 retrieve/rerank/generate가 무변경 → eval 지표 불변.
- **증분 빌드:** `body_hash`로 변경 없는 문서의 임베딩을 재사용(재임베딩 비용 회피).
- **검색 단위는 청크:** `chunk_body()`가 본문을 문단 토큰 윈도우로 나눠 청크별 임베딩,
  검색은 문서 청크 중 **최고 코사인(max)**. (문서 평균 벡터 폐기 — Stage 1)
- **그라운딩은 LLM이 담당:** off-domain 질의(데이터에 없는 주제)는 `recommend`의 `no_match`로 기권한다
  (실측 기권 정확도 1.000). 코사인 `similarity_floor`(0.40)는 '다른 유사 후보' 표시 필터용일 뿐, 기권 판정이 아니다.
- **품질은 측정으로 증명:** 변경마다 `eval/run_eval.py`로 before/after를 비교한다(dense 0.890→hybrid 0.906→
  rerank 0.987). 인프라성 변경(리팩터·아티팩트)은 "지표 불변"을 성공 기준으로 삼는다(동작 보존).
- **유사도 바닥값(상대 임계)**: '다른 유사 후보(others)' 노출 임계 = `min(similarity_floor 0.40,
  최고점수 × others_ratio 0.85)`(`pipeline.visible_others`). 추천(최고점수)은 floor와 무관하게 항상
  표시되므로, 짧은/제너럴 질의(코사인이 전반적으로 낮음)에서도 추천에 견줘 비슷한 후보는 노출한다
  (임계는 floor 이하라 기존 동작 비회귀). floor를 0으로 두지 말 것.
- **rag 패키지는 UI 의존 금지:** `load_index`는 인덱스가 없으면 `[]`를 반환(graceful),
  캐싱(`st.cache_data`)은 UI 계층에서만 감싼다(rag는 `lru_cache` 사용).

**.hwp 파서 (`ingest/loaders.py`의 `extract_hwp_text` 및 `_hwp_*`):**
HWP 5.0은 OLE 복합 파일이라 `olefile`로 `BodyText/Section*` 스트림을 읽는다.
FileHeader 압축 플래그가 켜져 있으면 zlib raw-deflate(`-15`)로 해제하고,
`HWPTAG_PARA_TEXT`(67) 레코드에서 제어문자(1워드/8워드)를 걸러 UTF-16LE 텍스트만 추출한다.
**표 셀도 문단 텍스트라 함께 추출된다** — 영향평가 수치(예: `Climate change_Total = 7.96E-02 kg CO2 eq`)가
본문에 이미 들어온다(표 구조만 평탄화). 그림은 제외. 신형 `.hwpx`(압축 XML)는 미지원(이 코퍼스엔 .hwpx 없음) —
새 형식 추가는 `parse_report_file()`에 분기를 더한다.
**추출 후 `clean_body_text`가 폼 노이즈를 정리한다**: 미선택 체크박스(`□○`)는 '고르지 않은 옵션'이라
제거(오매칭 방지), 선택 체크박스(`■▣●`)는 마커만 떼고 값 보존(예 `▣ 1차`→`1차`), `(수식)`·빈 괄호·단독 기호 줄 제거.
표 수치·짧은 토큰(디젤·1차)은 보존. 노이즈 12% 감소, **rerank MRR 0.962→0.968·hard 0.900→0.950**(그라운딩 1.000/0.974 유지).
※ 정제로 본문이 바뀌면 `body_hash`가 변해 재임베딩되므로, `clean_body_text` 수정 후엔 `build_index.py` 재실행 + eval 재측정 필수.

## 모델 설정 (config/rules.yaml)

- `model: gpt-5.4-nano` — 추천 문구·세부정보 요약 생성용 (OpenAI chat completions, `response_format=json_object`)
- `embedding_model: text-embedding-3-small` — 유사도 검색용
- 토크나이저는 `embed_encoding`(cl100k_base, tiktoken), `lru_cache`로 1회 로드

## 컨벤션

- 한국어 주석 + 한국어 UI, 영문 변수/함수명.
- LLM 호출 실패는 예외를 위로 던지지 않고 `None`/에러 dict로 흡수해 화면에 안내한다(이 패턴 유지).
- 새 보고서 형식 이름/내용 키는 `loaders.py`의 `_NAME_KEYS`/`_BODY_KEYS`에 추가(대소문자·공백 무시 매칭).
- **cSpell:** 새 기술 용어·라이브러리 이름은 루트 `cspell.json`의 `words`에 추가해 경고를 만들지 않는다.
- 새 기능은 `tests/`에 단위 테스트를 함께 추가(LLM 경로는 `FakeClient` mock). 검색 품질 변경은
  `eval/run_eval.py`로 Recall@k/MRR 회귀를 확인한다.
