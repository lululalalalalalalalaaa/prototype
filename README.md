# 국가 LCI DB 검색 (RAG)

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-2.44.0-412991?logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5.0-013243?logo=numpy&logoColor=white)
![tiktoken](https://img.shields.io/badge/tiktoken-0.13.0-000000)
![Pillow](https://img.shields.io/badge/Pillow-12.2.0-0A0A0A?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![tests](https://img.shields.io/badge/tests-139_passed-0A9EDC?logo=pytest&logoColor=white)

![LLM](https://img.shields.io/badge/LLM-gpt--5.4--nano-10A37F?logo=openai&logoColor=white)
![Vision](https://img.shields.io/badge/vision-gpt--5.4-10A37F?logo=openai&logoColor=white)
![Embedding](https://img.shields.io/badge/embedding-text--embedding--3--small-10A37F?logo=openai&logoColor=white)
![Retrieval](https://img.shields.io/badge/retrieval-Dense%2BBM25%20RRF%20%E2%86%92%20LLM%20rerank-0E5AA7)
![MRR](https://img.shields.io/badge/rerank%20MRR-0.972-success)

자연어로 질문하면 가장 적합한 **국가 LCI(전과정 목록분석) DB**를 추천하는 검색 시스템입니다.
사용자가 "전기차로 사람을 수송할 때 온실가스 배출 데이터"처럼 입력하면, 임베딩·BM25·LLM
리랭커를 거쳐 가장 알맞은 LCI DB를 근거와 함께 제시하고, 보고서 세부정보(기능단위·시스템경계·
영향평가 결과 등)를 요약해 보여줍니다.

> **PoC**입니다. 별도 벡터DB 서버 없이, 오프라인으로 만든 인덱스 아티팩트(`index/`)를
> Streamlit 앱이 읽기 전용으로 서빙합니다. 배포는 `index/` + 코드를 업로드하면 끝입니다.

---

## 핵심 특징

- **검색 품질을 수치로 검증** — 골든셋(105문항, **내용 질의 포함**) + `eval/run_eval.py`로 각 단계의 기여를 측정(아래 표).
- **구조 인식 인제스션** — HWP 폼 노이즈 정제 + 섹션(`N. 제목`)·표(`표.`) 분리 청킹 + **공정흐름도 이미지를 vision으로 텍스트화**.
- **하이브리드 + 리랭커** — Dense(임베딩) + BM25(정확 토큰)를 RRF로 융합하고, LLM이 관련도로 재정렬.
- **출처(provenance) + 토큰/비용 계측** — 추천 근거 청크를 **위치(📑 섹션명·표)와 함께** 인용 표시하고, 검색당 단계별 토큰·USD 비용을 집계.
- **단계별 로깅(관측성)** — 검색마다 임베딩→하이브리드→리랭커→추천의 입출력·타이밍을 서버 콘솔 + 인앱 패널에 노출.
- **build/serve 분리** — 임베딩·vision 비용은 오프라인 빌드 1회. 서빙은 읽기 전용 → 동시성 안전.
- **그라운딩** — 데이터에 없는 주제(철강·항공 등)는 LLM이 "적합 DB 없음"으로 올바르게 기권.

## 검색 품질 (골든셋 105문항: 90 정답 + 15 no_match, k=5)

| 단계 | Recall@5 | MRR | 비고 |
|---|---|---|---|
| Dense (임베딩만) | 0.956 | 0.811 | 청크-max 코사인 |
| + BM25 하이브리드(RRF) | 0.933 | 0.794 | 정확 토큰 보강(현실 질의엔 효과 작음) |
| **+ LLM 리랭커** | **0.989** | **0.972** | 의미·동의어(전라남도=전남) 해결 |

난이도별 rerank MRR: easy 0.980 / medium 0.978 / **hard 0.950**.
**그라운딩**(전체 파이프라인 실측): off-domain 15문항 기권 정확도 **1.000**, 답 있는 90문항 응답 정확도 **0.989**(과잉기권 0.011).
**내용 질의 12문항**(공정·연료·기능단위 — 이름 아닌 본문): 추천 정확도 **12/12**.

---

## 아키텍처 — build/serve 분리

```mermaid
flowchart LR
  subgraph Build["오프라인 빌드 · scripts/build_index.py (OpenAI 비용 1회)"]
    A["reports_upload/*.hwp"] --> B["loaders<br/>파싱 + clean_body_text"]
    B --> C["structure_chunks<br/>섹션·표 분리"]
    A --> I["images<br/>공정흐름도 → vision 설명"]
    C --> D["임베딩<br/>text-embedding-3-small"]
    I --> D
    B --> E["summarize_detail<br/>세부정보 메타데이터"]
    D --> F[("index/ 아티팩트<br/>docs.jsonl · chunks.jsonl · embeddings.npz")]
    E --> F
  end
  subgraph Serve["읽기 전용 서빙 · app.py (Streamlit)"]
    F --> G["load_index<br/>(reports 복원)"]
    G --> H["검색 파이프라인"]
  end
```

빌드는 `body_hash`로 **증분**(변경 없는 문서는 재임베딩하지 않음). 서빙은 인덱스를 메모리에 읽어
질의에 응답할 뿐 쓰지 않으므로, 다중 사용자에도 파일 경합이 없습니다.

## 검색 파이프라인

```mermaid
flowchart TD
  Q["검색어"] --> E["① 질의 임베딩"]
  E --> H["② hybrid_rank<br/>Dense 코사인 + BM25 → RRF 융합"]
  H --> P["top-15 후보 pool"]
  P --> R["③ LLM 리랭커<br/>관련도 재정렬 → top-5"]
  R --> G["④ LLM recommend<br/>추천 1개 + 다른 후보 + no_match"]
  G --> O["추천 결과 + 출처 인용 + 토큰/비용/타이밍 trace"]
```

검색 1회당 OpenAI 호출 3회(임베딩 → 리랭커 → 추천). 각 검색은 추천 근거 청크(출처)·단계별
토큰/비용·타이밍을 `trace`에 담아 인앱 `🔍 검색 과정 로깅` 패널과 서버 콘솔에 표시합니다.
세부정보는 빌드 시점에 미리 계산되어 모달은 즉시 표시됩니다.

## 단계별 품질 사다리

```mermaid
flowchart LR
  D["Dense<br/>MRR 0.811"] --> Hy["+ BM25 하이브리드<br/>MRR 0.794"]
  Hy --> Re["+ LLM 리랭커<br/>MRR 0.972"]
```

각 단계는 같은 골든셋으로 before/after를 측정해 기여를 증명했습니다(`eval/run_eval.py --mode dense|hybrid|rerank`).

---

## 빠른 시작

```bash
# 1) 의존성 설치
uv sync

# 2) OpenAI 키 설정 — .env.example을 복사해 키 입력
cp .env.example .env        # 그리고 OPENAI_API_KEY= 뒤에 키 붙여넣기

# 3) 보고서를 reports_upload/에 넣고, 인덱스 빌드(오프라인, 임베딩 비용)
uv run python scripts/build_index.py

# 4) 앱 실행 (index/를 읽기 전용 로드)
uv run streamlit run app.py
```

지원 보고서 형식: `.hwp`(한글 5.0) · `.txt` · `.md` · `.csv` · `.json` (자세한 규칙은 `reports_upload/README.md`).

## 평가

```bash
uv run python eval/run_eval.py --mode dense  --k 5   # 임베딩만
uv run python eval/run_eval.py --mode hybrid --k 5   # + BM25
uv run python eval/run_eval.py --mode rerank --k 5   # + LLM 리랭커 (검색 품질)
uv run python eval/run_eval.py --mode answer         # 전체 파이프라인 그라운딩(응답/기권)
```

`eval/golden.jsonl`이 정답지(질의 → 정답 DB 이름). 질의 임베딩·리랭크·응답 결과는 캐시되어
재실행은 빠릅니다. (키 + `index/` 필요)

## 테스트

```bash
uv run pytest          # 전체 (API 키 불필요 — mock 클라이언트)
```

각 단계(로더·청킹·임베딩·검색·BM25·리랭커·저장·빌드)가 단독 단위 테스트로 검증됩니다.
실제 `.hwp` 파일에 대한 회귀 테스트도 포함됩니다.

## 프로젝트 구조

```
config/rules.yaml        설정 단일 소스 (모델·임계치·청킹·RRF·rerank_pool)
scripts/build_index.py   오프라인 인덱서 → index/ (증분)
index/                   불변 아티팩트: docs.jsonl·chunks.jsonl·embeddings.npz (배포 위해 커밋)
rag/
  config·clients         설정 로더 / OpenAI 클라이언트 단일 생성 지점
  ingest/loaders         파일 파싱(HWP 포함) + clean_body_text(폼 노이즈 정제)
  ingest/chunk           structure_chunks(섹션·표 분리) + build_chunk_input(이름+섹션)
  ingest/images          HWP 이미지 추출 + 평탄화 + vision 설명(공정흐름도, 로고 dedup)
  embed·store            임베딩 / 인덱스 아티팩트 저장·로드(section·kind 포함)
  retrieve               코사인 + BM25(순수 파이썬) + hybrid_rank(RRF) + best_chunk(출처·섹션·위치)
  rerank·generate        LLM 리랭커 / 추천·세부정보
  usage                  토큰/비용 계측(UsageTracker, 모델별 단가)
  pipeline               search() = hybrid → rerank → recommend + trace + visible_others(상대 임계)
eval/                    golden.jsonl(105문항) + run_eval.py (dense|hybrid|rerank|answer)
tests/                   단계별 단위 테스트
app.py                   얇은 Streamlit UI (읽기 전용) + 출처/토큰/로깅 표시
```

## 기술 메모

- **모델**: 추천·요약 `gpt-5.4-nano`, **이미지(공정흐름도) `gpt-5.4`**(nano는 한국어 다이어그램을 못 읽음), 임베딩 `text-embedding-3-small` (전부 `config/rules.yaml`).
- **구조 인식 인제스션** — `clean_body_text`(폼 체크박스·수식 노이즈 제거) → `structure_chunks`(섹션·표 분리) → `images`(BinData 이미지를 흰배경 평탄화 후 vision 설명). 노이즈 −12%로 rerank 0.962→0.972.
- **이미지 평탄화 필수** — 원본은 팔레트+투명 PNG라 그대로 보내면 vision이 '검은 이미지'로 받아 헛읽음. Pillow로 흰 배경 합성.
- **순수 파이썬 BM25/코사인** — 검색 연산은 numpy 없이 손구현(규모가 커지면 numpy 벡터화 고려). numpy는 npz 저장에만 사용.
- **한글 BM25 토크나이저** — 음절 bigram(예: '경남'↔'경남권')+영숫자 토큰(MDF·LPG). 형태소 분석기 불필요.
- **macOS HWP 파일명**은 NFD라 NFC 정규화 후 처리(`clean_db_name`).
- **'다른 유사 후보' 노출(상대 임계)** — 임계 `min(similarity_floor 0.40, 최고점수×others_ratio 0.85)`
  (`pipeline.visible_others`). 짧은/제너럴 질의는 코사인이 전반적으로 낮아도 추천에 견줘 비슷한 후보를 노출(`config/rules.yaml`).
- **Streamlit 핫리로드** — `watchdog`로 코드 편집 시 `rag/` 하위 모듈까지 안정적으로 재로딩.
- **데이터 공개** — `index/`(보고서 본문 포함)는 배포를 위해 git에 커밋됩니다(데이터 소유자 공개 결정).
  `.env`(API 키)·`reports_upload/`(원본)·eval 캐시는 git에서 제외됩니다.
