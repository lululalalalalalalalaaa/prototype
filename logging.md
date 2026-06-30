# 작업 로그 (logging.md)

이 문서는 "**사용자가 어떤 프롬프트로 무엇을 요청했고, 그에 따라 어떤 작업·결정·검증을 했는지**"를
시간순으로 기록합니다. 수치는 모두 실측값입니다.

> 핵심 원칙(전 과정 공통): **measure-driven** — 변경마다 골든셋 eval로 before/after를 수치 비교하고,
> 인프라성 변경은 "지표 불변"을 성공 기준으로 삼는 **동작 보존**을 검증했습니다.

| # | 사용자 요청(요지) | 진행한 작업 / 결정 | 검증 |
|---|---|---|---|
| 1 | `/init` — 코드베이스 분석 후 CLAUDE.md 작성 | 단일 `app.py`(685줄) RAG 프로토타입 분석, CLAUDE.md 생성 | — |
| 2 | RAG 개발에서 누락·개선점을 단계적으로 정의 | RAG 10단계 로드맵 작성(평가→청크→하이브리드→리랭커→스토어→인제스션→그라운딩) | — |
| 3 | RAG 개발 시 고려사항은? | "RAG=검색·데이터 엔지니어링, 측정 우선" 관점 정리 | — |
| 4 | 모놀리식 app.py를 모듈화하고 단계별 검증 | **Stage 0 리팩터**: `rag/` 패키지로 레이어 분리(config·clients·ingest·embed·store·retrieve·generate·pipeline), `config/rules.yaml` 외부화, 단위 테스트 73개 | 73 passed, 실제 HWP 37개 회귀 |
| 5 | cspell.json 고려 + 이어서 할 것 | CLAUDE.md doc-drift 수정, cspell 사전 도입(피드백 메모리화) | cspell 클린 |
| 6 | .env 세팅 | `.env`/`.env.example` + `load_dotenv(override=True)`(셸보다 .env 우선) | 키 플러밍 검증 |
| 7 | 등록한 .env로 동작 확인 | 37개 HWP end-to-end 학습·검색 성공. **macOS NFD 파일명 버그 발견·수정**(NFC 정규화) | 74 passed, 검색 정상 |
| 8 | (Stage 0 eval) | 골든셋 14문항 + `run_eval.py` → **베이스라인 MRR 0.929, Recall@5 1.0** | 실측 |
| 9 | Stage 1로 품질 개선 | **청크 단위 검색**: 문서 평균 벡터 폐기 → 문단 토큰 윈도우 청킹 + 청크-max 집계 | **MRR 0.929→1.000**(화물기차 약점 해소) |
| 10 | 골든셋 보강(엄밀화) | 14→46문항(지역 모호성·평균vs구체·우회표현·**no_match 8건**). 천장 제거 | dense MRR 0.890으로 현실화 |
| 11 | Stage 2 (BM25) | 사용자 선택. **순수 파이썬 BM25 + RRF**(한글 bigram, 이름 색인). v1(이름+본문) 회귀 0.871 발견→**v2(이름만) 0.906** | dense 0.890 → **hybrid 0.906** |
| 12 | 골든셋이 엄밀한가? | 라벨 무결성 기계검증(46/46 일치) + **보고서 본문 대조**로 의심 라벨 확정/교정("기름"에서 LPG 제거) | 교정 후 hybrid 0.906 견고 |
| 13 | Stage 3 (리랭커) | **LLM 리랭커**(OpenAI 재사용, 의존성 0). top-15→재정렬→top-5, 폴백 내장. eval rerank 캐시 | hybrid 0.906 → **rerank 0.987**, 전라남도 등 의미실패 해결 |
| 14 | main.py 삭제 + Stage 5·6 | (선결) main.py 제거 | — |
| 15 | Stage 4 (인프라) | **build/serve 분리 + 불변 아티팩트(npz+jsonl)**. 오프라인 빌드(증분)·읽기전용 서빙. numpy 2.5.0 py3.14 확인 | eval **불변**(0.890/0.906/0.987) = 동작 보존 |
| 16 | cspell 자동화 요청 | 원인=`index/` 미무시(데이터). 무시 추가→소스 클린. 점검 루틴을 메모리화(자동 hook은 정책상 명시승인 필요) | cspell 0 이슈 |
| 17 | Stage 5·6 전체 처리 | 조사: **영향평가 수치는 이미 추출됨**(표 셀=문단 텍스트), .hwpx 없음 → 표 파싱 YAGNI. **Stage 5=메타데이터 빌드 precompute**. **Stage 6=진짜 그라운딩 측정** | (아래) |
| 18 | (Stage 5 결과) | 세부정보 5항목을 빌드 시점 1회 계산→`index/` 저장, 모달은 읽기만 | 재빌드(임베딩 재사용/metadata 생성), **rerank 0.987 불변** |
| 19 | (Stage 6 결과) | `--mode answer`로 전체 파이프라인 실측 → **프록시(0.375)는 오측정, 실제 LLM 기권 1.000**. 강화 불필요(측정이 막아줌) | 기권 1.000, 응답 0.974, 과잉기권 0 |
| 20 | 문서화 | logging.md·Nextsession.md·CLAUDE.md·README.md(Mermaid 3종) 작성 | — |
| 21 | Stage 6로 끝? UI에 반영 맞나? | **코드로 UI 검증**: 앱은 `pipeline.search`로 full 파이프라인(hybrid→rerank→recommend) 실행, Stage 4·5·6 모두 반영. 시스템 **PoC-complete** 결론 | grep 검증 |
| 22 | index/가 GitHub에 커밋돼야 하나? | 맞음 — gitignore면 배포 시 빈 인덱스. index/는 보고서 본문 포함 → **private repo에 커밋**(gitignore 해제). public이면 별도 업로드 필요 | — |
| 23 | 골든셋 대규모화 | 46→**93문항**(난이도·OR/AND 태그, 근거화, near-domain no_match). `run_eval` 난이도별 분해 + any/all 채점. 헤드룸 확보 | dense 0.821 / hybrid 0.825 / **rerank 0.962**(hard 0.925). **BM25 이득 미미 발견** |
| 24 | logging.md·Nextsession.md 갱신 | 문서 작성 | — |
| 25 | 의도했으나 구현 안 된 것? | eval 채점 헬퍼(`recall_at_k` any/all·`reciprocal_rank`) 추출 + 단위 테스트(계획에 적고 누락했던 항목). answer 난이도별 집계 | 114 passed, dense 0.821 불변(behavior-preserving) |
| 26 | #1 그라운딩 강화 + #2 BM25 재검토 모두 | **BM25 ablation**(`--pool dense`): rerank가 dense풀 vs hybrid풀 **둘 다 0.962**(net-neutral, hard +0.025/medium −0.014) → 유지. **그라운딩 강화**: recommend 프롬프트에 near-domain 거부 규칙 | near-domain 기권 0.867 → **1.000**, 응답 0.974·과잉기권 0.013(hard 1건) |
| 27 | 서버 켰을 때 각 과정 로깅 확인되나? | 당시 로깅 0% → **관측성 구현**: `pipeline.search`에 단계별 `trace`(타이밍) + `rag.pipeline` 콘솔 로깅 + 인앱 `🔍` 패널(리랭크 전/후 포함) | 라이브 검증, 앱 0 traceback |
| 28 | konetic-report-rag 참고 | 형제 프로젝트(같은 스택). 그들의 출처·토큰을 도입; 우리는 **eval·HWP로 앞섬**, Chroma/pkl은 과설계라 미도입 | — |
| 29 | 토큰·출처 엄밀히 됐나? | **출처(provenance)**: `best_chunk` 근거 청크 인용. **토큰/비용**: `UsageTracker`가 단계별 토큰·USD 집계. 화면·trace·로그 노출 | 121 passed, 라이브 토큰 17.5K·$0.001·출처 캡처 |
| 30 | placeholder가 no_match로 뜸 | 입력 예시가 철강(데이터에 없음)이라 오해 유발 → 검색되는 예시로 교정 | — |
| 31 | 커밋 푸시 | 데이터 소유자 **공개 결정**(본문 "공개 불가" 표기는 3회 고지) → `main` 직접 커밋·푸시, `index/` 포함, `.env` 제외 | 121 passed, `cb61926` 푸시 |
| 32 | 문서 동기화 점검 | README·CLAUDE·logging·Nextsession의 drift 수정(골든셋 93·출처·토큰·로깅·numpy·index 공개) | — |
| 33 | rerank+recommend 호출 병합 시도 | 단일 LLM 호출(재정렬+추천)+컨텍스트 다이어트(본문→청크) 구현·측정. **토큰 −74%·지연 −39%·호출 3→2**였으나 **품질 전면 회귀** → **되돌림(revert)**. 발견: rerank는 넓고 싸게(이름·15), recommend는 좁고 깊게(본문·5)로 컨텍스트 요구가 달라 단일 호출론 품질/비용 frontier를 못 넘음(다이어트=품질붕괴, 풀본문=비용↑) | rerank MRR 0.962→0.898(hard 0.925→0.752), 응답 0.974→0.795, 기권 1.000→0.800 → 비회귀 실패, revert |
| 34 | 제너럴 질의 결과 / 출처 / 배지 | ① others 노출을 **상대 임계**(`min(floor, 최고×0.85)`)로 → 짧은 질의도 유사 후보 노출(`visible_others`). ② 출처에 **위치(청크 N/M)** 표시(`best_chunk`). ③ watchdog 추가(Streamlit 핫리로드). ④ README **Tech Stack 배지** + 엄밀 동기화 | 132 passed, 라이브 '디젤'→화물디젤 노출 |
| 35 | "기반(추출·청킹)이 빵꾸 많다" 지적 | **HWP 추출 정제**(`clean_body_text`): 전 37문서 노이즈 정량조사(빈 체크박스 8.6%·채움 6.7%·수식 1%) → 미선택 `□○` 제거·선택 `■▣●` 마커만 제거·`(수식)`/빈괄호 제거. 표·수치·짧은토큰 보존. 재빌드(37 재임베딩) 후 측정 | 노이즈 −12%, **rerank 0.962→0.968·hard 0.900→0.950**, 그라운딩 1.000/0.974 유지(dense 0.821→0.804은 리랭커가 흡수) → **keep** |

## 누적 결과 요약

- **검색 품질**:
  - 옛 골든셋(46, k=5): Dense 0.890 → Hybrid 0.906 → Rerank MRR 0.987 (천장).
  - **확장 골든셋(93, k=5, 더 엄밀)**: Dense 0.821 → Hybrid 0.825 → **Rerank 0.962**.
    난이도별 rerank: easy 1.000 / medium 0.958 / **hard 0.925**(헤드룸 존재).
  - **발견**: 현실적 질의에선 **BM25(하이브리드) 이득이 미미**(0.821→0.825) — 리랭커가 대부분 흡수.
- **그라운딩**(전체 파이프라인, 93세트, recommend 프롬프트 강화 후): off-domain 기권 **1.000**(15/15),
  응답 정확도 **0.974**(hard 0.950), 과잉기권 0.013. (강화 전 near-domain 0.867 → 강화 후 1.000)
- **관측성·출처·비용**: 검색당 단계별 trace(타이밍·토큰·USD), 추천 근거 청크(출처) 인용, 서버 콘솔 + 인앱 패널.
- **인덱스**: 37 문서 / 745 청크 / 메타데이터 37. 빌드 증분. **public repo에 커밋·푸시(`cb61926`)**.
- **테스트**: 121 passed (전부 API 키 불필요, mock). cspell 0 이슈.
- **부수 정리**: NFD 파일명 버그 수정, `.env` 우선순위, cspell 사전, main.py/lci_reports.json/ingest.pipeline 제거.

## 반복적으로 작동한 패턴

- **측정 후 결정**: BM25 v1 회귀(0.871)·Stage 6 프록시 오측정(0.375)·**호출 병합 회귀(MRR 0.898·기권 0.800)**를 eval이 잡아내, 잘못된 변경을 막음(부정 결과도 자산).
- **동작 보존 마이그레이션**: 리팩터·아티팩트 전환을 "eval 지표 불변"으로 안전 검증.
- **YAGNI/over-engineering 차단**: 표 파싱·ANN·프롬프트 강화를 *실측 근거로* 제외.
