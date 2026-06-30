"""국가 LCI DB 검색 RAG 패키지.

단계별로 단독 검증이 가능하도록 레이어를 분리합니다.
  config   : rules.yaml 설정 로더
  clients  : OpenAI 클라이언트 단일 생성 지점
  ingest   : 파일 로더 → (청킹) → 임베딩 → 저장 파이프라인
  embed    : 임베딩 벡터 생성
  store    : 보고서·임베딩 영속 저장
  retrieve : 코사인 유사도 검색
  generate : LLM 추천·세부정보 요약
  pipeline : retrieve + generate 조립(검색)
"""
