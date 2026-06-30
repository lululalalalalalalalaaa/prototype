"""rag 설정 로더 — config/rules.yaml을 단일 소스로 읽어옵니다.

모델명·임계치는 코드에 하드코딩하지 않고 rules.yaml에서만 가져옵니다.
경로 상수(저장 파일·업로드 폴더)는 값이 아니라 위치이므로 코드가 소유합니다.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# 프로젝트 루트: rag/config.py 기준 한 단계 위
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_FILE = PROJECT_ROOT / "config" / "rules.yaml"

# 불변 인덱스 아티팩트 폴더(오프라인 빌드 산출물, 서빙은 읽기 전용; .gitignore 처리됨)
#   index/docs.jsonl · index/chunks.jsonl · index/embeddings.npz
INDEX_DIR = PROJECT_ROOT / "index"
# 관리자가 보고서 파일을 넣어두는 업로드 폴더(빌드 입력)
UPLOAD_DIR = PROJECT_ROOT / "reports_upload"
SUPPORTED_EXTS = {".hwp", ".txt", ".md", ".csv", ".json"}


@dataclass(frozen=True)
class Settings:
    model: str
    embedding_model: str
    top_k: int
    similarity_floor: float
    embed_encoding: str
    embed_max_tokens: int
    chunk_tokens: int = 400
    chunk_overlap: int = 80
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rerank_pool: int = 15


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """rules.yaml에서 설정을 읽어 Settings로 반환합니다(1회 로드 후 캐시)."""
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rag = data["rag"]
    return Settings(
        model=rag["model"],
        embedding_model=rag["embedding_model"],
        top_k=rag["top_k"],
        similarity_floor=rag["similarity_floor"],
        embed_encoding=rag["embed_encoding"],
        embed_max_tokens=rag["embed_max_tokens"],
        chunk_tokens=rag["chunk_tokens"],
        chunk_overlap=rag["chunk_overlap"],
        rrf_k=rag["rrf_k"],
        bm25_k1=rag["bm25_k1"],
        bm25_b=rag["bm25_b"],
        rerank_pool=rag["rerank_pool"],
    )
