"""불변 인덱스 아티팩트 저장/로드 (npz + jsonl).

오프라인 빌드가 만든 아티팩트를 서빙이 **읽기 전용**으로 로드한다(서빙 중 쓰기 없음 → 동시성 안전).
로드 결과는 기존과 **동일한** reports 구조라, 검색·추천 레이어가 한 줄도 안 바뀐다.

  reports = [{"db_name": str, "body": str, "chunks": [{"text": str, "embedding": list[float]}]}]

아티팩트(`index/`):
  docs.jsonl    : {doc_id, db_name, body, body_hash}        (문서당 1줄)
  chunks.jsonl  : {doc_id, chunk_id, text}                  (청크당 1줄, 행 순서 = 임베딩 행 순서)
  embeddings.npz: float32 배열 (n_chunks, dim)              (chunks.jsonl 행과 1:1 정렬)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from rag.config import INDEX_DIR

DOCS_FILE = "docs.jsonl"
CHUNKS_FILE = "chunks.jsonl"
EMB_FILE = "embeddings.npz"


def body_hash(db_name, body):
    """문서 동일성 판정용 해시(증분 빌드에서 변경 없는 문서 재사용에 사용)."""
    return hashlib.sha256(f"{db_name}\n{body}".encode("utf-8")).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def save_index(reports, index_dir=INDEX_DIR):
    """reports를 인덱스 아티팩트로 기록합니다."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_rows, chunk_rows, embeddings = [], [], []
    for doc_id, r in enumerate(reports):
        doc_rows.append({
            "doc_id": doc_id, "db_name": r["db_name"], "body": r["body"],
            "body_hash": body_hash(r["db_name"], r["body"]),
            "metadata": r.get("metadata"),  # 세부정보 5항목(빌드 시점 precompute) 또는 None
        })
        for chunk_id, c in enumerate(r["chunks"]):
            chunk_rows.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": c["text"],
                               "section": c.get("section", ""), "kind": c.get("kind", "body")})
            embeddings.append(c["embedding"])

    _write_jsonl(index_dir / DOCS_FILE, doc_rows)
    _write_jsonl(index_dir / CHUNKS_FILE, chunk_rows)
    arr = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(index_dir / EMB_FILE, embeddings=arr)


def load_index(index_dir=INDEX_DIR):
    """아티팩트를 읽어 reports 구조를 복원합니다(없으면 빈 목록).

    임베딩은 npz 행을 파이썬 list로 materialize하여 기존 동작(cosine_similarity)과 동일합니다.
    """
    index_dir = Path(index_dir)
    if not (index_dir / DOCS_FILE).exists():
        return []

    docs, order = {}, []
    for d in _read_jsonl(index_dir / DOCS_FILE):
        docs[d["doc_id"]] = {"db_name": d["db_name"], "body": d["body"],
                             "metadata": d.get("metadata"), "chunks": []}
        order.append(d["doc_id"])

    emb = np.load(index_dir / EMB_FILE)["embeddings"]
    for row, c in enumerate(_read_jsonl(index_dir / CHUNKS_FILE)):
        docs[c["doc_id"]]["chunks"].append({
            "text": c["text"], "embedding": emb[row].tolist(),
            "section": c.get("section", ""), "kind": c.get("kind", "body"),
        })
    return [docs[i] for i in order]
