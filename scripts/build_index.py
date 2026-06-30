"""오프라인 인덱스 빌더 — reports_upload/ → index/ 아티팩트.

서빙(app.py)은 이 산출물을 읽기 전용으로 로드만 한다. 인제스트(임베딩 비용)는 여기서 1회만.

실행:
  uv run python scripts/build_index.py

요구사항: OPENAI_API_KEY(.env).
증분: 이전 index/가 있으면 body_hash가 같은 문서의 임베딩을 재사용(변경분만 재임베딩).
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.clients import get_client  # noqa: E402
from rag.config import INDEX_DIR, SUPPORTED_EXTS, UPLOAD_DIR  # noqa: E402
from rag.embed import embed_texts  # noqa: E402
from rag.generate import summarize_detail  # noqa: E402
from rag.ingest.chunk import build_chunk_input, chunk_body  # noqa: E402
from rag.ingest.loaders import parse_report_file  # noqa: E402
from rag.store import body_hash, load_index, save_index  # noqa: E402


def build(upload_dir=UPLOAD_DIR, index_dir=INDEX_DIR, client=None):
    client = client or get_client()
    if client is None:
        sys.exit("OPENAI_API_KEY가 없습니다. .env를 확인하세요.")

    # 증분: 이전 아티팩트의 문서를 body_hash로 색인해 재사용
    prior = {body_hash(r["db_name"], r["body"]): r for r in load_index(index_dir)}

    files = sorted(
        p for p in Path(upload_dir).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.stem.lower() != "readme"
    )

    reports, errors = [], []
    seen, reused, embedded = set(), 0, 0
    for p in files:
        try:
            records = parse_report_file(p)
        except Exception as e:
            errors.append(f"{p.name}: {e}")
            continue
        for rec in records:
            h = body_hash(rec["db_name"], rec["body"])
            if h in seen:
                continue
            seen.add(h)
            if h in prior:
                prev = prior[h]               # 변경 없음 → 임베딩 재사용
                if prev.get("metadata") is None:   # 구 인덱스 마이그레이션: metadata만 보충
                    prev["metadata"] = summarize_detail(client, prev["db_name"], prev["body"])
                reports.append(prev)
                reused += 1
                continue
            try:
                texts = chunk_body(rec["body"])
                inputs = [build_chunk_input(rec["db_name"], t) for t in texts]
                embeddings = embed_texts(client, inputs)
                rec["chunks"] = [{"text": t, "embedding": e}
                                 for t, e in zip(texts, embeddings)]
                if not rec["chunks"]:
                    raise ValueError("청크가 생성되지 않았습니다(본문 비어있음).")
                # 세부정보 5항목을 빌드 시점에 1회 precompute(질의 시점 LLM 호출 제거)
                rec["metadata"] = summarize_detail(client, rec["db_name"], rec["body"])
                reports.append(rec)
                embedded += 1
            except Exception as e:
                errors.append(f"{rec['db_name']}: 임베딩 실패 ({e})")

    save_index(reports, index_dir)
    print(f"인덱스 빌드 완료 → {index_dir}")
    print(f"  문서 {len(reports)} (재사용 {reused} / 신규 임베딩 {embedded}) | 실패 {len(errors)}")
    for e in errors:
        print("  -", e)


if __name__ == "__main__":
    load_dotenv(override=True)
    build()
