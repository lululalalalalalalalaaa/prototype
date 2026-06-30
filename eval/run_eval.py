"""retrieval 레이어 단독 평가 — Recall@k / MRR (LLM 호출 없음).

리팩터 전후, 그리고 향후 모든 검색 품질 개선(청크 검색·하이브리드·리랭커)의
효과를 같은 골든셋으로 수치 비교하기 위한 기준 도구입니다.

실행:
  uv run python eval/run_eval.py            # 기본 k=10
  uv run python eval/run_eval.py --k 5

요구사항:
  - OPENAI_API_KEY (.env)        : 질의 임베딩에 필요
  - lci_reports.json             : 학습된 보고서·임베딩 (앱에서 '보고서 읽기' 후 생성)
질의 임베딩은 eval/.query_cache.json에 캐시해 반복 실행 비용을 줄입니다.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.clients import get_client  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.embed import embed_text  # noqa: E402
from rag.pipeline import search as pipeline_search  # noqa: E402
from rag.rerank import rerank  # noqa: E402
from rag.retrieve import hybrid_rank, rank  # noqa: E402
from rag.store import load_index  # noqa: E402

GOLDEN_FILE = Path(__file__).parent / "golden.jsonl"
QUERY_CACHE = Path(__file__).parent / ".query_cache.json"
RERANK_CACHE = Path(__file__).parent / ".rerank_cache.json"
ANSWER_CACHE = Path(__file__).parent / ".answer_cache.json"


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_golden():
    rows = []
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cache():
    if QUERY_CACHE.exists():
        return json.loads(QUERY_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache):
    QUERY_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def embed_query(client, query, cache):
    """질의 임베딩(캐시 우선). 캐시에 없으면 API 호출 후 저장."""
    if query in cache:
        return cache[query]
    vec = embed_text(client, query)
    cache[query] = vec
    return vec


def reciprocal_rank(retrieved, expected):
    """첫 정답의 역순위(없으면 0). retrieved 순서대로 expected 포함 여부 확인."""
    expected = set(expected)
    for i, name in enumerate(retrieved, start=1):
        if name in expected:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved, expected, match):
    """match=any → 하나라도 찾으면 1.0(OR). match=all → 교집합/expected(AND)."""
    expected = set(expected)
    if not expected:
        return 0.0
    inter = set(retrieved) & expected
    if match == "any":
        return 1.0 if inter else 0.0
    return len(inter) / len(expected)


def evaluate(k, mode="hybrid", pool="hybrid"):
    client = get_client()
    if client is None:
        sys.exit("OPENAI_API_KEY가 없습니다. .env를 확인하세요.")
    reports = load_index()
    if not reports:
        sys.exit("index/가 비어 있습니다. scripts/build_index.py로 먼저 인덱스를 빌드하세요.")

    settings = get_settings()
    floor = settings.similarity_floor
    golden = load_golden()
    cache = _load_cache()
    rerank_cache = _load_json(RERANK_CACHE)

    answerable = []   # (query, hit, rr, recall, difficulty, top3)
    nomatch = []      # (query, abstains_correctly, top_score, top1)
    for ex in golden:
        q = ex["query"]
        expected = set(ex["expected_db_names"])
        difficulty = ex.get("difficulty", "medium")
        match = ex.get("match", "all")
        q_emb = embed_query(client, q, cache)
        if mode == "dense":
            ranked = rank(q_emb, reports, top_k=k)
        elif mode == "hybrid":
            ranked = hybrid_rank(q, q_emb, reports, top_k=k)
        else:  # rerank: 넓은 후보 풀을 LLM 재정렬(결과 캐시 → 결정론적 재현)
            # pool=hybrid(기본): Dense+BM25 / pool=dense: Dense만 → BM25 ablation
            if pool == "dense":
                cand_pool = rank(q_emb, reports, top_k=settings.rerank_pool)
            else:
                cand_pool = hybrid_rank(q, q_emb, reports, top_k=settings.rerank_pool)
            ckey = f"{pool}||{q}||" + ",".join(r["db_name"] for _, r in cand_pool)
            if ckey in rerank_cache:
                by_name = {r["db_name"]: (s, r) for s, r in cand_pool}
                ranked = [by_name[n] for n in rerank_cache[ckey] if n in by_name][:k]
            else:
                ranked = rerank(client, q, cand_pool, k)
                rerank_cache[ckey] = [r["db_name"] for _, r in ranked]
        retrieved = [r["db_name"] for _, r in ranked]
        top_score = ranked[0][0] if ranked else 0.0

        if expected:
            rr = reciprocal_rank(retrieved, expected)
            recall = recall_at_k(retrieved, expected, match)
            answerable.append((q, rr > 0, rr, recall, difficulty, retrieved[:3]))
        else:
            # 정답 없음: 최상위 점수가 floor 미만이면 '올바르게 기권'한 것.
            nomatch.append((q, top_score < floor, top_score, retrieved[:1]))

    _save_cache(cache)
    if mode == "rerank":
        _save_json(RERANK_CACHE, rerank_cache)

    tag = f"mode={mode}" + (f", pool={pool}" if mode == "rerank" else "")
    print(f"\n=== Retrieval eval ({tag}, k={k}, 보고서 {len(reports)}건) ===")
    print(f"\n[답 있는 질의 {len(answerable)}건]")
    for q, hit, rr, recall, diff, top3 in answerable:
        print(f"  {'✓' if hit else '✗'} [{diff[0]}] rr={rr:.2f} rec={recall:.2f}  {q[:24]:24}  {top3}")
    if nomatch:
        print(f"\n[정답 없음(no_match) 질의 {len(nomatch)}건 — top<{floor}이면 정답]")
        for q, ok, ts, top1 in nomatch:
            print(f"  {'✓' if ok else '✗'} top={ts:.3f}  {q[:24]:24}  {top1}")

    print("\n--- 집계 ---")
    if answerable:
        def agg(rows, label):
            n = len(rows)
            hit = sum(1 for r in rows if r[1]) / n
            rec = sum(r[3] for r in rows) / n
            mrr = sum(r[2] for r in rows) / n
            print(f"  {label:14} (n={n:3}): Hit@{k} {hit:.3f} | Recall@{k} {rec:.3f} | MRR {mrr:.3f}")
        agg(answerable, "답 있는 질의")
        for d in ("easy", "medium", "hard"):
            sub = [r for r in answerable if r[4] == d]
            if sub:
                agg(sub, f"  └ {d}")
    if nomatch:
        nn = len(nomatch)
        abst = sum(1 for r in nomatch if r[1]) / nn
        print(f"  no_match 질의   (n={nn:3}): 기권 정확도 {abst:.3f}")


def evaluate_answer():
    """그라운딩 측정 — 전체 pipeline.search()(rerank+recommend)의 실제 응답/기권 품질.

    eval의 코사인-floor 프록시가 아니라, 앱이 실제로 내놓는 LLM recommend 결과를 채점한다.
    결과를 .answer_cache.json에 캐시(질의·코퍼스 고정 → 1회 비용 후 결정론적).
    """
    if get_client() is None:
        sys.exit("OPENAI_API_KEY가 없습니다. .env를 확인하세요.")
    reports = load_index()
    if not reports:
        sys.exit("index/가 비어 있습니다. scripts/build_index.py로 먼저 인덱스를 빌드하세요.")

    golden = load_golden()
    cache = _load_json(ANSWER_CACHE)

    ans, nm = [], []
    total = len(golden)
    for i, ex in enumerate(golden, start=1):
        q = ex["query"]
        expected = set(ex["expected_db_names"])
        if q in cache:
            res = cache[q]
        else:
            print(f"  [{i}/{total}] 검색 중: {q[:26]}", file=sys.stderr, flush=True)
            out = pipeline_search(reports, q)
            data = out.get("data") or {}
            res = {"no_match": (out.get("data") is None) or bool(data.get("no_match")),
                   "recommended": (data.get("recommended") or {}).get("db_name")}
            cache[q] = res
            _save_json(ANSWER_CACHE, cache)   # 증분 저장(중단돼도 진행분 보존)
        abstained, rec = res["no_match"], res["recommended"]
        if expected:
            ok = (not abstained) and (rec in expected)
            ans.append((q, ok, abstained, rec, ex.get("difficulty", "medium")))
        else:
            nm.append((q, abstained, rec))

    _save_json(ANSWER_CACHE, cache)

    print(f"\n=== Grounding eval (mode=answer, 보고서 {len(reports)}건) — 실제 LLM 응답 채점 ===")
    print(f"\n[답 있는 질의 {len(ans)}건]")
    for q, ok, abstained, rec, diff in ans:
        mark = "✓" if ok else ("기권" if abstained else "✗")
        print(f"  {mark:4} [{diff[0]}] {q[:24]:24} → {rec}")
    print(f"\n[정답 없음(off-domain) 질의 {len(nm)}건 — 기권해야 정답]")
    for q, abstained, rec in nm:
        print(f"  {'✓' if abstained else '✗'} {q[:24]:24} → {'(기권)' if abstained else rec}")

    print("\n--- 집계 ---")
    if ans:
        def agg(rows, label):
            n = len(rows)
            acc = sum(1 for r in rows if r[1]) / n
            over = sum(1 for r in rows if r[2]) / n
            print(f"  {label:14} (n={n:3}): 응답 정확도 {acc:.3f} | 과잉기권 {over:.3f}")
        agg(ans, "답 있는 질의")
        for d in ("easy", "medium", "hard"):
            sub = [r for r in ans if r[4] == d]
            if sub:
                agg(sub, f"  └ {d}")
    if nm:
        nn = len(nm)
        abst = sum(1 for r in nm if r[1]) / nn
        print(f"  off-domain 질의 (n={nn:3}): 기권 정확도 {abst:.3f}")


if __name__ == "__main__":
    load_dotenv(override=True)  # .env 값이 셸 환경변수보다 우선
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="top-k (기본 10)")
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "answer"],
                        default="hybrid", help="검색 모드 (기본 hybrid). answer=전체 파이프라인 그라운딩")
    parser.add_argument("--pool", choices=["dense", "hybrid"], default="hybrid",
                        help="rerank 모드의 후보 풀 (기본 hybrid). dense=BM25 제외 ablation")
    args = parser.parse_args()
    if args.mode == "answer":
        evaluate_answer()
    else:
        evaluate(args.k, args.mode, args.pool)
