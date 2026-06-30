"""청킹 — 보고서 본문을 검색용 청크로 나눕니다.

전략: 본문을 문단(`\n` 단위)으로 나눈 뒤, 토큰 한도(chunk_tokens)까지 문단을 그리디로
묶어 윈도우를 만들고, 인접 윈도우 사이에 chunk_overlap 토큰만큼 겹친다. 한 문단이 한도를
넘으면 토큰 단위로 쪼갠다. 토큰 계산은 embed.get_encoder(tiktoken)를 재사용한다.

각 청크의 임베딩 입력에는 DB 이름을 prefix로 붙여(build_chunk_input), 짧은 질의에도
이름 신호가 남도록 한다.
"""
import re

from rag.config import get_settings
from rag.embed import get_encoder

# 정부 LCI 보고서 구조 마커(37문서 일관): 섹션 'N. 제목', 표 '표.', 그림 '그림.'
_SEC_RE = re.compile(r"^\d+\.\s+\S")
_TBL_RE = re.compile(r"^(표\s*[.·]|\[\s*표)")
_FIG_RE = re.compile(r"^(그림\s*[.·]|\[\s*그림)")


def build_chunk_input(db_name, chunk):
    """청크의 임베딩 입력 텍스트(LCI DB 이름 + 섹션 맥락 + 청크 본문).

    chunk는 structure_chunks가 만든 dict({text, section, kind}). 섹션명을 prefix로 붙여
    고립된 청크도 어느 섹션(예: 영향평가 결과)인지 신호가 남게 한다(konetic context_text 아이디어).
    """
    section = (chunk.get("section") or "").strip()
    head = f"{db_name} | {section}".rstrip(" |")
    return f"{head}\n{chunk['text']}"


def _split_segments(body):
    """본문을 (섹션명, kind, 텍스트) 세그먼트로 나눈다. kind: 'body'|'table'.

    'N. 제목' 헤더로 섹션 경계를, '표.' 캡션으로 표 영역(다음 헤더 전까지)을 분리한다.
    섹션 헤더가 없는 본문(txt 등)은 통째로 ('', 'body', 본문) 한 세그먼트가 된다.
    """
    lines = body.split("\n")
    segs, section, buf = [], "", []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            segs.append((section, "body", text))
        buf.clear()

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if _SEC_RE.match(s) and len(s) < 40 and not s.endswith(">"):
            flush()
            section = re.sub(r"^\d+\.\s+", "", s).strip()
            i += 1
        elif _TBL_RE.match(s):
            flush()
            caption = re.sub(r"^(표\s*[.·]|\[\s*표\s*\d*\]?)\s*", "", s).strip()
            tbuf, i = [lines[i]], i + 1
            while i < n:
                t = lines[i].strip()
                if (_SEC_RE.match(t) and len(t) < 40) or _TBL_RE.match(t) or _FIG_RE.match(t):
                    break
                tbuf.append(lines[i])
                i += 1
            text = "\n".join(tbuf).strip()
            if text:
                segs.append((caption or section, "table", text))
        else:
            buf.append(lines[i])
            i += 1
    flush()
    return segs


def structure_chunks(body, max_tokens=None, overlap=None):
    """본문을 섹션·표 단위로 나눈 뒤 각 세그먼트를 토큰 윈도우로 청크화합니다.

    반환: [{"text", "section", "kind"}] (kind: 'body'|'table').
    섹션 헤더로 경계를 잡아 '한 청크 = 한 주제'가 되게 하고, 표('표.')는 독립 청크로
    분리해(kind='table') 영향평가 수치 검색 정밀도를 높인다. 큰 세그먼트는 chunk_body로 윈도우 분할.
    """
    out = []
    for section, kind, text in _split_segments(body):
        for ct in chunk_body(text, max_tokens, overlap):
            out.append({"text": ct, "section": section, "kind": kind})
    return out


def chunk_body(body, max_tokens=None, overlap=None):
    """본문을 문단 기반 토큰 윈도우 청크 목록(문자열)으로 나눕니다.

    - 각 청크의 토큰 수 ≤ max_tokens (문단 사이 줄바꿈 비용까지 고려)
    - 인접 청크는 overlap 토큰만큼 내용을 공유(문맥 단절 완화)
    - 빈 본문 → []
    """
    settings = get_settings()
    max_tokens = settings.chunk_tokens if max_tokens is None else max_tokens
    overlap = settings.chunk_overlap if overlap is None else overlap
    enc = get_encoder(settings.embed_encoding)

    paras = [p for p in (line.strip() for line in body.split("\n")) if p]
    if not paras:
        return []

    # 1) 각 문단을 단위(unit)로. 한도를 넘는 문단은 토큰 단위로 분할.
    units = []  # (text, ntokens)
    for p in paras:
        toks = enc.encode(p)
        if len(toks) <= max_tokens:
            units.append((p, len(toks)))
        else:
            for i in range(0, len(toks), max_tokens):
                sub = enc.decode(toks[i:i + max_tokens])
                units.append((sub, len(enc.encode(sub))))

    # 2) 그리디 패킹. 윈도우에 단위를 추가할 때 줄바꿈(1토큰) 비용도 더해 한도를 지킴.
    #    윈도우 간에는 끝쪽 단위를 overlap 토큰까지 이월해 겹침을 만든다.
    chunks = []
    n = len(units)
    start = 0
    while start < n:
        end = start
        tok = 0
        while end < n:
            cost = units[end][1] + (1 if end > start else 0)  # 둘째 단위부터 줄바꿈 1토큰
            if tok + cost > max_tokens:
                break
            tok += cost
            end += 1
        if end == start:          # 단일 단위가 그 자체로 한도(분할 결과) → 강제 포함
            end = start + 1
        chunks.append("\n".join(u[0] for u in units[start:end]))
        if end >= n:
            break
        # 다음 시작점: 끝에서부터 overlap 토큰까지 단위를 되돌려 겹침 생성(진행 보장)
        back = end
        carry = 0
        while back - 1 > start and carry + units[back - 1][1] <= overlap:
            carry += units[back - 1][1]
            back -= 1
        start = back if back > start else end
    return chunks
