"""임베딩 벡터 생성.

원본 app.py에서 그대로 옮기되, 토크나이저 캐싱을 Streamlit(st.cache_resource)에서
표준 functools.lru_cache로 바꿔 rag 패키지가 UI에 의존하지 않게 했습니다(동작 동일).
"""
from functools import lru_cache

import tiktoken

from rag.config import get_settings


@lru_cache(maxsize=4)
def get_encoder(encoding):
    """임베딩 모델용 토크나이저(인코딩별 1회 로드 후 캐시)."""
    return tiktoken.get_encoding(encoding)


def embed_texts(client, texts):
    """여러 텍스트를 한 번의 API 호출로 임베딩합니다(각 텍스트는 토큰 한도 내라고 가정).

    응답 순서를 index로 보장해 입력 순서대로 벡터 리스트를 반환합니다. 빈 입력 → [].
    청크 임베딩처럼 짧은 텍스트 다수를 효율적으로 처리할 때 사용합니다.
    """
    if not texts:
        return []
    resp = client.embeddings.create(model=get_settings().embedding_model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def _average_vectors(vectors, weights):
    """여러 임베딩 벡터를 가중평균해 하나의 벡터로 합칩니다."""
    dim = len(vectors[0])
    total = sum(weights) or 1
    avg = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for k in range(dim):
            avg[k] += vec[k] * w
    return [x / total for x in avg]


def embed_text(client, text, usage=None):
    """텍스트를 임베딩 벡터로 변환합니다.

    임베딩 모델의 토큰 한도(8,191)를 넘는 긴 본문은 embed_max_tokens씩 여러 청크로
    나눠 한 번의 API 호출로 각각 임베딩한 뒤, 청크의 토큰 수로 가중평균하여
    하나의 벡터로 합칩니다(긴 보고서가 한도 초과로 학습 실패하지 않도록).
    usage(UsageTracker)를 주면 토큰 사용량을 기록합니다.
    """
    settings = get_settings()
    enc = get_encoder(settings.embed_encoding)
    tokens = enc.encode(text)
    if len(tokens) <= settings.embed_max_tokens:
        resp = client.embeddings.create(model=settings.embedding_model, input=text)
        if usage is not None:
            usage.record("임베딩", settings.embedding_model, resp)
        return resp.data[0].embedding

    chunks = [tokens[i:i + settings.embed_max_tokens]
              for i in range(0, len(tokens), settings.embed_max_tokens)]
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=[enc.decode(c) for c in chunks],
    )
    if usage is not None:
        usage.record("임베딩", settings.embedding_model, resp)
    # 응답 순서를 index로 보장한 뒤, 청크 토큰 수를 가중치로 평균.
    vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    return _average_vectors(vectors, [len(c) for c in chunks])
