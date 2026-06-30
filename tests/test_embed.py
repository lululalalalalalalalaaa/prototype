"""임베딩 검증 — 가중평균 수학 + 단일/다중 청크 경로(가짜 클라이언트)."""
import rag.embed as embed_mod
from rag.config import Settings
from rag.embed import _average_vectors, embed_text, embed_texts


def test_embed_texts_batch_ordered(fake_client):
    # 길이 기반 벡터로 입력 순서대로 반환되는지 확인(index 정렬)
    client = fake_client(embed_vector_fn=lambda s, i: [float(len(s))])
    out = embed_texts(client, ["a", "bb", "ccc"])
    assert out == [[1.0], [2.0], [3.0]]
    assert len(client.embed_calls) == 1          # 배치 1회 호출
    assert client.embed_calls[0]["input"] == ["a", "bb", "ccc"]


def test_embed_texts_empty_returns_empty(fake_client):
    client = fake_client()
    assert embed_texts(client, []) == []


def test_average_vectors_weighted():
    # [2,0]*1 + [0,2]*3, 가중치 합 4 → [0.5, 1.5]
    assert _average_vectors([[2.0, 0.0], [0.0, 2.0]], [1, 3]) == [0.5, 1.5]


def test_average_vectors_zero_weight_guard():
    # 가중치 합 0이면 1로 나눠 0벡터 반환(0분모 방지)
    assert _average_vectors([[1.0, 1.0]], [0]) == [0.0, 0.0]


def test_embed_text_single_chunk(fake_client):
    # 짧은 텍스트 → 청킹 없이 단일 임베딩 호출, 그 벡터를 그대로 반환
    client = fake_client(embed_vector_fn=lambda s, i: [1.0, 2.0, 3.0])
    vec = embed_text(client, "짧은 본문")
    assert vec == [1.0, 2.0, 3.0]
    assert len(client.embed_calls) == 1
    assert client.embed_calls[0]["input"] == "짧은 본문"  # 리스트가 아닌 단일 문자열


def test_embed_text_multi_chunk_path(fake_client, monkeypatch):
    # embed_max_tokens를 작게 만들어 다중 청크 경로 강제
    small = Settings(model="m", vision_model="v", embedding_model="e", top_k=5,
                     similarity_floor=0.4, others_ratio=0.85,
                     embed_encoding="cl100k_base", embed_max_tokens=4)
    monkeypatch.setattr(embed_mod, "get_settings", lambda: small)

    # 모든 청크가 동일 벡터를 주면, 토큰수 가중치와 무관하게 평균도 그 벡터여야 함.
    # 응답을 일부러 역순으로 돌려 index 기준 재정렬도 함께 검증.
    class ReversingClient:
        def __init__(self):
            self.embeddings = self
            self.last_input = None
        def create(self, model, input):
            from types import SimpleNamespace
            self.last_input = input
            data = [SimpleNamespace(index=i, embedding=[1.0, 1.0])
                    for i, _ in enumerate(input)]
            return SimpleNamespace(data=list(reversed(data)))  # 순서 뒤섞기

    client = ReversingClient()
    text = "토큰 " * 50  # 4토큰 한도를 확실히 초과 → 다중 청크
    vec = embed_text(client, text)

    assert isinstance(client.last_input, list)   # 다중 청크 경로(리스트 입력)
    assert len(client.last_input) > 1
    assert vec == [1.0, 1.0]                      # 동일 벡터의 가중평균 = 동일 벡터
