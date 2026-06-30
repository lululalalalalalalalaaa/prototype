"""테스트 공용 fixture — API 키 없이 결정론적으로 검증하기 위한 가짜 OpenAI 클라이언트."""
from types import SimpleNamespace

import pytest


class _FakeEmbeddings:
    def __init__(self, owner, vector_fn):
        self._owner = owner
        self._vector_fn = vector_fn

    def create(self, model, input):
        self._owner.embed_calls.append({"model": model, "input": input})
        items = input if isinstance(input, list) else [input]
        data = [SimpleNamespace(index=i, embedding=self._vector_fn(s, i))
                for i, s in enumerate(items)]
        return SimpleNamespace(data=data)


class _FakeCompletions:
    def __init__(self, owner, content):
        self._owner = owner
        self._content = content

    def create(self, model, messages, response_format=None):
        self._owner.chat_calls.append(
            {"model": model, "messages": messages, "response_format": response_format}
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class FakeClient:
    """OpenAI 클라이언트 대역. 임베딩 벡터·채팅 응답을 주입해 결정론적으로 테스트합니다."""

    def __init__(self, *, embed_vector_fn=None, chat_content=None):
        self.embed_calls = []
        self.chat_calls = []
        vector_fn = embed_vector_fn or (lambda s, i: [float(len(s)), float(i)])
        self.embeddings = _FakeEmbeddings(self, vector_fn)
        self.chat = SimpleNamespace(completions=_FakeCompletions(self, chat_content))


@pytest.fixture
def fake_client():
    """기본 가짜 클라이언트 팩토리."""
    def _make(*, embed_vector_fn=None, chat_content=None):
        return FakeClient(embed_vector_fn=embed_vector_fn, chat_content=chat_content)
    return _make
