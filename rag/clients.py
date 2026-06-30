"""OpenAI 클라이언트 단일 생성 지점.

임베딩(embed.py)과 생성(generate.py)이 모두 이 함수만 사용합니다.
추후 로컬(Ollama 등)로 교체할 때 이 한 곳만 바꾸면 됩니다.
"""
import os

from openai import OpenAI


def get_client():
    """OPENAI_API_KEY로 OpenAI 클라이언트를 만듭니다. 키가 없으면 None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)
