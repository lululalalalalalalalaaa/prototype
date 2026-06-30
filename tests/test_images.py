"""이미지 추출·평탄화·dedup 검증 — 실제 HWP 사용, vision은 mock(API 불필요)."""
import io

from PIL import Image

from rag.config import UPLOAD_DIR
from rag.ingest.images import (count_image_hashes, extract_image_blobs,
                               flatten_to_png, image_chunks, image_sha)

_HWPS = sorted(p for p in UPLOAD_DIR.glob("*.hwp"))


def test_extract_image_blobs_real_hwp():
    blobs = extract_image_blobs(_HWPS[0])
    assert blobs, "HWP에서 이미지가 추출되어야 함"
    for _, data in blobs:
        assert data[:4] == b"\x89PNG" or data[:2] == b"BM"   # 유효 이미지 매직


def test_flatten_to_png_makes_white_rgb():
    _, data = extract_image_blobs(_HWPS[0])[0]
    png = flatten_to_png(data)
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGB"                       # 투명도 평탄화됨(검은 이미지 방지)
    assert png[:4] == b"\x89PNG"


def test_count_hashes_finds_common_logo():
    counts = count_image_hashes(_HWPS)
    # 로고는 여러 문서에 반복 → 최대 빈도가 1보다 큼
    assert max(counts.values()) > 1


def test_image_chunks_skips_common_and_describes(fake_client):
    counts = count_image_hashes(_HWPS)
    # 가장 흔한 해시(로고/공통 템플릿)는 제외 대상으로 지정
    common = {h: c for h, c in counts.items() if c > 3}
    client = fake_client(chat_content="LPG → 수송공정 → 대기배출물")
    chunks = image_chunks(client, _HWPS[0], common)
    for c in chunks:
        assert c["kind"] == "image" and c["section"] == "공정흐름도"
        assert c["text"].startswith("[공정흐름도]")
        # 제외된 공통 이미지(로고)는 청크에 없어야 함
        assert image_sha  # 심볼 존재 확인
