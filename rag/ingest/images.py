"""HWP 임베디드 이미지 추출 + vision 설명(공정흐름도·시스템경계 다이어그램).

HWP(OLE)의 BinData 스트림에서 PNG/BMP를 꺼내고, 팔레트+투명 PNG는 흰 배경으로 평탄화한 뒤
OpenAI vision으로 다이어그램을 텍스트 설명으로 변환한다. 로고·공통 템플릿(여러 문서에 반복되는
동일 이미지)은 해시 빈도로 걸러낸다. 빌드 시점 1회 비용(서빙엔 텍스트 설명만 남는다).

⚠️ 평탄화 필수: 원본은 팔레트(P)+투명이라, 그대로 보내면 vision이 '검은 이미지'로 받아 헛읽는다.
"""
import base64
import hashlib
import io
import zlib

import olefile
from PIL import Image

from rag.config import get_settings

_PNG = b"\x89PNG"
_BMP = b"BM"
_COMMON_MAX = 3  # 해시가 이 수보다 많은 문서에 나오면 로고·공통 템플릿으로 보고 제외


def extract_image_blobs(hwp_path):
    """HWP에서 (stream_name, image_bytes) 목록을 추출합니다(PNG/BMP, 필요시 압축 해제)."""
    blobs = []
    ole = olefile.OleFileIO(str(hwp_path))
    try:
        for e in ole.listdir():
            if not (len(e) > 1 and e[0] == "BinData"):
                continue
            raw = ole.openstream(e).read()
            data = raw
            if data[:4] != _PNG and data[:2] != _BMP:
                try:
                    data = zlib.decompress(raw, -15)
                except Exception:
                    continue
            if data[:4] == _PNG or data[:2] == _BMP:
                blobs.append(("/".join(e), data))
    finally:
        ole.close()
    return blobs


def image_sha(data):
    return hashlib.sha256(data).hexdigest()


def count_image_hashes(hwp_paths):
    """전체 코퍼스의 이미지 해시 빈도(로고·공통 템플릿 식별용)."""
    counts = {}
    for p in hwp_paths:
        seen = set()
        for _, data in extract_image_blobs(p):
            h = image_sha(data)
            if h not in seen:          # 한 문서 내 중복은 1회만
                seen.add(h)
                counts[h] = counts.get(h, 0) + 1
    return counts


def flatten_to_png(data):
    """투명/팔레트 이미지를 흰 배경 RGB PNG로 평탄화합니다(vision 검은 이미지 방지)."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    buf = io.BytesIO()
    bg.save(buf, "PNG")
    return buf.getvalue()


def describe_image(client, data, usage=None):
    """이미지를 vision 모델로 한국어 설명 텍스트로 변환합니다(실패/빈 결과 시 None)."""
    settings = get_settings()
    try:
        b64 = base64.b64encode(flatten_to_png(data)).decode()
    except Exception:
        return None
    try:
        r = client.chat.completions.create(
            model=settings.vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text":
                 "이 LCI 보고서 다이어그램(공정흐름도·시스템경계 등)에 보이는 글자를 그대로 옮기고, "
                 "입력물→공정→출력물 흐름을 한국어 1~2문장으로 요약하세요. 추측하지 말고 보이는 것만 쓰세요."},
                {"type": "image_url", "image_url":
                 {"url": f"data:image/png;base64,{b64}", "detail": "original"}}]}],
        )
    except Exception:
        return None
    if usage is not None:
        usage.record("이미지", settings.vision_model, r)
    text = (r.choices[0].message.content or "").strip()
    return text or None


def image_chunks(client, hwp_path, common_hashes, usage=None):
    """문서의 비-공통 이미지를 vision으로 설명해 청크 목록으로 반환합니다.

    반환: [{"text", "section": "공정흐름도", "kind": "image"}]. 로고·공통 템플릿은 제외.
    """
    chunks, seen = [], set()
    for _, data in extract_image_blobs(hwp_path):
        h = image_sha(data)
        if h in seen or common_hashes.get(h, 0) > _COMMON_MAX:
            continue
        seen.add(h)
        desc = describe_image(client, data, usage=usage)
        if desc:
            chunks.append({"text": f"[공정흐름도] {desc}", "section": "공정흐름도", "kind": "image"})
    return chunks
