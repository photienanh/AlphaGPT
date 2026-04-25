"""
knowledge/build_kb.py
Script chạy một lần để build vector DB từ 101 Formulaic Alphas.
Sử dụng sentence-transformers (all-MiniLM-L6-v2) để embed descriptions.
Lưu FAISS index tại knowledge/faiss_index/ và metadata tại knowledge/alpha_kb.json.

Usage:
    python knowledge/build_kb.py
"""
import os
import json
import sys
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KNOWLEDGE_DIR     = "data/knowledge_base"
INDEX_DIR  = os.path.join(KNOWLEDGE_DIR, "faiss_index")
KB_JSON    = os.path.join(KNOWLEDGE_DIR, "alpha_kb.json")

sys.path.insert(0, os.path.dirname(KNOWLEDGE_DIR))
from knowledge.alpha_kb_data import ALPHA_KB


def build_descriptions(alphas: list) -> list[str]:
    """Tạo text đầy đủ để embed cho mỗi alpha."""
    texts = []
    for a in alphas:
        text = (
            f"Description: {a.get('description', '')}. "
            f"Expression: {a.get('expression', '')}"
        )
        texts.append(text)
    return texts


def build_kb():
    os.makedirs(INDEX_DIR, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError as e:
        log.error(f"Thiếu dependency: {e}")
        log.error("Chạy: pip install sentence-transformers faiss-cpu --break-system-packages")
        sys.exit(1)

    log.info(f"Số alphas trong KB: {len(ALPHA_KB)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Đã load model all-MiniLM-L6-v2")

    texts = build_descriptions(ALPHA_KB)
    log.info("Đang embed descriptions...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    embeddings = embeddings.astype(np.float32)

    # Normalize để dùng cosine similarity (inner product sau normalize)
    import faiss
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-9)

    dim = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Lưu index
    index_path = os.path.join(INDEX_DIR, "alpha_index.faiss")
    faiss.write_index(index, index_path)
    log.info(f"FAISS index saved: {index_path}")

    # Lưu metadata
    with open(KB_JSON, "w", encoding="utf-8") as f:
        json.dump(ALPHA_KB, f, ensure_ascii=False, indent=2)
    log.info(f"KB metadata saved: {KB_JSON}")

    # Lưu texts để biết thứ tự index
    texts_path = os.path.join(INDEX_DIR, "texts.json")
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    log.info(f"Embed texts saved: {texts_path}")

    log.info("Build KB hoàn tất!")


if __name__ == "__main__":
    build_kb()