"""
knowledge/retriever.py
Retrieve similar alphas từ FAISS index.
Index được build tự động lần đầu và rebuild khi alpha_library có thêm entries.
Nguồn embed:
  - 38 alphas từ Kakushadze paper (alpha_kb_data.py)
  - Tất cả alpha từ alpha_library.json (tích lũy qua các run)
"""
import os
import sys
import json
import random
import logging
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

KB_DIR     = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR  = os.path.join(KB_DIR, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "alpha_index.faiss")
META_PATH  = os.path.join(INDEX_DIR, "alpha_meta.json")  # metadata song song với index

LIBRARY_PATH = os.environ.get("ALPHA_LIBRARY_PATH", "alpha_library.json")

_index_cache = None
_meta_cache: Optional[List[Dict]] = None
_model_cache = None


# ── Helpers ───────────────────────────────────────────────────────────

def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        import logging as _logging
        # Tắt log verbose từ sentence_transformers và huggingface khi load model
        for _lib in ["sentence_transformers", "transformers", "huggingface_hub",
                     "httpx", "httpcore", "filelock"]:
            _logging.getLogger(_lib).setLevel(_logging.ERROR)

        from sentence_transformers import SentenceTransformer
        # Thử load local trước (đã cache), tránh HTTP request mỗi lần
        try:
            _model_cache = SentenceTransformer(
                "all-MiniLM-L6-v2", local_files_only=True
            )
        except Exception:
            # Chưa cache → download lần đầu
            log.info("[Retriever] Downloading embedding model (chỉ lần đầu)...")
            _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
        return _model_cache
    except ImportError:
        return None


def _make_embed_text(alpha: Dict) -> str:
    return (
        f"Family: {alpha.get('family', 'unknown')}. "
        f"Description: {alpha.get('description', '')}. "
        f"Expression: {alpha.get('expression', '')}"
    )


def _collect_all_alphas() -> List[Dict]:
    """
    Gom tất cả alphas cần embed:
    1. Kakushadze KB (alpha_kb_data.py) — luôn có
    2. alpha_library.json — alpha OK từ các run thực
    """
    sys.path.insert(0, os.path.dirname(KB_DIR))
    from knowledge.alpha_kb_data import ALPHA_KB
    all_alphas = list(ALPHA_KB)

    if os.path.exists(LIBRARY_PATH):
        try:
            with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
                library = json.load(f)
            # Thêm source tag để phân biệt
            for a in library:
                a = dict(a)
                a["source"] = "historical_run"
                all_alphas.append(a)
            log.debug(f"[Retriever] Gom {len(ALPHA_KB)} KB + {len(library)} library = {len(all_alphas)} alphas")
        except Exception as e:
            log.warning(f"[Retriever] Không đọc được alpha_library.json: {e}")

    return all_alphas


def _index_is_stale() -> bool:
    """
    Kiểm tra index có cần rebuild không:
    - Chưa có index → stale
    - alpha_library.json mới hơn index → stale
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return True
    if not os.path.exists(LIBRARY_PATH):
        return False
    return os.path.getmtime(LIBRARY_PATH) > os.path.getmtime(INDEX_PATH)


def _build_index(all_alphas: List[Dict]) -> bool:
    """
    Build FAISS index từ all_alphas.
    Trả về True nếu thành công.
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        log.warning("[Retriever] faiss chưa install — pip install faiss-cpu")
        return False

    model = _load_model()
    if model is None:
        log.warning("[Retriever] sentence-transformers chưa install — pip install sentence-transformers")
        return False

    os.makedirs(INDEX_DIR, exist_ok=True)

    texts = [_make_embed_text(a) for a in all_alphas]
    log.info(f"[Retriever] Building FAISS index cho {len(texts)} alphas...")

    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = embeddings.astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-9)

    dim   = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)

    faiss.write_index(index, INDEX_PATH)

    # Lưu metadata song song — giữ đủ info để trả về khi retrieve
    meta = []
    for a in all_alphas:
        meta.append({
            "id":          a.get("id", ""),
            "expression":  a.get("expression", ""),
            "description": a.get("description", ""),
            "family":      a.get("family", "unknown"),
            "source":      a.get("source", "kakushadze"),
            "ic_oos":      a.get("ic_oos"),
        })
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(f"[Retriever] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return True


def _load_index():
    """Load index và metadata vào cache. Build nếu stale."""
    global _index_cache, _meta_cache

    if _index_is_stale():
        log.info("[Retriever] Index stale hoặc chưa có — rebuilding...")
        all_alphas = _collect_all_alphas()
        ok = _build_index(all_alphas)
        if not ok:
            _index_cache = None
            _meta_cache  = None
            return None, None
        # Invalidate cache
        _index_cache = None
        _meta_cache  = None

    if _index_cache is not None and _meta_cache is not None:
        return _index_cache, _meta_cache

    if not os.path.exists(INDEX_PATH):
        return None, None

    try:
        import faiss
        _index_cache = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta_cache = json.load(f)
        log.debug(f"[Retriever] Loaded index: {_index_cache.ntotal} vectors")
        return _index_cache, _meta_cache
    except Exception as e:
        log.warning(f"[Retriever] Không load được index: {e}")
        return None, None


# ── Public API ────────────────────────────────────────────────────────

def invalidate_cache():
    """
    Gọi sau khi alpha_library.json được cập nhật
    để lần retrieve tiếp theo sẽ rebuild index.
    """
    global _index_cache, _meta_cache
    _index_cache = None
    _meta_cache  = None
    log.debug("[Retriever] Cache invalidated")


def load_alpha_kb() -> List[Dict]:
    """Load Kakushadze KB (dùng cho fallback alphas)."""
    sys.path.insert(0, os.path.dirname(KB_DIR))
    from knowledge.alpha_kb_data import ALPHA_KB
    return list(ALPHA_KB)


def retrieve_similar_alphas(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k alphas liên quan đến query, đảm bảo family diversity.
    Tối đa 2 alpha mỗi family trong kết quả trả về.
    """
    index, meta = _load_index()

    if index is None or meta is None:
        all_alphas = _collect_all_alphas()
        return _diverse_sample(all_alphas, top_k)

    model = _load_model()
    if model is None:
        all_alphas = _collect_all_alphas()
        return _diverse_sample(all_alphas, top_k)

    try:
        import numpy as np
        vec = model.encode([query]).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_norm = vec / (norm + 1e-9)

        # Lấy nhiều hơn top_k để có chỗ filter diversity
        k_fetch = min(top_k * 3, index.ntotal)
        _, indices = index.search(vec_norm, k_fetch)

        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(meta):
                candidates.append(meta[idx])

        return _ensure_family_diversity(candidates, top_k)

    except Exception as e:
        log.warning(f"[Retriever] FAISS search failed: {e}")
        all_alphas = _collect_all_alphas()
        return _diverse_sample(all_alphas, top_k)
    
def _ensure_family_diversity(candidates: List[Dict], top_k: int,
                              max_per_family: int = 2) -> List[Dict]:
    """
    Từ danh sách candidates (đã sorted theo similarity),
    chọn top_k với constraint: tối đa max_per_family alpha mỗi family.
    """
    family_count = {}
    results = []

    for c in candidates:
        fam = c.get("family", "unknown")
        if family_count.get(fam, 0) < max_per_family:
            results.append(c)
            family_count[fam] = family_count.get(fam, 0) + 1
        if len(results) >= top_k:
            break

    # Bổ sung nếu chưa đủ top_k (bỏ qua constraint diversity)
    if len(results) < top_k:
        for c in candidates:
            if c not in results:
                results.append(c)
            if len(results) >= top_k:
                break

    return results


def _diverse_sample(alphas: List[Dict], top_k: int) -> List[Dict]:
    """Random sample với family diversity cho fallback case."""
    if len(alphas) <= top_k:
        return list(alphas)

    # Group by family
    by_family = {}
    for a in alphas:
        fam = a.get("family", "unknown")
        by_family.setdefault(fam, []).append(a)

    results = []
    families = list(by_family.keys())
    random.shuffle(families)

    # Round-robin từ các family
    while len(results) < top_k and any(by_family.values()):
        for fam in families:
            if by_family[fam] and len(results) < top_k:
                idx = random.randrange(len(by_family[fam]))
                results.append(by_family[fam].pop(idx))

    return results