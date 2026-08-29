"""CLIP image embeddings for reverse image search over the product catalog.

WHY THIS EXISTS. The text index answers "Japandi sofa under S$800" well, but
it cannot answer "find me one that looks like this". Product descriptions blur
exactly the properties furniture shopping turns on: two sofas described as
"beige fabric two-seat" can have completely different silhouettes, and a text
query naming a colour will happily return the wrong one because the colour word
is one token among fifty. CLIP embeds the actual product photograph, so
silhouette, proportion and material read directly.

WHAT IT IS NOT. CLIP does not know prices, stock, or whether a piece fits the
room. It is one signal among several, fused with the text score rather than
replacing it - see rag_engine.reverse_search.

COST AND CACHING. Embedding is the expensive part: every product image must be
downloaded and pushed through the model once (~0.2s each on CPU). The result is
cached to disk keyed by item id AND image URL, so a re-run is instant and a
changed image is re-embedded automatically. The model itself loads lazily, so
importing this module costs nothing - which matters because seed_data, solver
and the render selftests all import the package without needing CLIP.

OFFLINE. open_clip weights are downloaded once from HuggingFace and cached by
the library. After that this runs with no network and no API key, which keeps
the project's offline guarantee intact.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent / "assets" / "clip_cache.npz"

# ViT-B-32 is the smallest CLIP that is still good at this, and its 512-dim
# output keeps the extra Qdrant vector cheap. The LAION-2B weights outperform
# OpenAI's original on retrieval.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512


class ClipEmbedder:
    """Lazily-loaded CLIP, embedding both images and text into one space.

    Thread-safe: FastAPI may call this from a threadpool, and loading the model
    twice concurrently would waste a couple of hundred MB for no reason.
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.available = False
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401

            self.available = True
        except ImportError:
            # Not an error: image search is an enhancement, and every other
            # part of the app works without it.
            log.info(
                "open_clip/torch not installed; image search disabled "
                "(pip install open_clip_torch to enable)"
            )

    def _ensure(self) -> None:
        if self._model is not None or not self.available:
            return
        with self._lock:
            if self._model is not None:
                return
            import open_clip

            log.info("loading CLIP %s/%s…", MODEL_NAME, PRETRAINED)
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            log.info("CLIP ready (%d-dim)", EMBED_DIM)

    # --- embedding --------------------------------------------------------

    def embed_image(self, raw: bytes) -> np.ndarray | None:
        """A unit-normalized image vector, or None if the bytes are unusable."""
        self._ensure()
        if self._model is None:
            return None
        try:
            import torch
            from PIL import Image

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0)
            with torch.no_grad():
                vec = self._model.encode_image(tensor)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            log.info("could not embed image: %s", exc)
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
        """A unit-normalized text vector in the SAME space as embed_image.

        This is what makes a caption searchable against product photos: CLIP
        aligns both modalities, so a described object can be matched to images
        without ever rendering it.
        """
        self._ensure()
        if self._model is None or not text.strip():
            return None
        try:
            import torch

            with torch.no_grad():
                vec = self._model.encode_text(self._tokenizer([text]))
                vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            log.info("could not embed text: %s", exc)
            return None


# --- catalog image vectors -------------------------------------------------


def _fetch(url: str) -> bytes | None:
    """Download one product image, honouring the same guards as the renderer.

    A `data:` URI is decoded in place rather than fetched: a merchant with no
    CDN carries its photography embedded in the catalog, and there is no host
    to allowlist or round-trip. The size ceiling still applies.
    """
    from .render_engine import _fetchable

    if url.startswith("data:"):
        try:
            head, _, payload = url.partition(",")
            if not payload or ";base64" not in head:
                return None
            raw = base64.b64decode(payload)
        except Exception as exc:
            log.info("could not decode data: image: %s", exc)
            return None
        return raw if len(raw) <= config.IMAGE_FETCH_MAX_BYTES else None

    if not _fetchable(url):
        return None
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": config.IMAGE_FETCH_USER_AGENT}
        )
        with urllib.request.urlopen(
            req, timeout=config.IMAGE_FETCH_TIMEOUT_SECONDS
        ) as resp:
            data = resp.read(config.IMAGE_FETCH_MAX_BYTES + 1)
            if len(data) > config.IMAGE_FETCH_MAX_BYTES:
                return None
            return data
    except Exception as exc:
        log.info("could not fetch %s: %s", url[:80], exc)
        return None


def _load_cache() -> dict[str, np.ndarray]:
    """Cached vectors, keyed by "item_id|image_url".

    The URL is part of the key deliberately: a catalog edit that repoints an
    item at a different photo must invalidate that entry, and keying on the id
    alone would silently serve the old vector forever.
    """
    if not CACHE_PATH.exists():
        return {}
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as exc:
        log.warning("CLIP cache unreadable, rebuilding: %s", exc)
        return {}


def _image_key(url: str) -> str:
    """The cache-key half that identifies WHICH image a vector came from.

    Normally the URL itself, so a changed product photo re-embeds. An embedded
    `data:` URI is the image, though, and can run to tens of KB - npz stores
    keys as zip entry names, which cap at 65535 bytes, so the whole cache
    silently fails to write. Hashing the payload keeps the same property (a
    different photo yields a different key) at a fixed 16 bytes.
    """
    if url.startswith("data:"):
        return "data:" + hashlib.blake2b(url.encode(), digest_size=16).hexdigest()
    return url


def _save_cache(vectors: dict[str, np.ndarray]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, **vectors)
    except Exception as exc:
        log.warning("could not write CLIP cache: %s", exc)


def embed_catalog(
    items: list,
    embedder: ClipEmbedder | None = None,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Image vectors for every item, downloading and embedding only what is new.

    Returns {item_id: vector}. Items whose image cannot be fetched or embedded
    are simply absent rather than given a zero vector: a zero vector is
    equidistant from everything and would pollute every single search result.
    """
    embedder = embedder or ClipEmbedder()
    if not embedder.available:
        return {}

    cache = _load_cache()
    out: dict[str, np.ndarray] = {}
    fetched = 0

    for item in items:
        key = f"{item.id}|{_image_key(item.image_url)}"
        hit = cache.get(key)
        if hit is not None:
            out[item.id] = hit
            continue
        raw = _fetch(item.image_url)
        if raw is None:
            continue
        vec = embedder.embed_image(raw)
        if vec is None:
            continue
        cache[key] = vec
        out[item.id] = vec
        fetched += 1
        if progress and fetched % 25 == 0:
            print(f"  embedded {fetched} new images…")

    if fetched:
        _save_cache(cache)
        log.info("CLIP: embedded %d new images (%d cached)", fetched, len(out) - fetched)
    return out


if __name__ == "__main__":
    import time

    from .seed_data import SEED_ITEMS

    logging.basicConfig(level=logging.INFO)
    emb = ClipEmbedder()
    if not emb.available:
        raise SystemExit("open_clip/torch not installed")

    started = time.time()
    vectors = embed_catalog(SEED_ITEMS, emb, progress=True)
    elapsed = time.time() - started
    print(f"\n{len(vectors)}/{len(SEED_ITEMS)} items have image vectors "
          f"({elapsed:.1f}s)")
    missing = [i.id for i in SEED_ITEMS if i.id not in vectors]
    if missing:
        print(f"no vector for {len(missing)}: {missing[:5]}")

    # A quick sanity check that the space is doing something sensible: the
    # nearest neighbour of an item should be a plausible lookalike.
    if vectors:
        ids = list(vectors)
        matrix = np.stack([vectors[i] for i in ids])
        by_id = {i.id: i for i in SEED_ITEMS}
        probe = ids[0]
        sims = matrix @ vectors[probe]
        order = np.argsort(-sims)[1:4]
        print(f"\nnearest to {by_id[probe].title[:46]}:")
        for idx in order:
            print(f"   {sims[idx]:.3f}  {by_id[ids[idx]].title[:46]}")
