"""Scrape Castlery Singapore listings into the same shape as `products.json`.

WHY A SECOND MERCHANT. The IKEA scrape has no TV units, media consoles or
sideboards in it, and its price band tops out around S$1,699. Castlery is a
Singapore retailer, so its prices are already SGD and need no invented
exchange rate, and it publishes the categories IKEA's scrape is missing.

WHERE THE DATA COMES FROM. Every product page carries a schema.org `Product`
block in `application/ld+json`, which is the *vendor's own* structured feed -
sku, name, category, description, price, availability, dimensions, images,
rating and reviews. That is parsed instead of the rendered HTML because it is
a stable contract: the page markup is a Next.js build that changes with every
deploy, the JSON-LD does not. Nothing here reads the visual DOM.

WHAT IS POLITE ABOUT IT. robots.txt disallows only */wishlist, */checkout/ and
*/account/ - product pages are explicitly permitted, and the URL list is taken
from Castlery's own published sitemap rather than by crawling links. Requests
are serialised with a delay between them, results are cached to disk so a
re-run refetches nothing, and a failed page is recorded and skipped rather
than retried in a tight loop.

Run from the repository root:

    python -m scrapers.castlery_scrape            # full run, ~1,217 products
    python -m scrapers.castlery_scrape --limit 40 # quick sample

Writes `castlery_products.json` (the catalog) and `castlery_report.json` (what
failed and what was missing), alongside a page cache in `.cache/castlery/`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "castlery"
OUT_PRODUCTS = ROOT / "castlery_products.json"
OUT_REPORT = ROOT / "castlery_report.json"

SITEMAP = "https://www.castlery.com/sg/sitemap.xml"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Seconds between requests. Deliberately unhurried: the whole catalog is ~1,200
# pages, which at this rate is about 20 minutes and puts no meaningful load on
# the origin.
DELAY = 1.0


# --- fetching --------------------------------------------------------------
#
# curl rather than urllib: this environment's Python has no CA bundle wired up,
# so urllib fails SSL verification on every request while curl verifies fine.
# Going through curl keeps certificate checking ON, which rolling our own
# unverified SSLContext would not.

def fetch(url: str, timeout: int = 30) -> str | None:
    result = subprocess.run(
        ["curl", "-s", "--compressed", "--max-time", str(timeout),
         "-A", UA, "-H", "Accept-Language: en-SG,en;q=0.9",
         "-w", "\n%{http_code}", url],
        capture_output=True, text=True,
    )
    body = result.stdout
    if not body:
        return None
    body, _, code = body.rpartition("\n")
    return body if code.strip() == "200" else None


def fetch_cached(url: str) -> str | None:
    """Fetch through an on-disk cache so re-runs cost nothing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-z0-9]+", "_", url.rsplit("/", 1)[-1].lower())[:120]
    path = CACHE / f"{key}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    html = fetch(url)
    time.sleep(DELAY)
    if html:
        path.write_text(html, encoding="utf-8")
    return html


def product_urls() -> list[str]:
    """SG product URLs, from Castlery's published sitemap."""
    xml = fetch(SITEMAP)
    if not xml:
        sys.exit("Could not fetch the sitemap; aborting rather than guessing URLs.")
    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
    return sorted({u for u in locs if "/sg/products/" in u})


# Role guesses from the URL slug, used ONLY to spread a sample across the five
# roles the app lays out. The authoritative role comes from the scraped
# `category` in the importer - this is a sampling heuristic, so a slug that
# guesses wrong costs nothing but a slightly uneven quota.
#
# Ordered, and first match wins: "chaise-sectional-sofa" is a sofa, not a
# chaise-as-accent-chair, so the sofa pattern has to be tested before the
# lounge-seating one.
_ROLE_PATTERNS: list[tuple[str, str]] = [
    ("rug", r"-rug|rug-"),
    ("floor_lamp", r"floor-lamp|arc-lamp|arched-floor"),
    ("sofa", r"sofa|sectional|loveseat|daybed"),
    ("coffee_table", r"coffee-table|side-table|nesting-table|nest-table"),
    ("accent_chair", r"armchair|accent-chair|lounge-chair|swivel-chair|"
                     r"occasional-chair|ottoman|recliner"),
]


def _slug_role(url: str) -> str | None:
    slug = url.rsplit("/", 1)[-1].lower()
    for role, pattern in _ROLE_PATTERNS:
        if re.search(pattern, slug):
            return role
    return None


def balanced_sample(urls: list[str], total: int) -> list[str]:
    """`total` URLs spread evenly across the five roles.

    A flat head-of-list slice would be alphabetical, and Castlery's catalog is
    dominated by sofas (395 of 1,217 slugs) while floor lamps number 9 - so
    the naive slice returns almost no lamps and the catalog loses a role.
    Each role gets an equal quota; a role with fewer URLs than its quota hands
    the remainder back to the others rather than leaving the sample short.
    """
    buckets: dict[str, list[str]] = {role: [] for role, _ in _ROLE_PATTERNS}
    for u in urls:
        role = _slug_role(u)
        if role:
            buckets[role].append(u)

    # Deterministic but not alphabetical: an alphabetical slice would collect
    # every colour variant of the same early-alphabet range ("Adams", "Agnes")
    # and call it a catalog. Seeded, so a re-run picks the same products.
    rng = random.Random(20260829)
    for b in buckets.values():
        rng.shuffle(b)

    # Smallest bucket first: a role that cannot fill its quota releases the
    # shortfall, and the quota is recomputed over the roles still to come.
    picked: list[str] = []
    order = sorted(buckets, key=lambda r: len(buckets[r]))
    for i, role in enumerate(order):
        remaining_roles = len(order) - i
        quota = max(1, (total - len(picked)) // remaining_roles)
        picked.extend(buckets[role][:quota])
    return picked[:total]


# --- parsing ---------------------------------------------------------------

def product_ld(html: str) -> dict | None:
    """The schema.org Product block, which is the vendor's own product feed."""
    for block in re.findall(
        r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "Product":
            return data
    return None


# Castlery writes dimensions as one display string. Four shapes occur, and the
# difference between them matters because the solver treats dimensions as
# ground truth:
#
#   W152 x D88 x H82cm              - the ordinary case
#   W264 x D88/167 x H82cm          - a chaise: depth is min/max, not a range
#                                     to average. The footprint is the MAX.
#   Dia. 8.4 x H24cm                - round. Diameter is both width and depth.
#   Sofa: W322 ... ; Ottoman: W96   - a set. The first segment is the primary
#                                     piece; an ottoman's footprint is not the
#                                     sofa's and averaging them describes
#                                     neither.
_AXIS = re.compile(r"\b([WDLH])\s*\.?\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?", re.I)
# Diameter is written two ways: "Dia. 98" and the phi symbol "Φ98".
_DIA = re.compile(r"(?:\bDia\.?|[ΦφØø⌀])\s*(\d+(?:\.\d+)?)", re.I)

# A flat piece whose height Castlery omits entirely - "W153 x L244" on a rug.
# 1.5cm is the pile height Castlery publishes for the rugs that DO state one,
# so it is a category default rather than a guess pulled from nowhere, and a
# rug's height is not load-bearing for the solver: rugs do not collide.
_FLAT_HEIGHT_CM = 1.5


def parse_dimensions(value: str, name: str = "") -> tuple[dict | None, str | None]:
    """Return (dimensions_cm, note). Note records any interpretation applied.

    `name` is the product title, consulted only to tell a genuinely flat piece
    (a rug) from a listing with a missing height.
    """
    if not value:
        return None, None
    note = None
    text = value
    if ";" in text:
        # A multi-part set: keep the primary (first) piece and say so.
        head, _, rest = text.partition(";")
        label = head.split(":")[0].strip() if ":" in head else "first piece"
        note = f"primary piece only ({label}); full listing: {value}"
        text = head
    text = text.split(":", 1)[-1] if ":" in text else text

    dims: dict[str, float] = {}
    for axis, first, second in _AXIS.findall(text):
        axis = axis.upper()
        # "D88/167" on a chaise is min/max depth. The larger is the footprint
        # the piece actually occupies, and understating it would let the solver
        # place furniture into space the sofa fills.
        val = max(float(first), float(second)) if second else float(first)
        if axis == "L":  # rugs use Length where other categories use Depth
            axis = "D"
        dims.setdefault(axis, val)

    if "W" not in dims or "D" not in dims:
        dia = _DIA.search(text)
        if dia:
            d = float(dia.group(1))
            dims.setdefault("W", d)
            dims.setdefault("D", d)
            note = note or "round item: diameter used for width and depth"

    # A rug given as "W153 x L244" states no height at all. Only fill one in
    # for a piece that is genuinely flat - anything else with a missing
    # height is a listing this scraper should not pretend to understand.
    if "H" not in dims and "W" in dims and "D" in dims:
        if re.search(r"\brug\b|\bmat\b", f"{value} {name}", re.I):
            dims["H"] = _FLAT_HEIGHT_CM
            note = note or f"height not published; {_FLAT_HEIGHT_CM}cm pile assumed"

    if "W" not in dims or "D" not in dims or "H" not in dims:
        return None, note
    return (
        {"width_cm": dims["W"], "depth_cm": dims["D"], "height_cm": dims["H"]},
        note,
    )


def clean_reviews(ld: dict) -> dict:
    rating = ld.get("aggregateRating") or {}
    out = []
    for r in (ld.get("review") or [])[:10]:
        body = (r.get("reviewBody") or "").strip()
        if not body:
            continue
        stars = (r.get("reviewRating") or {}).get("ratingValue")
        out.append({
            "author": ((r.get("author") or {}).get("name") or "").strip() or None,
            "rating": float(stars) if stars not in (None, "") else None,
            "text": body,
            # Castlery stamps a time onto the date; only the day is meaningful.
            "date": (r.get("datePublished") or "").split(" ")[0] or None,
        })
    return {
        "rating": rating.get("ratingValue"),
        "review_count": rating.get("reviewCount"),
        "reviews": out,
    }


def images(ld: dict) -> list[str]:
    """Hero image first, then one per colour/size variant, de-duplicated."""
    urls: list[str] = []
    hero = ld.get("image")
    if isinstance(hero, str):
        urls.append(hero)
    elif isinstance(hero, list):
        urls.extend(u for u in hero if isinstance(u, str))
    for v in ld.get("hasVariant") or []:
        u = v.get("image")
        if isinstance(u, str):
            urls.append(u)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def build(ld: dict, url: str) -> tuple[dict, list[str]]:
    """Map one JSON-LD Product onto the products.json schema."""
    missing: list[str] = []
    props = {
        p.get("name"): p.get("value")
        for p in (ld.get("additionalProperty") or [])
        if isinstance(p, dict)
    }
    dims, dim_note = parse_dimensions(props.get("Dimension", ""), ld.get("name") or "")
    if not dims:
        missing.append("dimensions")

    offers = ld.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price = offers.get("price")
    if price in (None, ""):
        missing.append("price")

    # `Frame` is the only materials signal Castlery exposes in the feed; it is
    # a free-text phrase ("Solid oak, plywood"), so it is split rather than
    # matched against a vocabulary, and left empty when absent instead of
    # being guessed from the description.
    frame = props.get("Frame") or ""
    materials = [m.strip().lower() for m in re.split(r"[,/]| and ", frame) if m.strip()]

    item = {
        "product_id": ld.get("sku"),
        "name": ld.get("name"),
        "brand": (ld.get("brand") or {}).get("name") or "Castlery",
        "category": ld.get("category"),
        "description": (ld.get("description") or "").strip(),
        "price": float(price) if price not in (None, "") else None,
        "currency": offers.get("priceCurrency") or "SGD",
        "availability": offers.get("availability"),
        "dimensions": dims,
        # Mirrors products.json's `dimension_sources`: says where each number
        # came from, so a later importer can tell a measured footprint from an
        # interpreted one instead of trusting all three equally.
        "dimension_source": props.get("Dimension"),
        "dimension_note": dim_note,
        "appearance": {
            "colours": [],          # not in the feed; inferred by the importer
            "materials": materials,
            "finish": None,
            "pattern": None,
            "texture": None,
        },
        "seating": {
            "depth_cm": _cm(props.get("Seating depth")),
            "width_cm": _cm(props.get("Seatable width")),
            "height_cm": _cm(props.get("Seating height")),
        },
        "cover_type": props.get("Cover type"),
        "variant_count": len(ld.get("hasVariant") or []),
        "images": images(ld),
        "reviews": clean_reviews(ld),
        "merchant": "Castlery",
        "product_url": url,
    }
    if not item["product_id"]:
        missing.append("product_id")
    if not item["name"]:
        missing.append("name")
    if not item["images"]:
        missing.append("images")
    return item, missing


def _cm(value: str | None) -> float | None:
    if not value:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(m.group(1)) if m else None


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Castlery SG into products.json shape.")
    ap.add_argument("--limit", type=int, help="stop after N products (for a quick sample)")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="scrape N products spread evenly across the five roles, "
                         "instead of the whole catalog")
    ap.add_argument("--refresh", action="store_true", help="ignore the page cache")
    args = ap.parse_args()

    if args.refresh and CACHE.exists():
        for f in CACHE.glob("*.html"):
            f.unlink()

    urls = product_urls()
    total = len(urls)
    if args.sample:
        urls = balanced_sample(urls, args.sample)
        print(f"{total} product URLs in the sitemap -> "
              f"{len(urls)} sampled across {len(_ROLE_PATTERNS)} roles")
    else:
        if args.limit:
            urls = urls[: args.limit]
        print(f"{len(urls)} product URLs from the sitemap")

    products: list[dict] = []
    failed: list[dict] = []
    incomplete: list[dict] = []
    categories: Counter = Counter()

    for i, url in enumerate(urls, 1):
        html = fetch_cached(url)
        if not html:
            failed.append({"url": url, "issue": "fetch failed"})
        else:
            ld = product_ld(html)
            if not ld:
                failed.append({"url": url, "issue": "no Product JSON-LD on page"})
            else:
                item, missing = build(ld, url)
                products.append(item)
                categories[item["category"]] += 1
                if missing:
                    incomplete.append({"product_id": item["product_id"],
                                       "name": item["name"], "missing": missing})
        if i % 25 == 0 or i == len(urls):
            print(f"  {i}/{len(urls)}  kept={len(products)} failed={len(failed)}")

    OUT_PRODUCTS.write_text(json.dumps(products, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "merchant": "Castlery",
        "source": "schema.org Product JSON-LD on Castlery SG product pages",
        "urls_seen": len(urls),
        "products": len(products),
        "failed": failed,
        "incomplete": incomplete,
        "categories": dict(categories.most_common()),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(products)} products to {OUT_PRODUCTS.name} "
          f"({len(failed)} failed, {len(incomplete)} with missing fields)")


if __name__ == "__main__":
    main()
