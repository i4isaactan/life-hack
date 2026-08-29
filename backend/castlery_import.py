"""Build catalog items from the scraped Castlery Singapore listings.

WHAT THIS IS. The sibling of `ikea_import.py`. `castlery_products.json` is a
scrape of Castlery SG product pages (see `scrapers/castlery_scrape.py`); this
module turns the usable subset into `CatalogItem`s so the catalog spans two
merchants instead of one.

WHY A SECOND MERCHANT. Two gaps in the IKEA scrape, both noted in seed_data:

  1. NO TV UNITS, MEDIA CONSOLES OR SIDEBOARDS. Castlery publishes all three,
     which is what makes restoring a media role possible at all.

  2. ONE PRICE BAND. IKEA SG tops out near S$1,699, so "premium" had no
     meaning in the catalog. Castlery sits above it, giving the budget logic
     an actual range to work across.

Prices are SGD from a Singapore storefront, so - as with IKEA - they are
carried as SGD and nothing is converted.

WHERE THIS DIFFERS FROM THE IKEA IMPORTER. The failure modes are not the same,
so the cleanup is not either:

  - CASTLERY'S DIMENSIONS ARE TRUSTWORTHY. They are published as one display
    string per product and parsed at scrape time, with anything interpreted
    (a chaise's min/max depth, a set's primary piece) recorded in
    `dimension_note`. There is no cord-length-as-width problem here.

  - COLOUR IS IN THE TITLE, NOT A FIELD. The feed has no colour list, so the
    colour is parsed from the variant name.

  - VARIANTS ARE ALREADY COLLAPSED. Castlery's JSON-LD gives one canonical
    product per URL with its variants nested, so the aggressive per-family
    capping the IKEA scrape needed is a light touch here.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .models import CatalogItem, Dimensions, Role

PRODUCTS_JSON = Path(__file__).resolve().parent.parent / "castlery_products.json"

MAX_VARIANTS_PER_FAMILY = 3
CURRENCY = "SGD"


# --- role mapping ----------------------------------------------------------
#
# Scraped category -> Role, as an explicit allowlist for the same reason the
# IKEA importer uses one: "Outdoor Side Tables" contains "Side Table" but does
# not belong in a living room, and "Sofa Covers" contains "Sofa" but is a
# spare textile. Castlery's category names are granular, so this is long
# rather than clever.

CATEGORY_ROLE: dict[str, Role] = {
    # Sofas.
    "2 Seater Sofas": Role.SOFA,
    "3 Seater Sofas": Role.SOFA,
    "Extended 3 Seater Sofas": Role.SOFA,
    "Chaise Sectional Sofas": Role.SOFA,
    "L-Shape Sectional Sofas": Role.SOFA,
    "U-Shape Sectional Sofas": Role.SOFA,
    "Sectional Sofas": Role.SOFA,
    "Modular Corner Sofas": Role.SOFA,
    "Modular Side Corner Sofas": Role.SOFA,
    "Sofa Beds": Role.SOFA,
    "Loveseats": Role.SOFA,
    # Lounge seating.
    "Armchairs": Role.ACCENT_CHAIR,
    "Accent Armchairs": Role.ACCENT_CHAIR,
    "Accent Chairs": Role.ACCENT_CHAIR,
    "Lounge Chairs": Role.ACCENT_CHAIR,
    "Swivel Chairs": Role.ACCENT_CHAIR,
    "Multi-Room Chairs": Role.ACCENT_CHAIR,
    "Recliners": Role.ACCENT_CHAIR,
    "Ottomans": Role.ACCENT_CHAIR,
    # Low tables.
    "Coffee Tables": Role.COFFEE_TABLE,
    "Round Coffee Tables": Role.COFFEE_TABLE,
    "Side Tables": Role.COFFEE_TABLE,
    "Nested Side Tables": Role.COFFEE_TABLE,
    "Nested Coffee Tables": Role.COFFEE_TABLE,
    "Modular Side Table": Role.COFFEE_TABLE,
    "C Side Tables": Role.COFFEE_TABLE,
    # Rugs.
    "Area Rugs": Role.RUG,
    "Rugs": Role.RUG,
    # Lighting.
    "Floor Lamps": Role.FLOOR_LAMP,
    "Arc Floor Lamps": Role.FLOOR_LAMP,
}

# Categories that look like furniture but are components, spares or outdoor
# pieces. Listed explicitly so a category the scrape adds later is *ignored*
# rather than silently mapped by a keyword match.
_EXCLUDED_MARKERS = (
    "outdoor",
    "cover",
    "sofa covers",
    "replacement",
)

# Name fragments marking a listing as a part rather than a piece of furniture.
_COMPONENT_MARKERS = (
    "cover",
    "replacement",
    "swatch",
    "sample",
    "armrest",
    "add-on",
    "extension piece",
    "spare",
)


# --- appearance ------------------------------------------------------------
#
# Same job as the IKEA importer's table - the floor plan draws every piece as a
# filled rectangle, so each item needs a hex swatch. Castlery's palette is
# named differently (its own finish names: "Oyster", "Sand", "Pebble"), so the
# table carries those alongside the plain colour words.

_COLOUR_HEX: dict[str, str] = {
    "white": "#EDEBE6", "chalk": "#E8E4DA", "ivory": "#EFE9DC",
    "oyster": "#DED6C6", "cream": "#E9E0CD", "sand": "#D8C9AE",
    "beige": "#D6C9B2", "taupe": "#B8A894", "pebble": "#C9C2B6",
    "greige": "#C3BBA9", "mushroom": "#B5A897",
    "grey": "#9BA0A2", "gray": "#9BA0A2", "light grey": "#C6CACB",
    "dark grey": "#5E6366", "charcoal": "#4A4E52", "slate": "#6B7378",
    "graphite": "#45494D", "ash": "#B0B4B5",
    "black": "#33363A", "onyx": "#2E3134",
    "brown": "#7A5A3E", "chocolate": "#5B4130", "tan": "#B08762",
    "pecan": "#8C5E3C", "cocoa": "#5C4033", "dune": "#C8B49A",
    "moss": "#6A7350", "clay": "#B07C64", "stone": "#C2BCB2",
    "linen": "#DED7C7", "wheat": "#D6C39A", "espresso": "#4A3728",
    "camel": "#B98F5E", "cognac": "#9A5F35", "caramel": "#B57A45",
    "chestnut": "#6E4630", "mocha": "#6B5445", "toffee": "#8B6440",
    "blue": "#4A6785", "navy": "#36455C", "light blue": "#9DB4CC",
    "teal": "#3E6F70", "denim": "#5A7391",
    "green": "#5C7359", "olive": "#6B6B45", "sage": "#A3B098",
    "forest": "#3F4F3D", "emerald": "#3D6B54",
    "yellow": "#D7B15C", "mustard": "#C09A3E", "gold": "#B08D57",
    "orange": "#C2703F", "rust": "#A65A3A", "terracotta": "#B4634A",
    "red": "#A6453C", "burgundy": "#6E2F34", "wine": "#6E2F34",
    "pink": "#D3A7A5", "blush": "#E2C4C1", "rose": "#C98F8C",
    "purple": "#6E5B7B", "lilac": "#A79AB5",
    "natural": "#CBB894", "oak": "#C8A97E", "light oak": "#D3B78D",
    "walnut": "#6B4A2F", "teak": "#A9743F", "ash wood": "#C9B393",
    "mango": "#A9784A", "acacia": "#9C7346", "birch": "#D8C29A",
    "whitewash": "#DCD3C4", "white wash": "#DCD3C4",
    "brass": "#B08D57", "gunmetal": "#5A5E62", "chrome": "#C7CACE",
    "silver": "#B9BCC0", "marble": "#E4E1DA", "travertine": "#D9CDBA",
    "boucle": "#E0DACD", "bouclé": "#E0DACD",
}

# Typo guard, as in the IKEA importer: a malformed hex here would otherwise
# only surface when Pydantic validated an item.
_COLOUR_HEX = {
    k: v for k, v in _COLOUR_HEX.items()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", v)
}

_FALLBACK_HEX = "#B0A79B"


def _colour_from_name(name: str, image_url: str = "") -> tuple[str, str]:
    """(colour name, hex) parsed from the product title, then the image name.

    Castlery's feed has no colour field, so the title is the only source. It
    reads "Auburn Performance Bouclé 3 Seater Sofa, Ivory" or "... - Sand", so
    the tail after the last comma or dash is checked first, then the whole
    title. Longest match wins so "light grey" is not read as "grey".

    Matched on WORD BOUNDARIES, not as substrings: "Hugg Nesting Coffee Table"
    contains the letters of "tan" inside "Nesting", and a naive `in` test
    paints that table brown.
    """
    low = name.lower()
    tail = re.split(r"[,-]", low)[-1].strip() if re.search(r"[,-]", low) else ""

    # Castlery usually leaves the colour OUT of the title and puts it in the
    # image filename instead - "Adams-Armchair-Pearl-Beige-Silver-Front.png",
    # "Abanna-Area-Rug-5_x_8-Oyster.png". That filename is the vendor's own
    # variant label, so it is a real signal rather than a guess; without it
    # nearly every Castlery item falls back to the neutral swatch and the
    # floor plan renders the whole catalog the same beige.
    slug = ""
    if image_url:
        slug = re.sub(r"[-_]+", " ", image_url.rsplit("/", 1)[-1].rsplit(".", 1)[0]).lower()

    for haystack in (tail, low, slug):
        if not haystack:
            continue
        hits = []
        for c in _COLOUR_HEX:
            m = re.search(rf"\b{re.escape(c)}\b", haystack)
            if m:
                hits.append((m.start(), -len(c), c))
        if hits:
            # EARLIEST match wins, longest breaking the tie. The upholstery
            # leads the variant label and the frame finish trails it, so
            # "Adams-Armchair-Pearl-Beige-Silver" is a beige chair with silver
            # legs - taking the longest or last match paints it silver.
            best = min(hits)[2]
            return best, _COLOUR_HEX[best]
    return "mixed", _FALLBACK_HEX


# --- style -----------------------------------------------------------------
#
# Same discipline as the IKEA importer: a tag that fires on most of the
# catalog cannot discriminate, so each rule demands a specific signal and at
# most three survive. Castlery's own vocabulary differs from IKEA's - it sells
# a mid-century and a "coastal" look explicitly, and its material words appear
# in product titles rather than a bill of materials.

_SIGNAL_MATERIALS = {
    "rattan": ("Japandi", "Rustic"),
    "cane": ("Japandi",),
    "jute": ("Coastal", "Rustic"),
    "teak": ("Mid-Century Modern",),
    "walnut": ("Mid-Century Modern",),
    "oak": ("Scandinavian",),
    "marble": ("Luxe",),
    "travertine": ("Luxe",),
    "velvet": ("Luxe",),
    "boucle": ("Minimalist",),
    "bouclé": ("Minimalist",),
}

_QUIET_COLOURS = (
    "white", "chalk", "ivory", "oyster", "cream", "sand", "beige",
    "greige", "pebble", "natural", "oak", "light grey", "taupe",
)
_LOUD_COLOURS = (
    "red", "burgundy", "wine", "emerald", "mustard", "orange", "rust",
    "terracotta", "purple", "teal", "gold",
)


def _style_tags(name: str, colour: str, materials: list[str], image_url: str = "") -> list[str]:
    tags: list[str] = []
    # Castlery names the fabric and wood in the title, the materials list, or
    # the image's variant label - all three are searched, since "Velvet" or
    # "Walnut" frequently appears only in the last of them.
    slug = ""
    if image_url:
        slug = re.sub(r"[-_]+", " ", image_url.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    hay = f"{name} {' '.join(materials)} {slug}".lower()

    for needle, add in _SIGNAL_MATERIALS.items():
        if re.search(rf"\b{re.escape(needle)}\b", hay):
            tags.extend(add)

    # Leather counts only as a surface. Castlery names it in the title when the
    # piece is upholstered in it ("Pascal Leather Sling Armchair"), so unlike
    # the IKEA case there is no bill-of-materials trim to filter out.
    if "leather" in name.lower():
        tags.append("Mid-Century Modern")

    # Metal with no wood reads industrial; metal legs under wood do not.
    if any(w in hay for w in ("steel", "metal", "iron")) and not any(
        w in hay for w in ("wood", "oak", "walnut", "teak", "ash", "mango", "acacia")
    ):
        tags.append("Industrial")

    if any(c == colour or c in colour for c in _QUIET_COLOURS):
        tags.append("Minimalist")
    if any(c == colour or c in colour for c in _LOUD_COLOURS):
        tags.append("Maximalist")

    seen: set[str] = set()
    ordered = [t for t in tags if not (t in seen or seen.add(t))]

    # Castlery's house style is contemporary, used only as a floor so it stays
    # informative - the same reasoning as IKEA's "Scandinavian" fallback.
    if not ordered:
        ordered = ["Contemporary"]
    return ordered[:3]


# --- filtering -------------------------------------------------------------


def _is_component(name: str, category: str) -> bool:
    low = name.lower()
    cat = (category or "").lower()
    if any(m in cat for m in _EXCLUDED_MARKERS):
        return True
    return any(m in low for m in _COMPONENT_MARKERS)


def _family(name: str) -> str:
    """The product series - Castlery's model name, which is the first word.

    Castlery titles read "Adams Fabric 3 Seater Sofa" or "Hugg Nesting
    Rectangular Coffee Table": the range name leads and is capitalised like
    ordinary prose, so unlike IKEA's all-caps model names it cannot be
    detected by case. The first token is the range.
    """
    tokens = re.findall(r"[A-Za-z0-9']+", name)
    return tokens[0].title() if tokens else name


def _dimensions_for(item: dict, role: Role) -> Dimensions | None:
    """Validated cm dimensions, or None if this listing cannot be trusted.

    Returning None drops the item, for the same reason as in the IKEA
    importer: the solver treats these numbers as ground truth, so a guess here
    becomes a confidently wrong layout.
    """
    raw = item.get("dimensions") or {}

    def num(key: str) -> float | None:
        v = raw.get(key)
        return float(v) if isinstance(v, (int, float)) and v > 0 else None

    w, d, h = num("width_cm"), num("depth_cm"), num("height_cm")
    if w is None or d is None or h is None:
        return None

    # Rugs: lay the long edge out as width, matching how the solver and the
    # IKEA importer both describe them.
    if role is Role.RUG:
        w, d = max(w, d), min(w, d)

    # Plausibility band per role, mirroring the IKEA importer. A "sofa" 15cm
    # wide is a parsing artefact, not a product. Castlery's sofas run larger
    # than IKEA's, so the sofa band is wider at the top.
    bands = {
        Role.SOFA: ((120, 400), (60, 250), (40, 120)),
        Role.ACCENT_CHAIR: ((40, 140), (40, 140), (30, 130)),
        # Height capped at 60cm for the same reason as IKEA: a table serving a
        # sofa sits at or below seat height, and Castlery's "Side Tables"
        # category also holds taller end tables.
        Role.COFFEE_TABLE: ((30, 200), (30, 140), (20, 60)),
        Role.RUG: ((60, 400), (60, 400), (0.1, 10)),
        Role.FLOOR_LAMP: ((10, 90), (10, 90), (80, 220)),
    }
    (wlo, whi), (dlo, dhi), (hlo, hhi) = bands[role]
    if not (wlo <= w <= whi and dlo <= d <= dhi and hlo <= h <= hhi):
        return None

    return Dimensions(width_cm=round(w, 1), depth_cm=round(d, 1), height_cm=round(h, 1))


def _load_raw() -> list[dict]:
    if not PRODUCTS_JSON.exists():
        raise FileNotFoundError(
            f"{PRODUCTS_JSON} not found - run `python -m scrapers.castlery_scrape` first"
        )
    with PRODUCTS_JSON.open() as fh:
        return json.load(fh)


def build_items(
    max_variants: int = MAX_VARIANTS_PER_FAMILY,
    source: list[dict] | None = None,
) -> list[CatalogItem]:
    """The Castlery scrape, reduced to a usable catalog."""
    raw = source if source is not None else _load_raw()

    staged: list[tuple[Role, str, CatalogItem]] = []
    for item in raw:
        category = item.get("category") or ""
        role = CATEGORY_ROLE.get(category)
        if role is None:
            continue

        name = (item.get("name") or "").strip()
        if not name or _is_component(name, category):
            continue

        price = item.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            continue

        dims = _dimensions_for(item, role)
        if dims is None:
            continue

        images = item.get("images") or []
        if not images:
            continue

        pid = item.get("product_id")
        url = item.get("product_url")
        if not pid or not url:
            continue

        appearance = item.get("appearance") or {}
        materials = [m for m in (appearance.get("materials") or []) if m]
        colour_name, swatch = _colour_from_name(name, images[0])

        # Castlery's descriptions do NOT repeat the product name the way
        # IKEA's do, so there is nothing to strip - the prose is already the
        # signal. Guarded anyway, since a re-scrape could change that.
        desc = (item.get("description") or "").strip()
        if desc.startswith(name):
            desc = desc[len(name):].lstrip(" .-")

        # The visible upholstery or finish. Castlery names it in the title
        # tail, and `cover_type` from the feed is about removability, not
        # appearance, so it is not used here.
        finish = ""
        if role in (Role.SOFA, Role.ACCENT_CHAIR) and colour_name != "mixed":
            finish = colour_name.title()

        # Seat count is in the category or the title ("3 Seater Sofas"), which
        # is the only place Castlery states it.
        seats = None
        m = re.search(r"(\d+)\s*[- ]?seater", f"{category} {name}", re.I)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 8:
                seats = n

        availability = (item.get("availability") or "").lower()
        in_stock = "outofstock" not in availability

        staged.append((
            role,
            _family(name),
            CatalogItem(
                # Prefixed like the IKEA ids, so the two merchants cannot
                # collide in the index even if a SKU repeats.
                id=f"castlery-{re.sub(r'[^A-Za-z0-9]+', '', str(pid))}",
                merchant="Castlery",
                title=name,
                role=role,
                price_cents=int(round(price * 100)),
                currency=CURRENCY,
                dimensions=dims,
                materials=materials[:6],
                primary_color=colour_name,
                swatch=swatch,
                style_tags=_style_tags(name, colour_name, materials, images[0]),
                image_url=images[0],
                checkout_url=url,
                in_stock=in_stock,
                description=desc,
                finish=finish,
                seating_capacity=seats,
                series=_family(name),
            ),
        ))

    grouped: dict[tuple[Role, str], list[CatalogItem]] = defaultdict(list)
    for role, family, ci in staged:
        grouped[(role, family)].append(ci)

    out: list[CatalogItem] = []
    for (_role, _family_name), variants in grouped.items():
        variants.sort(key=lambda c: (not c.in_stock, c.price_cents))
        out.extend(variants[:max_variants])

    out.sort(key=lambda c: (c.role.value, c.price_cents))
    return out


if __name__ == "__main__":
    import collections

    items = build_items()
    by_role = collections.Counter(i.role.value for i in items)
    raw_n = len(_load_raw())
    print(f"{raw_n} scraped listings -> {len(items)} catalog items\n")
    for role, n in sorted(by_role.items()):
        sub = [i for i in items if i.role.value == role]
        ws = [i.dimensions.width_cm for i in sub]
        ps = [i.price_cents / 100 for i in sub]
        fams = len({i.series for i in sub})
        print(
            f"  {role:<14} {n:>3} items  {fams:>2} families  "
            f"{min(ws):>5.0f}-{max(ws):<5.0f}cm  "
            f"S${min(ps):>7.2f}-{max(ps):<8.2f}"
        )
