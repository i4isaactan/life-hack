"""Build the catalog from scraped IKEA Singapore listings.

WHAT THIS IS. `products.json` is a scrape of 1,579 IKEA SG listings. This
module turns the subset that is usable into `CatalogItem`s. It is run at import
time by seed_data.py, not as a build step, so the catalog always reflects the
current scrape.

WHY THE RAW SCRAPE IS NOT USED DIRECTLY. Three problems, each handled below:

  1. IT IS MOSTLY IRRELEVANT. 146 categories, most of which are towels,
     napkins, plant pots and baby bedding. Only six map onto a living room.

  2. IT IS WILDLY UNBALANCED. 701 sofas against 8 floor lamps - but those 701
     sofas are only 28 actual products, the rest being colour and fabric
     variants. Left alone, retrieval returns nine shades of the same UPPAKRA
     and calls it choice.

  3. SOME DIMENSIONS ARE WRONG IN A WAY THE SOLVER CANNOT SURVIVE. The scraper
     fell back to whatever measurement the product page listed first, which for
     lighting is the CORD LENGTH: a 380cm-wide floor lamp is a 3.8m cable. The
     solver treats dimensions as ground truth, so this is not cosmetic - it
     either blocks out half the room or gets rejected as unfittable.

WHAT IS DELIBERATELY DROPPED. Modular sofa *components* (single-seat sections,
armrests, corner units) are real products but not answers to "design my living
room" - you cannot furnish a room with an armrest. Covers, spare parts and
outdoor furniture go too.

Prices are SGD, and are carried as SGD rather than silently relabelled: the
scrape has no exchange rate in it and inventing one would misstate every price
in the app.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .models import CatalogItem, Dimensions, Role

PRODUCTS_JSON = Path(__file__).resolve().parent.parent / "products.json"

# At most this many colour/fabric variants of the same product family. Enough
# that the aesthetic matcher has a palette to choose from, few enough that one
# popular sofa cannot crowd out the rest of the catalog.
MAX_VARIANTS_PER_FAMILY = 3

CURRENCY = "SGD"


# --- role mapping ----------------------------------------------------------
#
# Scraped category -> Role. Explicit rather than keyword-matched: "Bedside
# tables & cabinets" contains "table" but is not a coffee table, and
# "Cabinet lighting" contains "cabinet" but is not furniture at all. An
# allowlist is the only mapping that stays correct as the scrape grows.

CATEGORY_ROLE: dict[str, Role] = {
    # Sofas. Modular sofas are included, but components are filtered out below.
    "Two seater sofas": Role.SOFA,
    "Three seater sofas": Role.SOFA,
    "Fabric sofas with chaise longues": Role.SOFA,
    "Corner sofas": Role.SOFA,
    "Modular sofas": Role.SOFA,
    "Recliner sofas": Role.SOFA,
    "Chaise longues": Role.SOFA,
    # Armchairs and lounge seating.
    "Fabric armchairs": Role.ACCENT_CHAIR,
    "Leather armchairs": Role.ACCENT_CHAIR,
    "Rattan armchairs": Role.ACCENT_CHAIR,
    "Lounge chairs": Role.ACCENT_CHAIR,
    "Upholstered chairs": Role.ACCENT_CHAIR,
    # Low tables. Side and nesting tables sit alongside a sofa and fill the
    # same slot in a layout, so they are offered for the role.
    "Coffee tables": Role.COFFEE_TABLE,
    "Side tables": Role.COFFEE_TABLE,
    "Nesting tables": Role.COFFEE_TABLE,
    "Rugs": Role.RUG,
    "Large & medium rugs": Role.RUG,
    "Floor, reading & LED lamps": Role.FLOOR_LAMP,
}

# Name fragments that mark a listing as a component, accessory or spare rather
# than a piece of furniture. Matched case-insensitively against the name.
_COMPONENT_MARKERS = (
    "section",
    "module",
    "cover",          # "…sofa cover", a spare textile
    "armrest",
    "backrest",
    "frame only",
    "leg ",
    "legs",
    "add-on",
    "extension",
    "spare",
)

# Lighting listings whose footprint came from a cord/cable measurement. The
# scraper recorded which field it used, so this is detectable rather than
# guessed: anything sourced from "length" is not a footprint.
_BAD_FOOTPRINT_SOURCES = {"length", "cord length", "cable length"}

# A floor lamp's real footprint is its base, which IKEA rarely publishes. These
# are honest defaults for the category rather than measurements - a lamp base is
# 25-50cm across, and the solver only needs it to be plausibly small.
_LAMP_FALLBACK_FOOTPRINT_CM = 40.0


# --- appearance ------------------------------------------------------------
#
# The floor plan draws each piece as a filled rectangle, so every item needs a
# hex swatch. The scrape gives colour *names*, so they are mapped to
# representative hexes. Approximate by nature: "beige" covers a wide range and
# this picks one point in it.

_COLOUR_HEX: dict[str, str] = {
    "white": "#EDEBE6", "off-white": "#E8E4DA", "beige": "#D6C9B2",
    "grey-beige": "#C3BBA9", "greige": "#C3BBA9",
    "grey": "#9BA0A2", "gray": "#9BA0A2", "dark grey": "#5E6366",
    "light grey": "#C6CACB", "anthracite": "#4A4E52",
    "black": "#33363A", "black-blue": "#2E3440",
    "brown": "#7A5A3E", "dark brown": "#5B4130", "light brown": "#A9825E",
    "red-brown": "#8A4B3A", "brown-red": "#8A4B3A",
    "blue": "#4A6785", "dark blue": "#36455C", "light blue": "#9DB4CC",
    "green": "#5C7359", "dark green": "#3F4F3D", "light green": "#A9BBA2",
    "turquoise": "#4E8C8A",
    "yellow": "#D7B15C", "dark yellow": "#C09A3E", "gold-colour": "#B08D57",
    "orange": "#C2703F", "red": "#A6453C", "dark red": "#7E3630",
    "pink": "#D3A7A5", "light pink": "#E2C4C1", "lilac": "#A79AB5",
    "purple": "#6E5B7B", "violet": "#6E5B7B",
    "natural": "#CBB894", "birch": "#D8C29A", "oak": "#C8A97E",
    "pine": "#C08E4E", "bamboo": "#C9AA6E", "rattan": "#C4A067",
    "walnut": "#6B4A2F", "stained": "#8B6A4F",
    "silver-colour": "#B9BCC0", "nickel-plated": "#B9BCC0",
    "chrome-plated": "#C7CACE", "brass-colour": "#B08D57",
    "multicolour": "#B0A79B",
}

# Typo guard: the table above is hand-written, so a malformed hex would only
# surface when Pydantic validated an item. Catch it at import instead.
_COLOUR_HEX = {
    k: v for k, v in _COLOUR_HEX.items()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", v)
}

_FALLBACK_HEX = "#B0A79B"


def _swatch_for(colours: list[str]) -> tuple[str, str]:
    """(primary colour name, hex) for a listing's colour list."""
    for c in colours:
        key = (c or "").strip().lower()
        if key in _COLOUR_HEX:
            return key, _COLOUR_HEX[key]
    # Try a partial match before giving up: "dark grey-green" contains "grey".
    for c in colours:
        key = (c or "").strip().lower()
        for name, hexval in _COLOUR_HEX.items():
            if name in key:
                return key, hexval
    return (colours[0].lower() if colours else "mixed"), _FALLBACK_HEX


# Style tags the app's aesthetics vocabulary understands, inferred from the
# listing. The scrape's own `style` field is empty on every one of the 1,579
# listings, so inference is the only source of style signal.
#
# THE FAILURE MODE THIS AVOIDS. An earlier version fired a tag on any weak
# signal, which put six tags on 47 items and "Scandinavian" on 87% of the
# catalog. A tag carried by seven items in eight cannot discriminate between
# them - it is indistinguishable from no tag at all, while still diluting the
# embedding. So each rule below demands a signal specific enough to be worth
# acting on, and at most three tags survive.

# Materials that genuinely indicate a look, as opposed to the construction
# filler ("polyurethane", "fibreboard", "plywood") that appears in nearly
# every upholstered piece and says nothing about appearance.
_SIGNAL_MATERIALS = {
    "rattan": ("Japandi", "Rustic"),
    "bamboo": ("Japandi", "Rustic"),
    "jute": ("Japandi", "Rustic"),
    "cane": ("Japandi",),
    "solid oak": ("Scandinavian",),
    "solid birch": ("Scandinavian",),
    "solid pine": ("Rustic",),
    "oak veneer": ("Scandinavian",),
    "birch veneer": ("Scandinavian",),
}

# Muted, pale colours read Scandinavian/Minimalist; saturated ones do not.
_QUIET_COLOURS = (
    "white", "off-white", "beige", "grey-beige", "greige", "light grey",
    "natural", "birch", "oak",
)
_LOUD_COLOURS = (
    "red", "yellow", "green", "blue", "pink", "orange", "purple", "violet",
    "turquoise", "multicolour", "dark red", "gold-colour",
)


def _style_tags(item: dict, colour: str, materials: list[str]) -> list[str]:
    tags: list[str] = []
    mats = " ".join(materials).lower()
    name = (item.get("name") or "").lower()

    for needle, add in _SIGNAL_MATERIALS.items():
        if needle in mats:
            tags.extend(add)

    # Leather is only a style signal when it is the SURFACE. IKEA's material
    # lists are a full bill of materials, so a fabric recliner lists leather
    # for a trim detail; the giveaway is the fabric name in the title, where
    # IKEA's coated-leather ranges are named explicitly.
    if "leather" in mats and any(k in name for k in ("bomstad", "grann", "glose", "kimstad")):
        tags.append("Mid-Century Modern")

    # Metal with no wood at all reads industrial; metal legs under a wooden
    # frame do not.
    if any(w in mats for w in ("steel", "aluminium")) and not any(
        w in mats for w in ("wood", "oak", "birch", "beech", "pine", "veneer", "rattan")
    ):
        tags.append("Industrial")

    if any(c == colour or c in colour for c in _QUIET_COLOURS):
        tags.append("Minimalist")
    if any(c == colour or c in colour for c in _LOUD_COLOURS):
        tags.append("Maximalist")

    # Deduplicate while keeping the order the rules fired in, so the most
    # specific signal (material, then surface, then colour) leads.
    seen: set[str] = set()
    ordered = [t for t in tags if not (t in seen or seen.add(t))]

    # IKEA's house style is the floor, not the ceiling: applied only when
    # nothing more specific fired, so it stays informative.
    if not ordered:
        ordered = ["Scandinavian"]
    return ordered[:3]


# --- filtering -------------------------------------------------------------


def _is_component(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _COMPONENT_MARKERS)


def _family(name: str) -> str:
    """The product series: IKEA's model name, e.g. "SODERHAMN", "IKEA PS 2026".

    Takes every leading upper-case/numeric token rather than just the first
    word, because several real series are multi-word ("IKEA PS 2026",
    "STOCKHOLM 2025"). Stopping at the first word would collapse every PS
    product into a bogus "IKEA" family and split the rest incorrectly.

    Used twice: to cap colour variants per product, and as the "matching set"
    signal for bundles.
    """
    tokens = name.split()
    if not tokens:
        return name
    lead: list[str] = []
    for tok in tokens:
        # Series tokens are upper-case words or bare years/numbers.
        if tok.isupper() or re.fullmatch(r"[A-ZÅÄÖÉÜ0-9./-]+", tok):
            lead.append(tok)
        else:
            break
    return " ".join(lead) if lead else tokens[0].upper()


def _dimensions_for(item: dict, role: Role) -> Dimensions | None:
    """Validated cm dimensions, or None if this listing cannot be trusted.

    Returning None drops the item. That is deliberate: the solver treats these
    numbers as ground truth, so a guess here becomes a confidently wrong layout.
    """
    raw = item.get("dimensions") or {}
    srcs = {k: (v or "").strip().lower() for k, v in (item.get("dimension_sources") or {}).items()}

    def num(key: str) -> float | None:
        v = raw.get(key)
        return float(v) if isinstance(v, (int, float)) and v > 0 else None

    w, d, h = num("width_cm"), num("depth_cm"), num("height_cm")

    # Rugs are recorded wrongly and consistently: "STOENSE Rug - off-white
    # 170x240 cm" arrives as width=170, depth=170, height=240. The rug's
    # LENGTH landed in the height field and depth is a copy of width, so every
    # rug looks 170cm square and 2.4m tall. The real footprint is in the name,
    # which is the only trustworthy source here, so parse it back out.
    if role is Role.RUG:
        m = re.search(r"(\d{2,3})\s*x\s*(\d{2,3})\s*cm", item.get("name") or "")
        if not m:
            return None
        a, b = float(m.group(1)), float(m.group(2))
        # Lay the rug out with its long edge as width, matching how the seed
        # catalog described rugs and how the solver reasons about them.
        w, d = max(a, b), min(a, b)
        # Pile height is never published; 2cm is right for the low-pile and
        # flatweave rugs here, and rugs are non-colliding anyway.
        h = 2.0

    # Cord length masquerading as a footprint. Only lamps are salvageable -
    # their real footprint is a small base - so anything else is dropped.
    if srcs.get("width_cm") in _BAD_FOOTPRINT_SOURCES:
        if role is not Role.FLOOR_LAMP:
            return None
        w = d = _LAMP_FALLBACK_FOOTPRINT_CM
    if srcs.get("depth_cm") in _BAD_FOOTPRINT_SOURCES:
        if role is not Role.FLOOR_LAMP:
            return None
        d = _LAMP_FALLBACK_FOOTPRINT_CM

    if w is None or d is None or h is None:
        return None

    # Plausibility band per role. A "sofa" 15cm wide or a rug 4m long is a
    # parsing artefact, not a product.
    bands = {
        Role.SOFA: ((120, 400), (60, 200), (40, 120)),
        Role.ACCENT_CHAIR: ((40, 140), (40, 140), (40, 130)),
        # Height capped at 60cm: a table serving a sofa sits at or below seat
        # height. IKEA's "side table" category also holds 78cm end tables,
        # which are the right height for an armchair but wrong beside a sofa.
        Role.COFFEE_TABLE: ((30, 200), (30, 140), (20, 60)),
        Role.RUG: ((60, 400), (60, 400), (0.1, 10)),
        Role.FLOOR_LAMP: ((10, 80), (10, 80), (80, 220)),
    }
    (wlo, whi), (dlo, dhi), (hlo, hhi) = bands[role]
    if not (wlo <= w <= whi and dlo <= d <= dhi and hlo <= h <= hhi):
        return None

    # models.Dimensions caps at 400cm; anything above is out of band anyway.
    return Dimensions(width_cm=round(w, 1), depth_cm=round(d, 1), height_cm=round(h, 1))


def _load_raw() -> list[dict]:
    if not PRODUCTS_JSON.exists():
        raise FileNotFoundError(
            f"{PRODUCTS_JSON} not found - the catalog is built from the IKEA scrape"
        )
    with PRODUCTS_JSON.open() as fh:
        return json.load(fh)


def build_items(
    max_variants: int = MAX_VARIANTS_PER_FAMILY,
    source: list[dict] | None = None,
) -> list[CatalogItem]:
    """The scrape, reduced to a usable catalog.

    Order within a family is by price ascending, so the cap keeps the cheapest
    variants rather than an arbitrary slice - which also keeps a budget-priced
    option in every family.
    """
    raw = source if source is not None else _load_raw()

    staged: list[tuple[Role, str, CatalogItem]] = []
    for item in raw:
        role = CATEGORY_ROLE.get(item.get("category") or "")
        if role is None:
            continue

        name = (item.get("name") or "").strip()
        if not name or _is_component(name):
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
        colours = [c for c in (appearance.get("colours") or []) if c]
        materials = [m for m in (appearance.get("materials") or []) if m]
        colour_name, swatch = _swatch_for(colours)

        # Every scraped description opens by repeating the product name, which
        # is already the first thing in embed_text(). Strip it so the embedding
        # spends its budget on the prose that follows.
        desc = (item.get("description") or "").strip()
        if desc.startswith(name):
            desc = desc[len(name):].lstrip(" .-")

        # IKEA titles are "MODEL type - Fabric colour"; the tail names the
        # visible upholstery, which `materials` does not distinguish. Only
        # upholstered roles have one - on a side table the same slot holds a
        # size ("black 38 cm"), which is not a finish and would be noise.
        finish = ""
        if role in (Role.SOFA, Role.ACCENT_CHAIR) and " - " in name:
            tail = name.split(" - ", 1)[1].strip()
            # Drop a trailing size fragment if one crept in.
            tail = re.sub(r"\s*\d+\s*x?\s*\d*\s*cm$", "", tail).strip()
            finish = tail

        seats = item.get("seating_capacity")
        if not isinstance(seats, int) or seats <= 0:
            seats = None

        # InStoreOnly is still buyable, so it counts as in stock; only an
        # explicit OutOfStock does not.
        availability = (item.get("availability") or "").lower()
        in_stock = "outofstock" not in availability

        staged.append((
            role,
            _family(name),
            CatalogItem(
                id=f"ikea-{pid.replace('.', '')}",
                merchant="IKEA",
                title=name,
                role=role,
                # SGD, kept as SGD - see the module docstring.
                price_cents=int(round(price * 100)),
                currency=CURRENCY,
                dimensions=dims,
                materials=materials[:6],
                primary_color=colour_name,
                swatch=swatch,
                style_tags=_style_tags(item, colour_name, materials),
                image_url=images[0],
                checkout_url=url,
                in_stock=in_stock,
                description=desc,
                finish=finish,
                seating_capacity=seats,
                series=_family(name),
            ),
        ))

    # Cap variants per (role, family). Cheapest first so the survivors span the
    # affordable end, and a family's one in-stock variant is never dropped in
    # favour of an out-of-stock sibling.
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
        fams = len({i.title.split()[0].upper() for i in sub})
        print(
            f"  {role:<14} {n:>3} items  {fams:>2} families  "
            f"{min(ws):>5.0f}-{max(ws):<5.0f}cm  "
            f"S${min(ps):>7.2f}-{max(ps):<8.2f}"
        )
