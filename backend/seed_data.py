"""Curated furniture catalog.

MERCHANTS AND PRODUCTS ARE FICTIONAL. Prices, stock and checkout URLs are
invented demo data. Real retailers are deliberately not named: their real
pricing and inventory cannot be verified here, and attaching made-up numbers to
a real company's name would misrepresent them.

Dimensions are realistic for their category so the spatial solver is exercised
against plausible constraints rather than toy numbers.
"""

from __future__ import annotations

from .models import CatalogItem, Dimensions, Role

# Fictional storefronts, each with a loose house style.
NORDHAUS = "Nordhaus"  # Scandinavian, light woods
CEDARLINE = "Cedarline"  # Mid-century modern, walnut
MURAYA = "Muraya"  # Japandi, low profile
HESPER = "Hesper & Co."  # Industrial, metal and leather


def _item(
    id: str,
    merchant: str,
    title: str,
    role: Role,
    price_cents: int,
    w: float,
    d: float,
    h: float,
    materials: list[str],
    color: str,
    swatch: str,
    styles: list[str],
    in_stock: bool = True,
) -> CatalogItem:
    slug = id.replace("_", "-")
    return CatalogItem(
        id=id,
        merchant=merchant,
        title=title,
        role=role,
        price_cents=price_cents,
        dimensions=Dimensions(width_cm=w, depth_cm=d, height_cm=h),
        materials=materials,
        primary_color=color,
        swatch=swatch,
        style_tags=styles,
        # Placeholder host; nothing is fetched from it.
        image_url=f"https://cdn.example-demo.invalid/roomcrafter/{slug}.jpg",
        checkout_url=f"https://shop.example-demo.invalid/{slug}",
        in_stock=in_stock,
    )


SEED_ITEMS: list[CatalogItem] = [
    # --- Sofas -------------------------------------------------------------
    _item("sofa-linnea-3s", NORDHAUS, "Linnea 3-Seat Sofa", Role.SOFA,
          129900, 212, 88, 78, ["oak", "wool blend"], "oatmeal", "#C9BCA6",
          ["Scandinavian", "Minimalist"]),
    _item("sofa-linnea-2s", NORDHAUS, "Linnea 2-Seat Sofa", Role.SOFA,
          94900, 164, 86, 78, ["oak", "wool blend"], "oatmeal", "#C9BCA6",
          ["Scandinavian", "Minimalist"]),
    _item("sofa-alder-loveseat", NORDHAUS, "Alder Loveseat", Role.SOFA,
          79900, 148, 82, 74, ["birch", "cotton"], "fog grey", "#B4B7B5",
          ["Scandinavian"]),
    _item("sofa-westport", CEDARLINE, "Westport Walnut Sofa", Role.SOFA,
          164900, 220, 92, 80, ["walnut", "boucle"], "cream", "#DED5C4",
          ["Mid-Century Modern"]),
    _item("sofa-brayden", CEDARLINE, "Brayden Tufted Sofa", Role.SOFA,
          142000, 198, 90, 82, ["walnut", "velvet"], "forest green", "#4A5D4A",
          ["Mid-Century Modern", "Maximalist"]),
    _item("sofa-tatami-low", MURAYA, "Tatami Low Sofa", Role.SOFA,
          118000, 190, 84, 64, ["ash", "linen"], "sand", "#D3C6B2",
          ["Japandi", "Minimalist"]),
    _item("sofa-kiri-modular", MURAYA, "Kiri Modular 3-Seat", Role.SOFA,
          156000, 226, 95, 68, ["ash", "linen"], "clay", "#B99C82",
          ["Japandi"]),
    _item("sofa-foundry", HESPER, "Foundry Leather Sofa", Role.SOFA,
          189900, 208, 94, 84, ["steel", "top-grain leather"], "cognac", "#8A5A34",
          ["Industrial"]),
    _item("sofa-rivet-compact", HESPER, "Rivet Compact Sofa", Role.SOFA,
          86500, 156, 84, 80, ["steel", "canvas"], "charcoal", "#4A4E52",
          ["Industrial", "Minimalist"], in_stock=False),

    # --- Coffee tables -----------------------------------------------------
    _item("ctable-vika-oval", NORDHAUS, "Vika Oval Coffee Table", Role.COFFEE_TABLE,
          34900, 120, 60, 42, ["oak veneer"], "natural oak", "#C8A97E",
          ["Scandinavian"]),
    _item("ctable-nest-pair", NORDHAUS, "Nest Nesting Tables", Role.COFFEE_TABLE,
          27500, 90, 50, 45, ["birch", "powder-coated steel"], "white", "#E8E6E1",
          ["Scandinavian", "Minimalist"]),
    _item("ctable-orbit", CEDARLINE, "Orbit Walnut Coffee Table", Role.COFFEE_TABLE,
          48900, 130, 68, 40, ["walnut", "brass"], "walnut", "#6B4A2F",
          ["Mid-Century Modern"]),
    _item("ctable-spindle", CEDARLINE, "Spindle Round Table", Role.COFFEE_TABLE,
          36500, 90, 90, 44, ["walnut"], "walnut", "#6B4A2F",
          ["Mid-Century Modern"]),
    _item("ctable-ishi-stone", MURAYA, "Ishi Stone-Top Table", Role.COFFEE_TABLE,
          62000, 110, 65, 36, ["travertine", "ash"], "ivory stone", "#DDD6C9",
          ["Japandi"]),
    _item("ctable-hana-low", MURAYA, "Hana Low Table", Role.COFFEE_TABLE,
          29800, 100, 55, 33, ["ash"], "pale ash", "#D9C9AE",
          ["Japandi", "Minimalist"]),
    _item("ctable-girder", HESPER, "Girder Industrial Table", Role.COFFEE_TABLE,
          41200, 122, 62, 41, ["reclaimed pine", "blackened steel"], "dark pine", "#5A4632",
          ["Industrial"]),
    _item("ctable-cinder", HESPER, "Cinder Concrete Table", Role.COFFEE_TABLE,
          53000, 105, 58, 38, ["concrete", "steel"], "slate", "#7C7F82",
          ["Industrial"]),

    # --- Rugs --------------------------------------------------------------
    _item("rug-frost-8x10", NORDHAUS, "Frost Wool Rug 8x10", Role.RUG,
          52900, 305, 244, 2, ["wool"], "ivory", "#EDE7DC",
          ["Scandinavian", "Minimalist"]),
    _item("rug-frost-5x8", NORDHAUS, "Frost Wool Rug 5x8", Role.RUG,
          32900, 244, 152, 2, ["wool"], "ivory", "#EDE7DC",
          ["Scandinavian", "Minimalist"]),
    _item("rug-lattice", CEDARLINE, "Lattice Geometric Rug", Role.RUG,
          44500, 274, 183, 3, ["wool", "cotton"], "rust", "#A9673F",
          ["Mid-Century Modern"]),
    _item("rug-sable-runner", CEDARLINE, "Sable Wide Rug", Role.RUG,
          38900, 244, 168, 3, ["wool"], "umber", "#8B6A4F",
          ["Mid-Century Modern", "Industrial"]),
    _item("rug-washi", MURAYA, "Washi Jute Rug", Role.RUG,
          28900, 275, 180, 2, ["jute"], "natural", "#CBB894",
          ["Japandi", "Minimalist"]),
    _item("rug-suna-large", MURAYA, "Suna Large Rug", Role.RUG,
          61000, 320, 240, 2, ["wool", "jute"], "dune", "#D7C7A9",
          ["Japandi"]),
    _item("rug-anvil", HESPER, "Anvil Distressed Rug", Role.RUG,
          35500, 260, 170, 4, ["polypropylene"], "graphite", "#6E7175",
          ["Industrial"]),

    # --- Accent chairs -----------------------------------------------------
    _item("chair-ora", NORDHAUS, "Ora Accent Chair", Role.ACCENT_CHAIR,
          46900, 72, 76, 82, ["oak", "wool"], "mustard", "#C4923E",
          ["Scandinavian"]),
    _item("chair-fjord", NORDHAUS, "Fjord Lounge Chair", Role.ACCENT_CHAIR,
          58000, 80, 84, 86, ["oak", "sheepskin"], "cream", "#E4DACA",
          ["Scandinavian"]),
    _item("chair-shell", CEDARLINE, "Shell Lounge Chair", Role.ACCENT_CHAIR,
          67500, 78, 80, 84, ["walnut", "leather"], "tan", "#A97C50",
          ["Mid-Century Modern"]),
    _item("chair-zabu", MURAYA, "Zabu Low Chair", Role.ACCENT_CHAIR,
          39800, 70, 74, 68, ["ash", "cotton"], "stone", "#BFB8AC",
          ["Japandi", "Minimalist"]),
    _item("chair-forge", HESPER, "Forge Armchair", Role.ACCENT_CHAIR,
          52000, 76, 82, 80, ["steel", "leather"], "espresso", "#4E3728",
          ["Industrial"]),

    # --- TV units ----------------------------------------------------------
    _item("tv-svea", NORDHAUS, "Svea Media Console", Role.TV_UNIT,
          58900, 160, 40, 48, ["oak veneer"], "natural oak", "#C8A97E",
          ["Scandinavian"]),
    _item("tv-lund-compact", NORDHAUS, "Lund Compact Console", Role.TV_UNIT,
          42000, 120, 38, 46, ["birch"], "white", "#E8E6E1",
          ["Scandinavian", "Minimalist"]),
    _item("tv-cascade", CEDARLINE, "Cascade Walnut Credenza", Role.TV_UNIT,
          78500, 180, 45, 55, ["walnut", "brass"], "walnut", "#6B4A2F",
          ["Mid-Century Modern"]),
    _item("tv-noru", MURAYA, "Noru Low Console", Role.TV_UNIT,
          51000, 150, 38, 40, ["ash", "rattan"], "pale ash", "#D9C9AE",
          ["Japandi"]),
    _item("tv-bracket", HESPER, "Bracket Media Unit", Role.TV_UNIT,
          46500, 155, 42, 50, ["reclaimed pine", "steel"], "dark pine", "#5A4632",
          ["Industrial"]),

    # --- Floor lamps -------------------------------------------------------
    _item("lamp-arc-nord", NORDHAUS, "Arc Floor Lamp", Role.FLOOR_LAMP,
          21900, 38, 38, 165, ["steel", "linen shade"], "white", "#E8E6E1",
          ["Scandinavian", "Minimalist"]),
    _item("lamp-tripod", CEDARLINE, "Tripod Walnut Lamp", Role.FLOOR_LAMP,
          26500, 45, 45, 158, ["walnut", "cotton shade"], "walnut", "#6B4A2F",
          ["Mid-Century Modern"]),
    _item("lamp-washi-column", MURAYA, "Washi Column Lamp", Role.FLOOR_LAMP,
          18900, 32, 32, 145, ["paper", "ash"], "warm white", "#F0E6D2",
          ["Japandi", "Minimalist"]),
    _item("lamp-boom", HESPER, "Boom Task Floor Lamp", Role.FLOOR_LAMP,
          23500, 42, 42, 170, ["steel"], "matte black", "#3A3D40",
          ["Industrial"]),
]


def validate_seed() -> None:
    """Fail loudly on malformed catalog data rather than at query time."""
    if len(SEED_ITEMS) < 30:
        raise ValueError(f"catalog must hold 30+ items, found {len(SEED_ITEMS)}")

    ids = [i.id for i in SEED_ITEMS]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate catalog ids: {sorted(dupes)}")

    # Every role needs at least one in-stock option, or a layout can silently
    # lose a whole category with a confusing "no_fit" reason.
    for role in Role:
        if not any(i.role == role and i.in_stock for i in SEED_ITEMS):
            raise ValueError(f"no in-stock item for role {role.value}")

    for i in SEED_ITEMS:
        d = i.dimensions
        if not (10 <= d.width_cm <= 400 and 10 <= d.depth_cm <= 400):
            raise ValueError(f"{i.id}: implausible footprint {d.width_cm}x{d.depth_cm}")
        if not i.swatch.startswith("#") or len(i.swatch) != 7:
            raise ValueError(f"{i.id}: swatch must be #rrggbb, got {i.swatch!r}")


if __name__ == "__main__":
    validate_seed()
    by_role: dict[str, int] = {}
    for it in SEED_ITEMS:
        by_role[it.role.value] = by_role.get(it.role.value, 0) + 1
    print(f"{len(SEED_ITEMS)} items OK")
    for role, n in sorted(by_role.items()):
        print(f"  {role:<14} {n}")
