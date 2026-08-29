"""The YEN KAI catalog: a small local supplier, entered by hand.

WHY THIS EXISTS SEPARATELY. IKEA and Castlery arrive as scrapes of a public
product feed. YEN KAI has no feed and no website to scrape - it is a local
supplier whose catalog is a photograph and a measurement. That is the third
shape a merchant can take, and the app should hold it without pretending it
came from somewhere it did not.

WHAT IS REAL HERE, AND WHAT IS NOT. This matters more than usual, because a
hand-entered item has no upstream source to check it against:

  - THE PHOTOGRAPH is real, taken of the actual piece.
  - THE FOOTPRINT (110 x 100cm) was measured and supplied by the merchant.
  - THE HEIGHT is an ESTIMATE read off the photograph against the catalog's
    armchair band. It is recorded as such in `MEASURED` below. Height is the
    one dimension the solver barely uses for a free-standing chair - nothing
    stacks on it and it collides in plan - so an estimate here is the cheapest
    of the three to be wrong about. It is still an estimate, and labelled one.
  - THERE IS NO PRICE from a price list. `PRICE_CENTS` is a placeholder and is
    marked with `price_is_placeholder`, because a made-up price on a real
    piece of furniture is the one field that could mislead someone into a
    purchase decision. Set it when YEN KAI quotes one.

THE IMAGE IS EMBEDDED, NOT HOTLINKED. The other two merchants' photos are CDN
URLs. YEN KAI has no CDN, so the photo ships with the repo and is emitted as a
`data:` URI - a scheme `main._http_image_url` already passes through, and one
that needs no new static route and no addition to the SSR-guard allowlist.
"""

from __future__ import annotations

import base64
from pathlib import Path

from .models import CatalogItem, Dimensions, Role

ASSETS = Path(__file__).resolve().parent / "assets" / "merchants"

MERCHANT = "YEN KAI"

# Whether each dimension was measured or estimated. The solver treats all
# three as ground truth, so which is which is worth stating rather than
# leaving for someone to guess from a round number.
MEASURED = {"width_cm": True, "depth_cm": True, "height_cm": False}

# No price list exists yet. Carried explicitly so a reader does not mistake it
# for a quote - see the module docstring.
PRICE_CENTS = 89_000
PRICE_IS_PLACEHOLDER = True


def _data_uri(filename: str) -> str:
    """The product photo as a `data:` URI, or "" when the file is missing.

    Returning "" rather than raising keeps a missing asset from taking down
    the whole catalog at import; the item is simply dropped by `build_items`,
    exactly as a scraped item with no image would be.
    """
    path = ASSETS / filename
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def build_items() -> list[CatalogItem]:
    """The YEN KAI catalog. One piece today; the list is the extension point."""
    image = _data_uri("yenkai-cube-armchair-beige.jpg")
    if not image:
        return []

    return [
        CatalogItem(
            id="yenkai-cube-armchair-beige",
            merchant=MERCHANT,
            title="YEN KAI Cube Armchair - beige leatherette",
            role=Role.ACCENT_CHAIR,
            price_cents=PRICE_CENTS,
            currency="SGD",
            # 110 x 100cm measured by the merchant. Height estimated from the
            # photograph - a low boxy lounge chair sits around 70cm to the
            # back, which is mid-band for this role.
            dimensions=Dimensions(width_cm=110.0, depth_cm=100.0, height_cm=70.0),
            materials=["leatherette", "foam", "wood"],
            primary_color="beige",
            swatch="#C9B79C",
            # Squared-off cube silhouette with contrast piping and dark square
            # legs. Read off the photograph, not asserted by the merchant.
            style_tags=["Contemporary", "Minimalist"],
            image_url=image,
            # No storefront to link to. Empty rather than a fabricated URL:
            # the UI already handles a missing checkout link, and a dead one
            # would be worse than none.
            checkout_url="",
            in_stock=True,
            description=(
                "Squared-off lounge chair in beige leatherette with dark "
                "contrast piping along the arms and back, on black square "
                "legs. Deep single seat cushion and a low boxy frame."
            ),
            finish="Beige leatherette",
            seating_capacity=1,
            series="Cube",
        ),
    ]


if __name__ == "__main__":
    items = build_items()
    print(f"{MERCHANT}: {len(items)} item(s)")
    for i in items:
        d = i.dimensions
        est = "" if all(MEASURED.values()) else "  (height estimated)"
        price = f"S${i.price_cents / 100:.2f}"
        if PRICE_IS_PLACEHOLDER:
            price += " [placeholder]"
        print(
            f"  {i.role.value:<14} {i.title[:44]:<46} "
            f"{d.width_cm:.0f}x{d.depth_cm:.0f}x{d.height_cm:.0f}cm  {price}{est}"
        )
