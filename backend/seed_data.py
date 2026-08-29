"""The furniture catalog: real IKEA Singapore listings.

The catalog is built at import time from `products.json`, a scrape of 1,579
IKEA SG listings, reduced by `ikea_import.py` to the subset that is usable for
laying out a living room. Nothing here is invented: names, prices, dimensions,
product photos and checkout links all come from the scrape.

PRICES ARE IN SINGAPORE DOLLARS. `price_cents` is SGD cents and `currency` says
so on every item. The scrape carries no exchange rate, and converting with an
invented one would misstate every price in the app, so nothing is converted.

WHAT THE CATALOG DOES NOT COVER. The scrape contains no TV units, media
consoles or sideboards - the `tv_unit` role was removed from the app rather
than filled with mock data, or with side tables pretending to be consoles. If a
later scrape adds them, restoring the role means re-adding it to `Role`,
`PLACEMENT_ORDER`, `ROLE_PRECISION` and `BUDGET_SHARE`, and mapping its
categories in `ikea_import.CATEGORY_ROLE`.

Availability is as scraped. IKEA's "in store only" is treated as in stock,
since it is still buyable; only an explicit out-of-stock is not.
"""

from __future__ import annotations

from .ikea_import import build_items
from .models import CatalogItem, Role

SEED_ITEMS: list[CatalogItem] = build_items()


def validate_seed() -> None:
    """Fail loudly on a malformed catalog rather than at query time.

    These are the properties the rest of the app assumes. They matter more now
    that the catalog is derived from an external scrape: a re-scrape that
    renames a category or drops a product line should fail at import, not
    surface later as a mysteriously empty result.
    """
    if len(SEED_ITEMS) < 60:
        raise ValueError(f"catalog must hold 60+ items, found {len(SEED_ITEMS)}")

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
        if not i.image_url.startswith(("http://", "https://")):
            raise ValueError(f"{i.id}: image_url must be fetchable, got {i.image_url!r}")

    # Both properties the importer's variant cap exists to preserve. Asserting
    # them here means a scrape or a cap change that leaves a role with a single
    # price point, or with nothing that fits a small room, fails at seed time
    # rather than producing confusing "over budget" or "no fit" results.
    for role in Role:
        stocked = [i for i in SEED_ITEMS if i.role == role and i.in_stock]
        prices = [i.price_cents for i in stocked]
        if max(prices) < min(prices) * 2:
            raise ValueError(
                f"{role.value}: price range too narrow to exercise a budget "
                f"constraint ({min(prices)}-{max(prices)} cents)"
            )
        # Rugs and lamps are exempt from the footprint spread: a lamp's
        # footprint barely varies, and a rug that fits a small room is a
        # different product, not a smaller one.
        if role in (Role.RUG, Role.FLOOR_LAMP):
            continue
        widths = [i.dimensions.width_cm for i in stocked]
        if max(widths) < min(widths) * 1.5:
            raise ValueError(
                f"{role.value}: width spread {min(widths)}-{max(widths)}cm is "
                f"too narrow for both small and large rooms"
            )


if __name__ == "__main__":
    validate_seed()
    by_role: dict[str, int] = {}
    for it in SEED_ITEMS:
        by_role[it.role.value] = by_role.get(it.role.value, 0) + 1
    print(f"{len(SEED_ITEMS)} items OK")
    for role, n in sorted(by_role.items()):
        sub = [i for i in SEED_ITEMS if i.role.value == role]
        lo = min(i.price_cents for i in sub) / 100
        hi = max(i.price_cents for i in sub) / 100
        print(f"  {role:<14} {n:>3}   S${lo:.2f}-{hi:.2f}")
