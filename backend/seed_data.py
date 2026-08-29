"""The furniture catalog: real Singapore listings from three merchants.

Two are scrapes, each reduced by its own importer to the subset usable for
laying out a living room; the third is entered by hand:

  - `products.json`          1,579 IKEA SG listings      -> `ikea_import.py`
  - `castlery_products.json` Castlery SG listings        -> `castlery_import.py`
  - (no feed)                YEN KAI, a local supplier   -> `yenkai_import.py`

Names, dimensions, product photos and checkout links come from the scrapes;
nothing in them is invented.

YEN KAI IS THE EXCEPTION, AND IT IS DELIBERATE. A local supplier with no
website is the third shape a merchant takes, and the app should hold one
without pretending it was scraped. Its measurements come from the merchant and
its photo is real, but its height is an estimate and its price is a
PLACEHOLDER, both labelled as such in `yenkai_import.py`. Anything that shows
a price to a user should read `yenkai_import.PRICE_IS_PLACEHOLDER` first.

WHY TWO MERCHANTS. IKEA SG alone gave the catalog one price band - it tops out
near S$1,699, so a "premium" budget selected the same pieces as a mid one.
Castlery sits above it, which gives the budget logic a real range to work
across and the retrieval layer a genuine choice between merchants.

PRICES ARE IN SINGAPORE DOLLARS. Both merchants are Singapore storefronts
quoting SGD, so `price_cents` is SGD cents throughout and nothing is
converted - there is no exchange rate in either scrape, and inventing one
would misstate every price in the app.

WHAT THE CATALOG DOES NOT COVER. Neither importer maps a media role, so there
are still no TV units in `SEED_ITEMS`. Castlery *does* publish sideboards and
media consoles, so restoring the `tv_unit` role is now possible where it was
not before: it means re-adding it to `Role`, `PLACEMENT_ORDER`,
`ROLE_PRECISION` and `BUDGET_SHARE`, then mapping the categories in
`castlery_import.CATEGORY_ROLE`.

Availability is as scraped. IKEA's "in store only" is treated as in stock,
since it is still buyable; only an explicit out-of-stock is not.

IDS ARE MERCHANT-PREFIXED (`ikea-…`, `castlery-…`, `yenkai-…`) so no two
merchants can collide on a shared SKU.
"""

from __future__ import annotations

from .castlery_import import build_items as build_castlery
from .ikea_import import build_items as build_ikea
from .models import CatalogItem, Role
from .yenkai_import import build_items as build_yenkai


def _build_catalog() -> list[CatalogItem]:
    """Both merchants, merged.

    A missing Castlery scrape is not fatal: the file is optional and the app
    still runs on IKEA alone, which is what every existing test asserts
    against. A missing IKEA scrape IS fatal, since it is the catalog's floor.
    """
    items = list(build_ikea())
    try:
        items.extend(build_castlery())
    except FileNotFoundError:
        pass  # Castlery not scraped yet; IKEA alone is a valid catalog.
    # Hand-entered, so it needs no file and returns [] if its photo is missing
    # rather than raising.
    items.extend(build_yenkai())
    items.sort(key=lambda c: (c.role.value, c.price_cents))
    return items


SEED_ITEMS: list[CatalogItem] = _build_catalog()


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
        # The bar is "a browser can load this", which is what main._http_image_url
        # enforces on the way out - not "this is a URL". An embedded data: URI
        # qualifies, and is how a merchant with no CDN carries its photography.
        if not i.image_url.startswith(("http://", "https://", "data:")):
            raise ValueError(
                f"{i.id}: image_url must be browser-loadable "
                f"(http, https or data:), got {i.image_url[:60]!r}"
            )

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
