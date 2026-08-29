# Scrapers

Each merchant in the catalog comes from a scrape that lands in a JSON file at
the repository root, which a matching importer in `backend/` reduces to
`CatalogItem`s. The two stages are kept apart on purpose: scraping is slow and
network-bound, importing is fast and pure, so the catalog can be re-derived
(new role mappings, different variant caps) without re-fetching anything.

| Merchant | Scraper | Raw file | Importer |
| --- | --- | --- | --- |
| IKEA SG | *(scraped externally)* | `products.json` | `backend/ikea_import.py` |
| Castlery SG | `scrapers/castlery_scrape.py` | `castlery_products.json` | `backend/castlery_import.py` |

## Castlery

```bash
python -m scrapers.castlery_scrape --sample 80  # what the catalog uses
python -m scrapers.castlery_scrape              # full run, ~1,200 products
python -m scrapers.castlery_scrape --limit 40   # first N, alphabetical
python -m scrapers.castlery_scrape --refresh    # ignore the page cache
```

**The committed scrape is `--sample 80`.** Castlery's catalog is 395 sofas to
9 floor lamps, so `--limit` (a flat alphabetical slice) returns almost no
lamps and the catalog silently loses a role. `--sample` gives each of the five
roles an equal quota, guesses the role from the URL slug, and hands back the
shortfall from any role that cannot fill its quota — floor lamps contribute
all 9 that exist, and the surplus is spread across the rest. The pick is
seeded, so a re-run selects the same products.

Of those 80, **52 become catalog items**; the rest are outdoor furniture,
covers and bed sets, dropped by the importer's allowlist.

Writes `castlery_products.json` and `castlery_report.json`, and caches every
fetched page under `.cache/castlery/` so a re-run costs no requests.

**Where the data comes from.** Every product page carries a schema.org
`Product` block in `application/ld+json` — the vendor's own structured feed,
with sku, name, category, description, price, availability, dimensions,
images, rating and reviews. That is parsed instead of the rendered HTML
because it is a stable contract: the page markup is a Next.js build that
changes with every deploy, the JSON-LD does not.

**On being a good citizen.** `robots.txt` disallows only `*/wishlist`,
`*/checkout/` and `*/account/` — product pages are explicitly permitted. URLs
come from Castlery's own published sitemap rather than from crawling links,
requests are serialised with a delay between them, and a failed page is
recorded in the report and skipped rather than retried in a tight loop.

**The one genuinely tricky field is dimensions.** Castlery publishes them as a
single display string, in four shapes that mean different things:

| Shape | Example | Handling |
| --- | --- | --- |
| Ordinary | `W152 x D88 x H82cm` | Parsed directly |
| Chaise | `W264 x D88/167 x H82cm` | Depth is min/max — the **max** is the footprint |
| Round | `Dia. 8.4 x H24cm` | Diameter becomes both width and depth |
| Multi-part | `Sofa: W322…; Ottoman: W96…` | Primary (first) piece only |
| Rug | `W153 x L244 x H1.5cm` | `L` is read as depth |
| Phi | `Φ98 x H40cm` | Phi is diameter, same as `Dia.` |
| No height | `W153 x L244` (a rug) | 1.5cm pile assumed — **rugs only** |

Anything interpreted rather than measured is recorded in the item's
`dimension_note`, and the original string is always kept in
`dimension_source`, so the importer can tell the two apart instead of trusting
all three numbers equally. A listing whose dimensions cannot be parsed is
dropped by the importer rather than guessed at — the solver treats these
numbers as ground truth, so a guess becomes a confidently wrong layout.

**Colour comes from the image filename.** Castlery's feed has no colour field
and its titles usually omit it, but the variant image is named
`Adams-Armchair-Pearl-Beige-Silver-Front.png`. That is the vendor's own
variant label, so it is a real signal rather than an inference. The importer
matches on word boundaries and takes the *earliest* match, because the
upholstery leads the label and the frame finish trails it — the chair above is
beige with silver legs, not silver.
