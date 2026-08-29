"""Build a HipVan-shaped merchant feed for the onboarding demo.

The point of this file is that HipVan does NOT speak our schema. Every column
is named something else, money is a string with a symbol, dimensions are one
"W x D x H" blob in a single column, and stock is a schema.org URL. The
onboarding page has to normalize all of it - so the demo is only honest if the
input really is shaped like that.
"""
import csv, json, random, re

SRC = "data/all_products.json"
OUT = "demo/hipvan_catalog.csv"

data = json.load(open(SRC))
hv = [x for x in data if x.get("merchant") == "HipVan"]

# A believable medium-merchant spread: a few of each category rather than 30
# sofas. Sorted for a deterministic file - regenerating must not churn the diff.
by_cat = {}
for p in sorted(hv, key=lambda x: x["product_id"]):
    by_cat.setdefault(p["category"], []).append(p)

# A believable mix, deliberately NOT cherry-picked for completeness: roughly a
# third fully measured, the rest missing exactly the dimension HipVan does not
# publish. That ratio is what makes the estimator worth showing - a feed where
# every row is clean demonstrates nothing.
def completeness(p):
    d = p.get("dimensions") or {}
    return sum(1 for k in ("width_cm", "depth_cm", "height_cm") if d.get(k) is not None)

full, partial = [], []
for cat in sorted(by_cat):
    for p in by_cat[cat]:
        c = completeness(p)
        if c == 3 and len(full) < 18:
            full.append(p)
        elif c in (1, 2) and len(partial) < 30:
            partial.append(p)

picked = sorted(full + partial, key=lambda x: x["product_id"])[:48]

def dim_columns(d):
    """HipVan's own measurement shape.

    A complete product gets the single "W x D x H" blob their site shows. An
    incomplete one CANNOT use that blob: "58 x 46" would be read positionally
    as width x depth, silently turning a depth into a width. So a partial
    product publishes its known dimensions as separate labelled columns, which
    is what their export actually does and what any honest feed must do.
    """
    fmt = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    w, dp, h = d.get("width_cm"), d.get("depth_cm"), d.get("height_cm")
    if None not in (w, dp, h):
        return {"measurements": f"{fmt(w)} x {fmt(dp)} x {fmt(h)}",
                "width_cm": "", "depth_cm": "", "height_cm": ""}
    return {
        "measurements": "",
        "width_cm":  fmt(w)  if w  is not None else "",
        "depth_cm":  fmt(dp) if dp is not None else "",
        "height_cm": fmt(h)  if h  is not None else "",
    }


def stock_word(url):
    return (url or "").rsplit("/", 1)[-1]      # schema.org URL -> "InStock"

rows = []
for p in picked:
    ap = p.get("appearance") or {}
    rows.append({
        # Their vocabulary, not ours - this is the whole point.
        "item_code":      p["product_id"],
        "product_name":   p["name"],
        "product_type":   p["category"],
        "retail_price":   f'S${p["price"]:,.2f}',      # symbol + thousands comma
        "curr":           p.get("currency") or "SGD",
        **dim_columns(p.get("dimensions") or {}),
        # Their feed also carries a seat count, which is the single best anchor
        # for estimating a missing sofa width.
        "seats":          p.get("seating_capacity") or "",
        "composition":    ", ".join((ap.get("materials") or [])[:3]),
        "shade":          (ap.get("colours") or ["natural"])[0],
        "hero_image":     (p.get("images") or [""])[0],
        "product_page":   p.get("product_url") or "",
        "stock_status":   stock_word(p.get("availability")),
        "blurb":          (p.get("description") or "")[:110],
    })

# --- The scuffed rows -------------------------------------------------------
# Real feeds are never clean. Each of these breaks in a way the normalizer has
# a specific, nameable answer for, so the merchant sees WHY a row was held back
# rather than a generic failure. Inserted at known offsets so the demo's error
# list is stable.
def clone(i):
    return dict(rows[i])

scuffed = []

# 1. No price at all - the one thing that can never be defaulted, because
#    defaulting it to zero publishes a free product.
r = clone(0); r["item_code"] = "HV-NOPRICE"; r["product_name"] = "Marlow Fabric 3 Seater Sofa"
r["retail_price"] = ""; scuffed.append((6, r))

# 2. Price that is not a number. Parsing must refuse rather than guess.
r = clone(1); r["item_code"] = "HV-POA"; r["product_name"] = "Anton Marble Coffee Table"
r["retail_price"] = "Price on request"; scuffed.append((11, r))

# 3. No dimensions at all - nothing for the estimator to anchor to, so this
#    row must stay rejected rather than be invented.
r = clone(2); r["item_code"] = "HV-DIM2"; r["product_name"] = "Kyoto Rattan Bench"
r["measurements"] = ""; r["width_cm"] = ""; r["depth_cm"] = ""; r["height_cm"] = ""
scuffed.append((16, r))

# 4. A zero dimension - present, parseable, and still wrong.
r = clone(3); r["item_code"] = "HV-DIMZERO"; r["product_name"] = "Wallace Wall Shelf"
r["measurements"] = "90 x 0 x 22"; r["width_cm"] = r["depth_cm"] = r["height_cm"] = ""
scuffed.append((21, r))

# 5. Missing title.
r = clone(4); r["item_code"] = "HV-NONAME"; r["product_name"] = ""; scuffed.append((26, r))

# 6. Missing SKU.
r = clone(5); r["item_code"] = ""; r["product_name"] = "Unnamed Storage Cabinet"; scuffed.append((31, r))

for offset, row in sorted(scuffed, key=lambda t: t[0]):
    rows.insert(min(offset, len(rows)), row)

cols = ["item_code","product_name","product_type","retail_price","curr",
        "measurements","width_cm","depth_cm","height_cm","seats","composition",
        "shade","hero_image","product_page","stock_status","blurb"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# The merchant page fetches its samples over HTTP, so they must live under the
# served frontend root as well as in demo/.
import shutil, pathlib
mirror = pathlib.Path("frontend/samples/hipvan_catalog.csv")
mirror.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(OUT, mirror)

print(f"wrote {OUT} (+ {mirror}): {len(rows)} rows ({len(scuffed)} deliberately broken)")
