"""Normalize intentionally inconsistent merchant feeds for the demo."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
UPLOADS = ROOT / "mock_uploads"

def first(row, *keys, default=""):
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            return value
    return default

def dimensions(row):
    d = row.get("size") or row.get("dimensions") or {}
    if isinstance(d, dict):
        values = [d.get("w", d.get("width")), d.get("d", d.get("depth")), d.get("h", d.get("height"))]
        unit = str(d.get("unit", "cm")).lower()
    else:
        values = re.split(r"\s*[x×]\s*", str(d))
        unit = "cm"
    values = [float(v) if str(v).replace(".", "", 1).isdigit() else None for v in values[:3]]
    if len(values) < 3 or any(v is None for v in values):
        return None
    if not d and row.get("width_in"):
        values = [row.get("width_in"), row.get("depth_in"), row.get("height_in")]
        unit = "in"
    if unit in {"in", "inch", "inches"} or any(k.endswith("_in") for k in row):
        values = [round(v * 2.54, 1) for v in values]
    return {"width": values[0], "depth": values[1], "height": values[2], "unit": "cm"}

def normalize(raw, filename):
    merchant = (raw.get("merchant_name") or raw.get("seller") or raw.get("merchant")) if isinstance(raw, dict) else None
    merchant = merchant or filename.split("_")[1].title()
    rows = (raw.get("products") or raw.get("items") or raw) if isinstance(raw, dict) else raw
    if isinstance(rows, dict): rows = [rows]
    output, warnings = [], []
    for index, row in enumerate(rows, 1):
        price = row.get("price") if isinstance(row.get("price"), dict) else None
        cents = round(float(price.get("value") if price else first(row, "price", "unit_price_usd", "amount", default=0)) * (1 if row.get("amount") else 100))
        dims = dimensions(row)
        if not dims: warnings.append({"row": index, "issue": "Missing or invalid dimensions"})
        item = {
            "id": first(row, "sku", "product_id", "item_code", "product_code", "id"),
            "merchant": merchant,
            "name": first(row, "name", "title", "product_title", "display_name"),
            "category": first(row, "category", "type", "product_type", "department", default="other").lower(),
            "price_cents": cents,
            "currency": (price or {}).get("currency", first(row, "currency", default="USD")),
            "color": first(row, "colour", "color", "finish", "hue"),
            "material": first(row, "material", "fabric", "composition"),
            "dimensions_cm": dims,
            "image_url": first(row, "image", "photo", "product_image", "hero_image", default=((row.get("media") or [{}])[0].get("url", ""))),
            "checkout_url": first(row, "checkout", "url", "product_page", "buy_link", default=(row.get("links") or {}).get("checkout", "")),
            "stock_status": first(row, "availability", "stock", default=(row.get("inventory") or {}).get("status", "unknown")).lower(),
        }
        if not item["id"] or not item["name"]: warnings.append({"row": index, "issue": "Missing product ID or name"})
        output.append(item)
    return merchant, output, warnings

def load(path):
    if path.suffix == ".json": return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def main():
    catalog, report = [], {"files": [], "total_products": 0, "warnings": 0}
    for path in sorted(UPLOADS.iterdir()):
        merchant, items, warnings = normalize(load(path), path.stem)
        catalog.extend(items)
        report["files"].append({"file": path.name, "merchant": merchant, "products": len(items), "warnings": warnings})
        report["total_products"] += len(items); report["warnings"] += len(warnings)
    (ROOT / "normalized_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (ROOT / "import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Normalized {report['total_products']} products from {len(report['files'])} merchants with {report['warnings']} warnings.")

if __name__ == "__main__": main()
