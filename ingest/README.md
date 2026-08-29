# Merchant catalog ingester

This demo shows how RoomCrafter can accept catalogs from merchants of different sizes and normalize them into one product schema.

Run from the repository root:

```text
python -m ingest.normalizer
```

Inputs live in `mock_uploads/`. The normalizer accepts CSV and JSON, maps common merchant-specific column names, converts dimensions to centimetres, normalizes prices, and writes `normalized_catalog.json` plus `import_report.json`.

The five mock uploads represent:

- `01_small_studio.csv` - a small maker with simple CSV fields
- `02_small_marketplace.json` - a small seller using nested JSON
- `03_medium_retailer.csv` - a medium retailer using inch dimensions
- `04_medium_design_house.csv` - a medium seller with incomplete inventory data
- `05_large_retailer.json` - a large retailer with a richer product feed

The warnings are intentional: they demonstrate that merchants can upload imperfect data while the ingester explains what needs attention.
