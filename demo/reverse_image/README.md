# Reverse image search demo

Two product shots, classified against the catalog and named for the role each
one matches. Both are single-object crops on a white ground, which is what
`/api/shop-the-look` expects — a wide room shot averages every object in it
into one vector and matches nothing in particular.

| File | Role | What it is | Top match | `confident` |
| --- | --- | --- | --- | --- |
| `accent_chair_yenkai_cube.jpg` | `accent_chair` | Beige leatherette cube armchair, contrast piping, black square legs | YEN KAI **Cube Armchair**, S$890 | **true** |
| `accent_chair_taupe_armchair.png` | `accent_chair` | Taupe/greige upholstered armchair, tapered light-wood legs | Castlery **Madison Leather Armchair**, S$1,299 | false |
| `coffee_table_oak_square.png` | `coffee_table` | Square light-oak table, four straight legs, apron rail | Castlery **Bradley Square Coffee Table**, S$519 | false |

Each returns a **different merchant at #1**, which is the point worth showing:
the catalog spans three merchants and the ranking is by appearance, not by
merchant.

The YEN KAI shot is the one **true identification** in the set — it is a photo
of a piece the catalog actually carries, so it comes back `confident: true`
and matches itself at rank 1. The other two are lookalikes; see below.

## Running it

Needs no API key — CLIP runs locally.

```bash
curl -s -X POST http://localhost:8000/api/shop-the-look \
  -F "image=@demo/reverse_image/coffee_table_oak_square.png" \
  -F "limit=5" | python -m json.tool
```

Add `-F "role=coffee_table"` to constrain the search to one category. Without
it the search runs across every role, which is the more honest demo — the
match has to beat sofas and rugs too.

## `confident` is false on two of the three, and that is correct

Each response carries `confident`, which asks whether the top match actually
*resembles* the object or is merely the nearest thing in a small catalog.
The two screenshots come back `false`, because neither is a photo of a product
this catalog sells — they are lookalikes. The scores rank them sensibly; they
do not identify them. Presenting a `confident: false` result as "this is the
Bradley" would be the one genuinely misleading thing this demo could do.

`accent_chair_yenkai_cube.jpg` comes back `true`, and that is the contrast
worth demoing: the same pipeline, on a photo of a piece the catalog really
carries, says so. Running the two side by side shows the flag doing its job
rather than always reading the same way.

## Rebuilding the vectors

The image vectors live in `backend/assets/clip_cache.npz`, keyed by
`item_id|image_url` so a changed product photo re-embeds automatically. After
any catalog change:

```bash
python -m backend.clip_engine   # 227/227 items, ~30s for new images
```

A host serving product imagery must be in `IMAGE_FETCH_ALLOWED_HOSTS`
(`backend/config.py`) or its images are skipped silently — that allowlist is
an SSRF guard, and Castlery's photos are on `res.cloudinary.com` rather than
`castlery.com`.
