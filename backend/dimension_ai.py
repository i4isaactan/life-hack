"""Recover missing product dimensions during merchant catalog ingest.

WHY THIS EXISTS. A real merchant feed is not missing dimensions at random: it
is missing the one measurement that is hardest to publish. HipVan's export
carries depth and height for most upholstery and omits width, because width
varies per configuration. Only 102 of 592 of their products have all three,
so a validator that requires three columns rejects 83% of a real catalog and
tells the merchant to go fix their PIM. That is a correct answer and a useless
one.

WHAT IT DOES. For a row that is missing SOME but not all dimensions, estimate
the missing ones from what is present plus the product's own words - category,
seat count, title. A 3-seater recliner with depth 58 and height 46 has a width
that is genuinely constrained; guessing it within a few centimetres is a much
better outcome for both sides than dropping the product.

WHAT IT MUST NOT DO. An estimate must never be laundered into the merchant's
own data. Every value this module produces is reported as inferred, carries
its source ("openai" or "table"), and the merchant can see exactly which
numbers were not theirs. A row with NO dimensions at all is left alone: with
nothing to anchor to, an estimate is a fabrication, and the honest answer is
the rejection the merchant already gets.

Degrades the same way the rest of this codebase does: model if a key is set,
a documented lookup table if not, and the caller is told which ran.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import config

log = logging.getLogger(__name__)

# Dimensions we know how to recover, in the order a prompt reads best.
_DIMS = ("width_cm", "depth_cm", "height_cm")

# The offline table. Typical footprints in cm for the shapes this catalog
# actually contains, used when no model is configured. Deliberately coarse:
# it exists so the demo runs offline and so a model outage degrades instead of
# failing, not to compete with the model. Keyed by substrings matched against
# "<category> <title>", most specific first.
_TYPICAL: tuple[tuple[tuple[str, ...], dict[str, float]], ...] = (
    (("1 seater", "1-seater", "armchair", "lounge chair", "accent chair"),
     {"width_cm": 85, "depth_cm": 85, "height_cm": 80}),
    (("2 seater", "2-seater", "loveseat"),
     {"width_cm": 150, "depth_cm": 90, "height_cm": 80}),
    (("3 seater", "3-seater"),
     {"width_cm": 210, "depth_cm": 92, "height_cm": 80}),
    (("4 seater", "4-seater"),
     {"width_cm": 250, "depth_cm": 95, "height_cm": 80}),
    (("l-shaped", "sectional", "chaise"),
     {"width_cm": 280, "depth_cm": 165, "height_cm": 82}),
    (("sofa", "couch", "settee"),
     {"width_cm": 200, "depth_cm": 90, "height_cm": 80}),
    (("coffee table",),   {"width_cm": 110, "depth_cm": 60, "height_cm": 40}),
    (("side table", "bedside"), {"width_cm": 45, "depth_cm": 45, "height_cm": 55}),
    (("dining table",),   {"width_cm": 180, "depth_cm": 90, "height_cm": 75}),
    (("study table", "desk"), {"width_cm": 140, "depth_cm": 65, "height_cm": 75}),
    (("tv console", "sideboard", "cabinet"),
     {"width_cm": 160, "depth_cm": 40, "height_cm": 55}),
    (("bench", "ottoman"), {"width_cm": 110, "depth_cm": 45, "height_cm": 45}),
    (("dining chair", "side chair", "bar chair", "stool"),
     {"width_cm": 45, "depth_cm": 50, "height_cm": 85}),
    (("rug", "carpet"),   {"width_cm": 200, "depth_cm": 140, "height_cm": 1}),
    (("bed",),            {"width_cm": 160, "depth_cm": 200, "height_cm": 100}),
    (("shelf", "bookcase"), {"width_cm": 80, "depth_cm": 30, "height_cm": 180}),
    (("floor lamp", "lamp"), {"width_cm": 35, "depth_cm": 35, "height_cm": 160}),
)

# A recovered dimension outside this range is not plausible furniture, and is
# far more likely to be a model slip or a unit error than a real product. Such
# a value is dropped, and the row keeps its ordinary "missing dimensions"
# rejection rather than being published with nonsense.
_MIN_CM = 5.0
_MAX_CM = 400.0

# Enough context for the estimate, and nothing else. Titles and categories are
# merchant-controlled text, so they are passed as data in a JSON payload rather
# than interpolated into the instructions.
_PROMPT = """You estimate missing furniture dimensions for a product catalog.

For each item you are given its title, category, seat count (may be null) and
the dimensions that ARE known, in centimetres. Estimate ONLY the dimensions
listed in that item's "missing" array.

Rules:
- Use the known dimensions as anchors. They are measured facts; your estimates
  must be consistent with them.
- width_cm is the side-to-side span, depth_cm front-to-back, height_cm floor
  to the highest point.
- Answer in centimetres, as plain numbers. No ranges, no units, no prose.
- If you genuinely cannot estimate a value, use null for it. A null is far
  better than a guess: a wrong dimension makes a product unplaceable in a room
  layout, while a null simply leaves the row for the merchant to fix.

Return ONLY a JSON array, one object per input item, in the same order:
[{"id": "<the id you were given>", "width_cm": 210, "depth_cm": null}]
"""


def _blob(item: dict[str, Any]) -> str:
    return f"{item.get('category') or ''} {item.get('title') or ''}".lower()


def _from_table(item: dict[str, Any], missing: list[str]) -> dict[str, float]:
    """Offline estimate: the first matching shape's typical footprint."""
    blob = _blob(item)
    for needles, dims in _TYPICAL:
        if any(n in blob for n in needles):
            return {k: dims[k] for k in missing if k in dims}
    return {}


def _plausible(value: Any) -> float | None:
    """A dimension we are willing to publish, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (_MIN_CM <= number <= _MAX_CM):
        return None
    return round(number, 1)


def _parse_reply(raw: str) -> list[dict[str, Any]]:
    """The model's JSON array, tolerant of a ```json fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("dimension estimator returned non-JSON; ignoring")
        return []
    return parsed if isinstance(parsed, list) else []


class DimensionEstimator:
    """Fills in missing dimensions, by model where available and table if not."""

    def __init__(self) -> None:
        self._client = None
        self.source = "table"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("dimension estimator falling back to table: %s", exc)

    def estimate(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Estimate missing dimensions for a batch of partially-measured rows.

        `items` are dicts with id/title/category/seats and whichever of
        width_cm/depth_cm/height_cm are known. Returns {id: {dim: value, ...,
        "_source": "openai"|"table"}} holding ONLY the values that were
        missing and could be recovered plausibly.

        One call for the whole batch: a 500-row feed must not become 500
        requests, and the model does not need rows to be independent.
        """
        pending: list[tuple[dict[str, Any], list[str]]] = []
        for item in items:
            missing = [d for d in _DIMS if item.get(d) in (None, "")]
            known = [d for d in _DIMS if d not in missing]
            # Nothing missing: nothing to do. Nothing known: nothing to anchor
            # an estimate to, so this is a rejection rather than a guess.
            if missing and known:
                pending.append((item, missing))
        if not pending:
            return {}

        out: dict[str, dict[str, Any]] = {}
        if self._client is not None:
            out = self._from_model(pending)

        # Whatever the model declined or the outage cost us, the table covers -
        # so a partial model answer still publishes the rest of the batch.
        for item, missing in pending:
            key = str(item.get("id"))
            got = out.setdefault(key, {"_source": "table"})
            for dim in missing:
                if got.get(dim) is not None:
                    continue
                value = _plausible(_from_table(item, [dim]).get(dim))
                if value is not None:
                    got[dim] = value
                    # A row the model partly answered is still partly a table
                    # estimate; say so rather than crediting it all to the model.
                    if got.get("_source") == "openai":
                        got["_source"] = "openai+table"
        return {k: v for k, v in out.items() if len(v) > 1}

    def _from_model(self, pending: list[tuple[dict[str, Any], list[str]]]) -> dict[str, dict[str, Any]]:
        payload = [
            {
                "id": str(item.get("id")),
                "title": str(item.get("title") or "")[:120],
                "category": str(item.get("category") or "")[:60],
                "seats": item.get("seats"),
                "known": {d: item[d] for d in _DIMS if item.get(d) not in (None, "")},
                "missing": missing,
            }
            for item, missing in pending
        ]
        try:
            resp = self._client.chat.completions.create(
                model=config.DIMENSION_MODEL,
                max_tokens=config.DIMENSION_MAX_TOKENS,
                temperature=0,          # an estimate should not vary per upload
                messages=[
                    {"role": "system", "content": _PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - never fail an upload
            # An upload must not fail because the estimator was slow or down.
            # The table below still runs, so the merchant gets an answer.
            log.warning("dimension estimator call failed: %s", exc)
            return {}

        wanted = {str(i.get("id")): m for i, m in pending}
        out: dict[str, dict[str, Any]] = {}
        for row in _parse_reply(raw):
            if not isinstance(row, dict):
                continue
            key = str(row.get("id"))
            if key not in wanted:
                continue
            got: dict[str, Any] = {"_source": "openai"}
            for dim in wanted[key]:
                value = _plausible(row.get(dim))
                if value is not None:
                    got[dim] = value
            if len(got) > 1:
                out[key] = got
        return out


_estimator: DimensionEstimator | None = None


def get_estimator() -> DimensionEstimator:
    """Process-wide estimator, so the client is built once."""
    global _estimator
    if _estimator is None:
        _estimator = DimensionEstimator()
    return _estimator
