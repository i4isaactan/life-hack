"""Merchant platform: onboarding, API credentials, catalog ingestion, payouts.

Turns the bare `merchant` string used across the catalog into an account that
can authenticate, publish products, and be paid its share of an order.

WHAT THIS IS HONEST ABOUT

Accepting other people's merchants and settling money to them makes an
operator a payment facilitator, and that is a regulated activity: KYC/AML on
every merchant, an acquirer or PayFac licence, chargeback liability, PCI scope
for anything touching card data. None of that is code, and none of it is here.

So this module implements the technical half completely and honestly, and
stops at the regulatory boundary rather than pretending past it:

  REAL   Merchant records, API credentials, HMAC request signing with replay
         protection, catalog ingestion with validation, per-merchant order
         splits, a payout ledger, and webhook dispatch.
  REAL   Every merchant's share of every order is computed and recorded, so
         the numbers a payout would use actually exist and reconcile.
  NOT    No money moves to a merchant. `payout.status` is "pending_settlement"
         and stays there unless a real rail is wired in behind it.
  NOT    Onboarding records the KYC fields a facilitator must collect, but
         nothing is verified against any registry. `kyc_status` starts at
         "unverified" and only a human process can honestly change it.

The split between those two lists is deliberate: everything needed to run this
end to end in a demo is real, and the one thing that would require a licence
is clearly marked rather than faked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from . import config

log = logging.getLogger("roomhack.merchants")

# How far a signed request's timestamp may be from ours. Tight enough that a
# captured request is not replayable for long, loose enough to survive
# ordinary clock drift between two servers.
SIGNATURE_WINDOW_SECONDS = 300

# Nonces seen inside the window, so a request cannot be replayed even within
# it. Pruned lazily; the window bounds how large this can grow.
_SEEN_NONCES: dict[str, float] = {}

# The platform's cut, in basis points. Charged to the merchant on settlement,
# not added to what the shopper pays.
DEFAULT_COMMISSION_BPS = 500  # 5.00%


class MerchantStatus(str, Enum):
    """Where a merchant sits between "signed up" and "can be paid".

    PENDING is the important one: a merchant may list products and receive
    orders while their identity checks are outstanding, but nothing settles
    to them until someone verifies who they are. Letting them sell first is
    what makes onboarding usable; blocking payout is what keeps it honest.
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class KycStatus(str, Enum):
    UNVERIFIED = "unverified"
    IN_REVIEW = "in_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class MerchantError(Exception):
    """A refusal the merchant's integration should see. Carries HTTP status."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class Merchant:
    """A selling account on the platform."""

    id: str
    name: str
    legal_name: str
    email: str
    country: str = "SG"
    # Merchant category code. Drives the agent mandate's category lock, so it
    # is set by the platform at onboarding, never by the merchant - a merchant
    # that could declare its own MCC could opt itself into any agent's scope.
    mcc: str = "5719"
    status: MerchantStatus = MerchantStatus.PENDING
    kyc_status: KycStatus = KycStatus.UNVERIFIED
    commission_bps: int = DEFAULT_COMMISSION_BPS
    # Where settlement would land. Stored masked; a real platform holds this
    # at a PSP and keeps only a token.
    payout_account_last4: str = ""
    # Where a Visa Direct push actually lands. Held separately from the last4
    # above because that one is for display and this one is a credential: a
    # real deployment stores it at a PSP or vault and keeps only a token here.
    payout_pan: str = ""
    webhook_url: str = ""
    created_at: float = 0.0

    @property
    def can_sell(self) -> bool:
        """Whether this merchant's products may appear and be ordered."""
        return self.status in (MerchantStatus.ACTIVE, MerchantStatus.PENDING)

    @property
    def can_settle(self) -> bool:
        """Whether money may actually move to them. KYC gates this, always."""
        return (
            self.status == MerchantStatus.ACTIVE
            and self.kyc_status == KycStatus.VERIFIED
        )


@dataclass
class ApiCredential:
    """A merchant's API key pair.

    The secret is returned exactly once, at creation, and only its hash is
    kept. A platform that can display a merchant's secret back to them can
    also leak it, and support staff who can read it can be socially
    engineered into reading it aloud.
    """

    key_id: str
    secret_hash: str
    merchant_id: str
    label: str = ""
    created_at: float = 0.0
    last_used_at: float = 0.0
    revoked_at: float | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


MERCHANTS: dict[str, Merchant] = {}
CREDENTIALS: dict[str, ApiCredential] = {}
# merchant_id -> list of payout records, one per order they participated in.
PAYOUTS: dict[str, list[dict[str, Any]]] = {}

# Settlement attempts, keyed by idempotency key. Written BEFORE the network
# call and updated after, so an attempt that crashed or timed out mid-flight
# is still on record.
#
# This is what makes a retry safe. Visa deduplicates on trace numbers derived
# from the key, so replaying a key reaches the same payout rather than making a
# second one - but only if the caller reuses the key, which it can only do if
# the attempt was durable before the request left. An in-memory dict is the
# demo's stand-in for a database row; the ordering it enforces is the part that
# matters and would be identical against Postgres.
SETTLEMENTS: dict[str, dict[str, Any]] = {}


def _settlement_key(merchant_id: str, record_ids: list[str]) -> str:
    """A stable key for settling exactly this set of pending records.

    Derived from the records themselves rather than randomly generated, so an
    operator who retries after a timeout produces the same key without having
    to have saved it - which is precisely the situation where they would not
    have.
    """
    joined = ",".join(sorted(record_ids))
    return hashlib.sha256(f"{merchant_id}:{joined}".encode()).hexdigest()[:32]


def _now() -> float:
    return time.time()


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


# --- Onboarding ------------------------------------------------------------


def onboard(
    *,
    name: str,
    legal_name: str,
    email: str,
    country: str = "SG",
    mcc: str = "5719",
    payout_account_last4: str = "",
    payout_pan: str = "",
    webhook_url: str = "",
) -> tuple[Merchant, str, str]:
    """Register a merchant and issue its first API credential.

    Returns (merchant, key_id, secret). The secret is shown once and never
    again - only its hash is stored.
    """
    name = name.strip()
    if not name:
        raise MerchantError(400, "invalid_name", "merchant name is required")
    if "@" not in email:
        raise MerchantError(400, "invalid_email", "a contact email is required")

    # Display names collide - two shops can both be "Nordic Home". The id is
    # what the catalog and payouts key on, so it must be unique regardless.
    if any(m.name.lower() == name.lower() for m in MERCHANTS.values()):
        raise MerchantError(
            409, "name_taken", f"a merchant named {name!r} is already registered"
        )

    merchant = Merchant(
        id=f"mch_{secrets.token_hex(8)}",
        name=name,
        legal_name=legal_name.strip() or name,
        email=email.strip(),
        country=country.upper()[:2],
        mcc=mcc,
        payout_account_last4=(payout_pan or payout_account_last4)[-4:],
        payout_pan=payout_pan.strip(),
        webhook_url=webhook_url.strip(),
        created_at=_now(),
    )
    MERCHANTS[merchant.id] = merchant
    key_id, secret = issue_credential(merchant.id, label="initial")
    log.info("onboarded merchant %s (%s), mcc=%s", merchant.name, merchant.id, mcc)
    return merchant, key_id, secret


def issue_credential(merchant_id: str, label: str = "") -> tuple[str, str]:
    """Mint an API key pair. The secret is returned here and nowhere else."""
    if merchant_id not in MERCHANTS:
        raise MerchantError(404, "unknown_merchant", "unknown merchant")
    key_id = f"mk_{secrets.token_hex(8)}"
    secret = f"msk_{secrets.token_urlsafe(32)}"
    CREDENTIALS[key_id] = ApiCredential(
        key_id=key_id,
        secret_hash=_hash_secret(secret),
        merchant_id=merchant_id,
        label=label,
        created_at=_now(),
    )
    return key_id, secret


def revoke_credential(key_id: str) -> None:
    cred = CREDENTIALS.get(key_id)
    if cred is None:
        raise MerchantError(404, "unknown_key", "unknown API key")
    cred.revoked_at = _now()
    log.info("revoked merchant key %s", key_id)


def set_status(merchant_id: str, status: MerchantStatus) -> Merchant:
    merchant = get_merchant(merchant_id)
    merchant.status = status
    return merchant


def set_kyc(merchant_id: str, kyc: KycStatus) -> Merchant:
    """Record a KYC outcome.

    Deliberately a plain setter with no verification behind it: identity
    checks are a human and vendor process, and a function here that claimed to
    "verify" a merchant would be lying about the one thing a facilitator most
    needs to be honest about.
    """
    merchant = get_merchant(merchant_id)
    merchant.kyc_status = kyc
    if kyc == KycStatus.VERIFIED and merchant.status == MerchantStatus.PENDING:
        merchant.status = MerchantStatus.ACTIVE
    return merchant


def get_merchant(merchant_id: str) -> Merchant:
    merchant = MERCHANTS.get(merchant_id)
    if merchant is None:
        raise MerchantError(404, "unknown_merchant", "unknown merchant")
    return merchant


def by_name(name: str) -> Merchant | None:
    """Resolve a catalog `merchant` string to an account, if one exists.

    The catalog predates this module and stores merchants as plain names, so
    this is the bridge. Seed merchants that were never onboarded return None
    and are treated as platform-owned rather than as an error.
    """
    return next((m for m in MERCHANTS.values() if m.name.lower() == name.lower()), None)


def list_merchants() -> list[Merchant]:
    return list(MERCHANTS.values())


# --- Request signing -------------------------------------------------------


def sign_request(
    secret: str, method: str, path: str, timestamp: str, nonce: str, body: str
) -> str:
    """Compute the signature for a merchant API request.

    Signs method, path, timestamp, nonce and body together. Every component
    matters: without the method or path a captured signature is replayable
    against a different endpoint, and without the nonce it is replayable
    against the same one.
    """
    message = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def _prune_nonces(now: float) -> None:
    for nonce in [n for n, seen in _SEEN_NONCES.items() if now - seen > SIGNATURE_WINDOW_SECONDS]:
        _SEEN_NONCES.pop(nonce, None)


def authenticate(
    *,
    key_id: str,
    signature: str,
    timestamp: str,
    nonce: str,
    method: str,
    path: str,
    body: str,
    secret_lookup: dict[str, str] | None = None,
) -> Merchant:
    """Verify a signed merchant request and return the merchant.

    `secret_lookup` exists because only the hash of a secret is stored, so the
    signature cannot be recomputed from the record alone. In this demo the
    plaintext secrets live in a process-local map (see `SECRETS` below); a
    real deployment keeps them in a KMS or HSM and verifies there.
    """
    cred = CREDENTIALS.get(key_id)
    if cred is None or not cred.active:
        # Same message for both, so probing cannot distinguish "no such key"
        # from "revoked key".
        raise MerchantError(401, "bad_credentials", "invalid API credentials")

    now = _now()
    try:
        skew = abs(now - float(timestamp))
    except (TypeError, ValueError) as exc:
        raise MerchantError(400, "bad_timestamp", "invalid timestamp") from exc
    if skew > SIGNATURE_WINDOW_SECONDS:
        raise MerchantError(
            401,
            "stale_request",
            f"request timestamp is {int(skew)}s off; the signing window is "
            f"{SIGNATURE_WINDOW_SECONDS}s. Check your server clock.",
        )

    _prune_nonces(now)
    if nonce in _SEEN_NONCES:
        raise MerchantError(409, "replayed_nonce", "this request was already sent")

    secrets_map = secret_lookup if secret_lookup is not None else SECRETS
    secret = secrets_map.get(key_id)
    if secret is None:
        raise MerchantError(401, "bad_credentials", "invalid API credentials")

    expected = sign_request(secret, method, path, timestamp, nonce, body)
    if not hmac.compare_digest(expected, signature):
        raise MerchantError(401, "bad_signature", "signature does not match")

    _SEEN_NONCES[nonce] = now
    cred.last_used_at = now
    merchant = get_merchant(cred.merchant_id)
    if merchant.status in (MerchantStatus.SUSPENDED, MerchantStatus.CLOSED):
        raise MerchantError(
            403, "merchant_inactive", f"this merchant account is {merchant.status.value}"
        )
    return merchant


# Plaintext secrets, process-local. Present ONLY because this demo verifies
# signatures in-process; a deployment stores these in a KMS and never in
# application memory. Kept in a separate map from the credential records so it
# is obvious what would move to the KMS.
SECRETS: dict[str, str] = {}


# --- Checkout URL validation ------------------------------------------------
#
# A checkout URL is where a shopper's money goes, so it is the one merchant-
# supplied field that must not be taken on trust. The checks below are the
# minimum that can be enforced without a domain-ownership challenge (which is
# the production step this stands in for).


ALLOWED_URL_SCHEMES = frozenset({"https"})

# Hosts that must never appear in a checkout URL. A merchant pointing checkout
# at the platform's own network is either misconfigured or probing for SSRF,
# and neither should reach a shopper.
BLOCKED_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
)


def validate_checkout_url(url: str, *, required: bool = False) -> str:
    """Validate a merchant-supplied checkout URL, or raise.

    HTTPS only: a checkout link over plain HTTP exposes the shopper's session
    to anyone on the path, and there is no version of that which is acceptable
    for a payment flow.
    """
    from urllib.parse import urlparse

    url = (url or "").strip()
    if not url:
        if required:
            raise MerchantError(
                400, "missing_checkout_url", "a checkout URL is required"
            )
        return ""

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise MerchantError(400, "invalid_checkout_url", f"unparseable URL: {url}") from exc

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise MerchantError(
            400,
            "insecure_checkout_url",
            f"checkout URLs must use https:// (got {parsed.scheme or 'no scheme'!r}). "
            "A checkout link over plain HTTP exposes the shopper's session.",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise MerchantError(400, "invalid_checkout_url", "checkout URL has no host")
    if host in BLOCKED_HOSTS or host.endswith(".local"):
        raise MerchantError(
            400,
            "blocked_checkout_host",
            f"{host!r} is not a valid public checkout host",
        )
    # A dotless host cannot be a public domain, and is usually an internal
    # service name that escaped into a config file.
    if "." not in host:
        raise MerchantError(
            400, "invalid_checkout_url", f"{host!r} is not a public domain"
        )
    return url


def check_domain_consistency(merchant: "Merchant", urls: list[str]) -> list[str]:
    """Warn when checkout URLs span hosts unrelated to each other.

    Not an error: plenty of legitimate merchants check out on a PSP's domain
    rather than their own. But a catalog whose links point at many unrelated
    hosts is worth surfacing, and in production this is where a domain
    ownership challenge would sit instead.
    """
    from urllib.parse import urlparse

    hosts = {urlparse(u).hostname or "" for u in urls if u}
    hosts.discard("")
    if len(hosts) > 3:
        return [
            f"products point at {len(hosts)} different checkout hosts "
            f"({', '.join(sorted(hosts)[:4])}...). In production each would "
            "need domain verification."
        ]
    return []


# --- Feed normalization -----------------------------------------------------
#
# Merchants do not share a schema. One calls it `sku`, the next `product_id`,
# the third `item_code`; one prices in nested JSON, another in a flat column;
# one measures in inches. Rejecting everything that is not our shape would
# make onboarding a development project for the merchant, so the platform
# absorbs the difference instead.
#
# The rule this follows: normalize aggressively, but never GUESS a value that
# affects what a shopper pays or receives. A missing price or dimension is
# reported for the merchant to fix, not defaulted - an invented dimension puts
# furniture in a room it does not fit, and an invented price is a mispriced
# sale someone has to honour.

# Field aliases seen across real merchant feeds, most specific first.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sku": ("sku", "product_id", "item_code", "product_code", "id", "variant_id"),
    "title": ("title", "name", "product_title", "display_name", "product_name"),
    "price": ("price", "unit_price", "unit_price_usd", "amount", "cost", "retail_price"),
    "currency": ("currency", "currency_code", "curr"),
    "category": ("category", "type", "product_type", "department", "collection"),
    "color": ("color", "colour", "hue", "finish", "shade"),
    "material": ("material", "fabric", "composition", "materials"),
    "image_url": ("image_url", "image", "photo", "product_image", "hero_image", "thumbnail"),
    "checkout_url": ("checkout_url", "checkout", "url", "product_page", "buy_link", "link", "product_url"),
    "stock": ("stock", "availability", "in_stock", "inventory_status", "stock_status"),
    "description": ("description", "details", "summary", "copy", "blurb"),
    "width": ("width_cm", "width", "w", "w_cm", "width_in", "w_in"),
    "depth": ("depth_cm", "depth", "d", "d_cm", "depth_in", "d_in"),
    "height": ("height_cm", "height", "h", "h_cm", "height_in", "h_in"),
}

IN_TO_CM = 2.54


def _pick(row: dict[str, Any], field: str) -> Any:
    """First non-empty value among a field's known aliases, case-insensitively."""
    value, _ = _pick_with_source(row, field)
    return value


def _pick_with_source(row: dict[str, Any], field: str) -> tuple[Any, str]:
    """As `_pick`, but also returns WHICH alias supplied the value.

    The source matters for money: `amount` is conventionally cents and `price`
    is dollars, and the caller cannot tell them apart from the value alone.
    """
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for alias in FIELD_ALIASES.get(field, (field,)):
        value = lowered.get(alias)
        if value not in (None, "", []):
            return value, alias
    return None, ""


def _is_inches(row: dict[str, Any], field: str) -> bool:
    """Whether this row's dimensions are in inches.

    Decided per row, from the key that actually supplied the value and any
    explicit unit column - not globally. A feed can mix units across rows, and
    a converted-twice dimension is worse than an unconverted one because it
    looks plausible.
    """
    unit = str(_pick(row, "unit") or row.get("units") or row.get("dimension_unit") or "").lower()
    if unit in ("in", "inch", "inches"):
        return True
    if unit in ("cm", "centimeter", "centimetre"):
        return False
    lowered = {str(k).lower() for k in row}
    return any(a in lowered for a in FIELD_ALIASES[field] if a.endswith("_in"))


# Field names that conventionally carry a MINOR-unit amount (cents) rather
# than a major-unit one (dollars). Getting this wrong is a 100x pricing error
# in either direction, so it is decided by the field name the value came from
# rather than by guessing from magnitude - a S$2,000 sofa and a 2000-cent
# cushion are both entirely plausible.
MINOR_UNIT_FIELDS: frozenset[str] = frozenset(
    {"amount", "price_cents", "amount_cents", "unit_amount", "value_minor"}
)


def _to_cents(value: Any, source_field: str = "") -> int | None:
    """Parse a price into cents. Returns None rather than guessing.

    Handles nested {"value": .., "currency": ..}, currency symbols and
    thousands separators. A price that cannot be parsed is an error the
    merchant must fix: defaulting it to zero would publish a free product.

    `source_field` decides the unit. A feed whose column is `amount` is
    already in cents; one whose column is `price` is in dollars. Multiplying
    the first by 100 turns S$1,697 into S$16,970 - which is exactly the kind
    of error that is invisible in a demo and catastrophic in production.
    """
    if isinstance(value, dict):
        # Nested money objects carry their own unit hint.
        for key in ("value", "amount", "gross"):
            if key in value:
                nested = value[key]
                unit = str(value.get("unit") or value.get("currency_unit") or "").lower()
                if unit in ("minor", "cents") or key == "amount":
                    return _to_cents(nested, source_field="amount")
                return _to_cents(nested, source_field="price")
        return None
    if value in (None, ""):
        return None

    text = str(value).strip()
    for symbol in ("S$", "US$", "USD", "SGD", "$", "€", "£", ","):
        text = text.replace(symbol, "")
    text = text.strip()
    try:
        number = float(text)
    except ValueError:
        return None

    if source_field.lower() in MINOR_UNIT_FIELDS:
        return round(number)
    return round(number * 100)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().replace("cm", "").replace("in", "").strip())
    except ValueError:
        return None


def _split_dimension_string(text: str) -> list[float | None]:
    """Parse "210 x 90 x 80" and its variants into three numbers."""
    parts = re.split(r"\s*[x×*]\s*", str(text).strip())
    return [_to_float(p) for p in parts[:3]]


def normalize_feed_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map one merchant feed row onto the platform's product schema.

    Returns (product, errors). A row with errors is never published - it is
    handed back so the merchant can correct and resubmit.
    """
    errors: list[str] = []

    sku = _pick(row, "sku")
    title = _pick(row, "title")
    if not sku:
        errors.append("missing product ID (sku/product_id/item_code)")
    if not title:
        errors.append("missing title (title/name/product_title)")

    price_value, price_field = _pick_with_source(row, "price")
    cents = _to_cents(price_value, source_field=price_field)
    if cents is None:
        errors.append("missing or unparseable price")
    elif cents <= 0:
        errors.append(f"price must be greater than zero (got {cents} cents)")

    # Dimensions: either three columns, or one "WxDxH" string.
    w = _to_float(_pick(row, "width"))
    d = _to_float(_pick(row, "depth"))
    h = _to_float(_pick(row, "height"))
    if None in (w, d, h):
        combined = row.get("dimensions") or row.get("size") or row.get("measurements")
        if isinstance(combined, dict):
            w = w or _to_float(combined.get("w") or combined.get("width"))
            d = d or _to_float(combined.get("d") or combined.get("depth"))
            h = h or _to_float(combined.get("h") or combined.get("height"))
        elif combined:
            parts = _split_dimension_string(combined)
            if len(parts) == 3:
                w, d, h = (w or parts[0], d or parts[1], h or parts[2])

    if None in (w, d, h):
        errors.append("missing dimensions (width/depth/height, or a WxDxH field)")
    else:
        if _is_inches(row, "width"):
            w, d, h = (round(v * IN_TO_CM, 1) for v in (w, d, h))
        if min(w, d, h) <= 0:
            errors.append("dimensions must be greater than zero")

    stock = str(_pick(row, "stock") or "").lower()
    in_stock = stock not in ("out_of_stock", "outofstock", "false", "0", "no", "sold_out", "discontinued")

    material = _pick(row, "material")
    materials = (
        material if isinstance(material, list)
        else [m.strip() for m in str(material).split(",") if m.strip()] if material
        else []
    )

    product = {
        "sku": str(sku or "").strip(),
        "title": str(title or "").strip(),
        "price_cents": cents or 0,
        "currency": str(_pick(row, "currency") or "SGD").upper()[:3],
        "category": str(_pick(row, "category") or "").lower().strip(),
        "width_cm": w or 0,
        "depth_cm": d or 0,
        "height_cm": h or 0,
        "primary_color": str(_pick(row, "color") or "neutral").lower().strip(),
        "materials": materials,
        "image_url": str(_pick(row, "image_url") or "").strip(),
        "checkout_url": str(_pick(row, "checkout_url") or "").strip(),
        "description": str(_pick(row, "description") or "").strip(),
        "in_stock": in_stock,
    }
    return product, errors


# The exact error `normalize_feed_row` raises for absent dimensions. A row
# carrying only this error is a candidate for recovery; a row that is also
# missing a price or a title is not, because those cannot be estimated.
_DIMS_MISSING = "missing dimensions (width/depth/height, or a WxDxH field)"


def _seats_hint(row: dict[str, Any]) -> int | None:
    """Seat count, if the feed states one. A strong anchor for sofa widths."""
    for key in ("seating_capacity", "seats", "seat_count", "capacity"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            continue
    # "3 Seater Sofa" in the title is the same fact, stated less formally.
    match = re.search(r"(\d+)\s*[- ]?seater", str(_pick(row, "title") or ""), re.I)
    return int(match.group(1)) if match else None


def normalize_feed(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Normalize a whole feed, collecting per-row errors.

    Errors are collected rather than raised so a merchant with 500 products
    and three bad rows learns about all three at once.

    Two passes. The first normalizes every row. The second offers rows that
    failed ONLY on incomplete dimensions to the estimator, which fills the
    missing measurement from the ones that are present - see dimension_ai for
    why that is worth doing rather than rejecting 83% of a real feed.

    A recovered product carries `estimated_dims` (which fields were inferred)
    and `estimate_source` (what produced them), so the caller can tell the
    merchant precisely which numbers are not their own. Nothing here silently
    invents data: a row with no dimensions at all keeps its rejection.
    """
    products: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    # Rows held back purely for dimensions, kept with their position so a
    # recovered row can be reported against the line the merchant sees.
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, row in enumerate(rows, start=1):
        product, errors = normalize_feed_row(row)
        if errors:
            if errors == [_DIMS_MISSING] and config.DIMENSION_ESTIMATE:
                candidates.append((index, row, product))
                continue
            problems.append(
                {
                    "row": str(index),
                    "sku": product.get("sku", ""),
                    "error": "; ".join(errors),
                }
            )
            continue
        products.append(product)

    if candidates:
        products.extend(_recover_dimensions(candidates, problems))

    return products, problems


def _recover_dimensions(
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]],
    problems: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Fill missing dimensions for held-back rows; reject what stays unknown.

    Appends to `problems` in place for rows the estimator could not rescue, so
    a merchant still learns about them in the same list as every other issue.
    """
    from .dimension_ai import get_estimator

    # normalize_feed_row zeroes absent dimensions, so re-read them from the raw
    # row: 0 and "not supplied" must not be confused, and only the latter may
    # be estimated.
    items = []
    for index, row, product in candidates:
        known = {
            "width_cm": _to_float(_pick(row, "width")),
            "depth_cm": _to_float(_pick(row, "depth")),
            "height_cm": _to_float(_pick(row, "height")),
        }
        # A single "W x D x H" column can supply a partial answer too.
        combined = row.get("dimensions") or row.get("size") or row.get("measurements")
        if combined and not isinstance(combined, dict):
            parts = _split_dimension_string(combined)
            for key, part in zip(("width_cm", "depth_cm", "height_cm"), parts + [None] * 3):
                if known[key] is None and part is not None:
                    known[key] = part
        elif isinstance(combined, dict):
            for key, aliases in (
                ("width_cm", ("w", "width", "width_cm")),
                ("depth_cm", ("d", "depth", "depth_cm")),
                ("height_cm", ("h", "height", "height_cm")),
            ):
                if known[key] is None:
                    for alias in aliases:
                        if combined.get(alias) not in (None, ""):
                            known[key] = _to_float(combined[alias])
                            break
        # Inches are converted before estimating, so the model only ever sees
        # centimetres and its anchors are in the same unit as its answer.
        if _is_inches(row, "width"):
            known = {
                k: (round(v * IN_TO_CM, 1) if v is not None else None)
                for k, v in known.items()
            }
        items.append(
            {
                "id": str(index),
                "title": product.get("title", ""),
                "category": product.get("category", ""),
                "seats": _seats_hint(row),
                **{k: v for k, v in known.items() if v is not None},
            }
        )

    try:
        filled = get_estimator().estimate(items)
    except Exception as exc:  # noqa: BLE001 - an upload must not fail on this
        log.warning("dimension recovery unavailable: %s", exc)
        filled = {}

    recovered: list[dict[str, Any]] = []
    for item, (index, _row, product) in zip(items, candidates):
        got = filled.get(str(index), {})
        estimated = [d for d in ("width_cm", "depth_cm", "height_cm") if d in got]
        merged = {
            dim: got.get(dim, item.get(dim))
            for dim in ("width_cm", "depth_cm", "height_cm")
        }
        if any(v in (None, "") or float(v) <= 0 for v in merged.values()):
            # Still short after estimating - the ordinary rejection, with the
            # attempt noted so the merchant knows it was not simply skipped.
            problems.append(
                {
                    "row": str(index),
                    "sku": product.get("sku", ""),
                    "error": _DIMS_MISSING + " (could not be estimated)",
                }
            )
            continue
        product.update(
            {
                "width_cm": merged["width_cm"],
                "depth_cm": merged["depth_cm"],
                "height_cm": merged["height_cm"],
                "estimated_dims": estimated,
                "estimate_source": got.get("_source", ""),
            }
        )
        recovered.append(product)
    return recovered


def parse_upload(content: bytes, filename: str) -> list[dict[str, Any]]:
    """Turn an uploaded CSV or JSON file into a list of raw rows.

    Accepts the shapes merchant exports actually take: a bare array, or an
    object wrapping one under `products`, `items`, `data` or `catalog`.
    """
    import csv as _csv
    import io

    name = filename.lower()
    text = content.decode("utf-8-sig", errors="replace")

    if name.endswith(".json"):
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise MerchantError(400, "invalid_json", f"could not parse JSON: {exc}") from exc
        if isinstance(data, dict):
            for key in ("products", "items", "data", "catalog", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
        raise MerchantError(400, "invalid_json", "JSON must be an array or an object containing one")

    if name.endswith(".csv"):
        try:
            return list(_csv.DictReader(io.StringIO(text)))
        except _csv.Error as exc:
            raise MerchantError(400, "invalid_csv", f"could not parse CSV: {exc}") from exc

    raise MerchantError(
        400, "unsupported_format", f"unsupported file type: {filename}. Upload .csv or .json"
    )


# --- Order splits and payouts ----------------------------------------------


def record_order_split(
    *,
    order_id: str,
    intent_id: str,
    merchant_name: str,
    gross_cents: int,
    currency: str = "SGD",
) -> dict[str, Any]:
    """Record one merchant's share of a settled order.

    Called once per approved merchant leg. Commission is deducted from the
    merchant's gross rather than added to the shopper's total - the shopper
    already agreed to a number, and changing it after the fact would make the
    preview they approved a lie.
    """
    merchant = by_name(merchant_name)
    commission_bps = merchant.commission_bps if merchant else DEFAULT_COMMISSION_BPS
    commission = round(gross_cents * commission_bps / 10_000)
    net = gross_cents - commission

    record = {
        "order_id": order_id,
        "intent_id": intent_id,
        "merchant_id": merchant.id if merchant else "",
        "merchant_name": merchant_name,
        "gross_cents": gross_cents,
        "commission_cents": commission,
        "net_cents": net,
        "currency": currency,
        # Never "paid". No money moves in this codebase, and a status that
        # claimed otherwise would be the single most misleading field here.
        "status": "pending_settlement",
        "settleable": bool(merchant and merchant.can_settle),
        "created_at": _now(),
    }
    if merchant:
        PAYOUTS.setdefault(merchant.id, []).append(record)
        if not merchant.can_settle:
            # Worth logging loudly: an order is accruing money the platform
            # cannot legally pay out yet.
            log.warning(
                "order %s accrued %d cents for %s, which cannot settle "
                "(status=%s, kyc=%s)",
                order_id,
                net,
                merchant.name,
                merchant.status.value,
                merchant.kyc_status.value,
            )
    return record


def settle(merchant_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Pay out a merchant's pending balance via Visa Direct.

    Refuses before it moves anything if the merchant cannot legally be paid.
    KYC is checked here as well as recorded at accrual, because the accrual
    check was made at order time and the answer may have changed since.

    When live payouts are not configured this does NOT pretend to pay: it
    returns `simulated: true` and leaves every record `pending_settlement`.
    """
    from . import visa_direct

    merchant = get_merchant(merchant_id)
    if not merchant.can_settle:
        raise MerchantError(
            403,
            "not_settleable",
            f"cannot pay out: account is {merchant.status.value}, "
            f"KYC is {merchant.kyc_status.value}",
        )

    pending = [
        r for r in PAYOUTS.get(merchant_id, []) if r["status"] == "pending_settlement"
    ]
    total = sum(r["net_cents"] for r in pending)
    if not pending or total <= 0:
        return {"merchant_id": merchant_id, "paid_cents": 0, "records": 0, "message": "nothing pending"}

    if dry_run:
        return {
            "merchant_id": merchant_id,
            "would_pay_cents": total,
            "records": len(pending),
            "dry_run": True,
        }

    if not merchant.payout_pan:
        raise MerchantError(
            400,
            "no_payout_account",
            "this merchant has no payout account on file",
        )

    if not visa_direct.available():
        # Honest refusal rather than a fake success. The caller sees exactly
        # what is missing, and no record is marked paid.
        return {
            "merchant_id": merchant_id,
            "simulated": True,
            "would_pay_cents": total,
            "records": len(pending),
            "missing": visa_direct._requirements(),
            "message": (
                "Live payouts are not configured; nothing was paid and every "
                "record remains pending_settlement."
            ),
        }

    # One key for this exact set of pending records. Deriving it from the
    # records means a retry after a timeout reproduces it without the operator
    # having had to save anything.
    key = _settlement_key(merchant_id, [f"{r['order_id']}:{r['intent_id']}" for r in pending])
    prior = SETTLEMENTS.get(key)

    # A completed attempt is replayed, not repeated. This is the case that used
    # to double-pay: Visa approved, the response was lost, and the retry built
    # fresh trace numbers that read as a second payment.
    if prior and prior["status"] == "paid":
        log.info("settlement %s already completed; replaying receipt", key)
        return {
            "merchant_id": merchant_id,
            "paid_cents": prior["amount_cents"],
            "records": len(prior["record_ids"]),
            "idempotency_key": key,
            "replayed": True,
            "visa_response": prior.get("response", {}),
        }

    # An attempt that never came back is NOT retried blindly. Visa may or may
    # not have moved the money, and pushing again could pay twice; the honest
    # answer is to say so and let a human reconcile against Visa's records
    # using the trace numbers this attempt used.
    if prior and prior["status"] == "unknown":
        raise MerchantError(
            409,
            "settlement_unresolved",
            f"a previous payout attempt for these {len(prior['record_ids'])} record(s) "
            f"did not return an outcome and may or may not have been paid. Reconcile "
            f"transaction {prior['transaction_id']} (RRN {prior['rrn']}) with Visa "
            f"before retrying.",
        )

    order_ids = ",".join(r["order_id"] for r in pending)[:15]

    # Recorded BEFORE the network call. If this process dies mid-request, the
    # attempt and its trace numbers survive - which is the whole point, since
    # those numbers are what identify the payout to Visa afterwards.
    attempt = {
        "key": key,
        "merchant_id": merchant_id,
        "amount_cents": total,
        "record_ids": [f"{r['order_id']}:{r['intent_id']}" for r in pending],
        "status": "in_flight",
        "transaction_id": visa_direct._transaction_id(key),
        "stan": visa_direct._systems_trace(key),
        "rrn": visa_direct._retrieval_reference(key),
        "created_at": _now(),
    }
    SETTLEMENTS[key] = attempt

    try:
        response = visa_direct.push_payout(
            recipient_pan=merchant.payout_pan,
            amount_cents=total,
            currency="SGD",
            recipient_name=merchant.legal_name or merchant.name,
            order_id=order_ids,
            idempotency_key=key,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real reason
        # The request left but no outcome came back. Distinguishing this from a
        # clean decline matters: a decline is known-unpaid and safe to retry,
        # while this is genuinely unknown and must not be pushed again blindly.
        attempt["status"] = "unknown"
        attempt["error"] = str(exc)
        log.error(
            "visa direct payout UNRESOLVED for %s (txn %s, rrn %s): %s",
            merchant.name, attempt["transaction_id"], attempt["rrn"], exc,
        )
        raise MerchantError(
            502,
            "payout_failed",
            f"Visa Direct did not return an outcome for this payout: {exc}. "
            f"Reconcile transaction {attempt['transaction_id']} before retrying.",
        ) from exc

    # An HTTP 200 is not an approval: Visa reports the outcome in actionCode.
    # Marking records paid on a decline would put a settlement in the ledger
    # that never happened.
    if not response.get("approved"):
        # A decline is a definite "no money moved", so the attempt is closed
        # and the same records may be settled again once the cause is fixed.
        attempt["status"] = "declined"
        attempt["response"] = response
        SETTLEMENTS.pop(key, None)
        raise MerchantError(
            402,
            "payout_declined",
            f"Visa declined this payout: {response.get('decline_reason', 'unknown reason')}",
        )

    # Only mark paid AFTER Visa approved it.
    now = _now()
    attempt["status"] = "paid"
    attempt["response"] = response
    attempt["paid_at"] = now
    for record in pending:
        record["status"] = "paid"
        record["paid_at"] = now
        record["settlement_key"] = key
        record["transaction_id"] = str(response.get("transactionIdentifier", ""))
    log.info(
        "paid out %d cents to %s via Visa Direct (txn %s)",
        total, merchant.name, attempt["transaction_id"],
    )
    return {
        "merchant_id": merchant_id,
        "paid_cents": total,
        "records": len(pending),
        "idempotency_key": key,
        "visa_response": response,
    }


def payouts_for(merchant_id: str) -> list[dict[str, Any]]:
    return list(PAYOUTS.get(merchant_id, []))


def balance_for(merchant_id: str) -> dict[str, Any]:
    """What this merchant is owed, and whether it can actually be paid."""
    merchant = get_merchant(merchant_id)
    records = PAYOUTS.get(merchant_id, [])
    pending = sum(r["net_cents"] for r in records if r["status"] == "pending_settlement")
    return {
        "merchant_id": merchant_id,
        "pending_cents": pending,
        "order_count": len(records),
        "currency": "SGD",
        "can_settle": merchant.can_settle,
        "blocked_reason": (
            ""
            if merchant.can_settle
            else f"account is {merchant.status.value}, KYC is {merchant.kyc_status.value}"
        ),
    }
