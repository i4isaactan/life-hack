"""Simulated card rail: intent -> verification -> per-merchant authorization.

NOTHING HERE TOUCHES A PAYMENT NETWORK. No HTTP call leaves this module, no
card number exists, and every auth code is random hex. It is a faithful model
of the *shape* of a card transaction, not a transaction.

Why model it at this depth rather than returning a fabricated receipt: this
application lets an agent assemble a purchase across several merchants on the
user's behalf. The interesting question is not whether the money moves - it
never does - but how a user grants and withholds permission for it to move.
That question only has an honest answer if the steps a real authorization has
are actually present:

  1. INTENT     the agent prices everything and stops. Nothing is chargeable.
  2. PREVIEW    the user reads exactly what will be charged, by whom, on which
                card, before anything is authorized.
  3. STEP-UP    for anything unusual, the user proves they are present.
  4. AUTHORIZE  the user, never the agent, releases the charge - once, under
                an idempotency key, against the total they actually read.
  5. RECEIPT    an audit trail of what was agreed and when.

The invariant every function here preserves: an agent can reach state 1 and
nothing further. States 3-5 require a user action that the agent cannot
synthesise.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time

from .models import (
    AuthorizationReceipt,
    Cart,
    CardNetwork,
    CartLine,
    MerchantCharge,
    PaymentIntent,
    PaymentIntentStatus,
    PaymentMethod,
    RiskSignal,
    VerificationChallenge,
)

log = logging.getLogger("roomhack.payments")

# How long a priced preview stays chargeable. Prices, stock and the user's
# intent all go stale; re-pricing is cheap and a surprise charge is not.
INTENT_TTL_SECONDS = 15 * 60
CHALLENGE_TTL_SECONDS = 5 * 60

# Flat simulated shipping per merchant, and a flat tax rate. Both fictional,
# but present because a preview that omits them lies about the total - the
# single most common way a checkout surprises someone.
SHIPPING_PER_MERCHANT_CENTS = 4900
FREE_SHIPPING_THRESHOLD_CENTS = 100_000
TAX_RATE = 0.0875


# --- Stored payment methods ------------------------------------------------
#
# Fictional cards with test-range last4s. `4242` is the universally recognised
# test Visa; using it signals "this is not real" to anyone technical who looks.

WALLET: list[PaymentMethod] = [
    PaymentMethod(
        id="pm_visa_4242",
        network=CardNetwork.VISA,
        last4="4242",
        exp_month=8,
        exp_year=2029,
        holder="A. Demo",
        label="Personal Visa",
        is_default=True,
        step_up_threshold_cents=50_000,
    ),
    PaymentMethod(
        id="pm_visa_1881",
        network=CardNetwork.VISA,
        last4="1881",
        exp_month=3,
        exp_year=2028,
        holder="A. Demo",
        label="Household Visa",
        step_up_threshold_cents=150_000,
    ),
    PaymentMethod(
        id="pm_mc_5454",
        network=CardNetwork.MASTERCARD,
        last4="5454",
        exp_month=11,
        exp_year=2027,
        holder="A. Demo",
        label="Backup Mastercard",
        step_up_threshold_cents=30_000,
    ),
]

# Merchants the user has bought from before. A first-time merchant is a real
# risk signal and the preview says so rather than treating every shop as
# equally familiar.
KNOWN_MERCHANTS: frozenset[str] = frozenset({"Nordhaus", "Cedarline"})

# The one merchant whose authorization always declines, so the flow can
# demonstrate a partial failure. A checkout that only ever succeeds teaches
# nothing about what happens when one of three charges does not go through.
ALWAYS_DECLINE_LAST4 = "5454"


def wallet() -> list[PaymentMethod]:
    return list(WALLET)


def get_method(method_id: str) -> PaymentMethod | None:
    return next((m for m in WALLET if m.id == method_id), None)


def default_method() -> PaymentMethod:
    return next((m for m in WALLET if m.is_default), WALLET[0])


# --- In-memory stores ------------------------------------------------------
#
# Process-local, like every other store in this app. An intent is a short-lived
# negotiation, not a record; the receipt is the record.

INTENTS: dict[str, PaymentIntent] = {}
CHALLENGES: dict[str, dict] = {}
# idempotency_key -> receipt. Replaying a key returns the original outcome
# instead of charging again.
RECEIPTS: dict[str, AuthorizationReceipt] = {}


def _now() -> float:
    return time.time()


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


# --- Pricing ---------------------------------------------------------------


def _price_merchant(
    merchant: str, lines: list[CartLine], method: PaymentMethod
) -> MerchantCharge:
    """Fully price one merchant's basket, shipping and tax included."""
    subtotal = sum(line.line_total_cents for line in lines)
    shipping = (
        0 if subtotal >= FREE_SHIPPING_THRESHOLD_CENTS else SHIPPING_PER_MERCHANT_CENTS
    )
    tax = round((subtotal + shipping) * TAX_RATE)
    # Deterministic per merchant so the same basket always quotes the same
    # delivery window - a preview whose ETA jitters on refresh looks made up.
    eta = 5 + (int(hashlib.sha256(merchant.encode()).hexdigest(), 16) % 16)
    return MerchantCharge(
        merchant=merchant,
        lines=lines,
        subtotal_cents=subtotal,
        shipping_cents=shipping,
        tax_cents=tax,
        total_cents=subtotal + shipping + tax,
        payment_method_id=method.id,
        eta_days=eta,
    )


def _assess_risk(
    charges: list[MerchantCharge],
    total_cents: int,
    budget_cents: int,
    methods: dict[str, PaymentMethod],
    initiated_by: str,
) -> tuple[list[RiskSignal], bool]:
    """Explain, in the user's terms, why this purchase is or is not routine.

    Returns the signals and whether they compel a step-up challenge. Every
    signal is shown to the user - including the reassuring ones - because a
    risk display that only ever appears when something is wrong trains people
    to dismiss it.
    """
    signals: list[RiskSignal] = []
    step_up = False

    if initiated_by == "agent":
        signals.append(
            RiskSignal(
                code="agent_initiated",
                detail=(
                    "This purchase was assembled by the Room Hack agent. "
                    "It cannot be charged until you authorize it yourself."
                ),
            )
        )

    # A card's own ceiling is the sharpest signal: the user set it.
    for charge in charges:
        method = methods.get(charge.merchant)
        if method and charge.total_cents > method.step_up_threshold_cents:
            step_up = True
            signals.append(
                RiskSignal(
                    code="amount_over_threshold",
                    detail=(
                        f"{_money(charge.total_cents)} to {charge.merchant} is over "
                        f"the {_money(method.step_up_threshold_cents)} limit you set "
                        f"for {method.display}."
                    ),
                    triggers_step_up=True,
                )
            )

    if len(charges) > 1:
        merchants = ", ".join(c.merchant for c in charges)
        signals.append(
            RiskSignal(
                code="multi_merchant",
                detail=(
                    f"{len(charges)} separate charges from {merchants}. These will "
                    f"appear as {len(charges)} separate lines on your statement."
                ),
            )
        )

    new = [c.merchant for c in charges if c.merchant not in KNOWN_MERCHANTS]
    if new:
        step_up = True
        signals.append(
            RiskSignal(
                code="new_merchant",
                detail=(
                    f"First time buying from {', '.join(new)}. "
                    "Verifying your identity for a new merchant."
                ),
                triggers_step_up=True,
            )
        )

    if budget_cents > 0 and total_cents > budget_cents:
        over = total_cents - budget_cents
        step_up = True
        signals.append(
            RiskSignal(
                code="over_budget",
                detail=(
                    f"{_money(over)} over the {_money(budget_cents)} budget you set "
                    "for this room."
                ),
                triggers_step_up=True,
            )
        )

    if not step_up:
        signals.append(
            RiskSignal(
                code="routine",
                detail=(
                    "Within your usual limits and merchants. Confirmation only - "
                    "no extra verification needed."
                ),
            )
        )

    return signals, step_up


def create_intent(
    lines: list[CartLine],
    *,
    session_id: str | None = None,
    budget_cents: int = 0,
    payment_method_ids: dict[str, str] | None = None,
    initiated_by: str = "agent",
) -> PaymentIntent:
    """Price a purchase and park it awaiting human approval.

    This is everything the agent is allowed to do. The returned intent is
    inert: it names a total, but nothing in this module will move against it
    without a subsequent user-supplied authorization.
    """
    assignments = payment_method_ids or {}

    grouped: dict[str, list[CartLine]] = {}
    for line in lines:
        grouped.setdefault(line.merchant, []).append(line)

    charges: list[MerchantCharge] = []
    methods: dict[str, PaymentMethod] = {}
    for merchant, merchant_lines in sorted(grouped.items()):
        method = get_method(assignments.get(merchant, "")) or default_method()
        methods[merchant] = method
        charges.append(_price_merchant(merchant, merchant_lines, method))

    subtotal = sum(c.subtotal_cents for c in charges)
    shipping = sum(c.shipping_cents for c in charges)
    tax = sum(c.tax_cents for c in charges)
    total = subtotal + shipping + tax

    risk, step_up = _assess_risk(charges, total, budget_cents, methods, initiated_by)

    now = _now()
    intent = PaymentIntent(
        id=f"pi_sim_{secrets.token_hex(8)}",
        session_id=session_id,
        status=(
            PaymentIntentStatus.REQUIRES_VERIFICATION
            if step_up
            else PaymentIntentStatus.REQUIRES_CONFIRMATION
        ),
        charges=charges,
        subtotal_cents=subtotal,
        shipping_cents=shipping,
        tax_cents=tax,
        total_cents=total,
        budget_cents=budget_cents,
        risk=risk,
        requires_step_up=step_up,
        initiated_by=initiated_by,  # type: ignore[arg-type]
        created_at=now,
        expires_at=now + INTENT_TTL_SECONDS,
    )
    INTENTS[intent.id] = intent
    log.info(
        "payment intent %s: %s across %d merchant(s), step_up=%s",
        intent.id,
        _money(total),
        len(charges),
        step_up,
    )
    return intent


# --- Lifecycle -------------------------------------------------------------


class PaymentError(Exception):
    """A refusal the user should see. Carries an HTTP status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def get_intent(intent_id: str) -> PaymentIntent:
    intent = INTENTS.get(intent_id)
    if intent is None:
        raise PaymentError(404, "unknown payment intent")
    # Expiry is checked on read rather than swept: an intent nobody looks at
    # again is harmless, and lazily expiring keeps the receipt of a completed
    # one readable indefinitely.
    if (
        intent.status
        in (
            PaymentIntentStatus.REQUIRES_CONFIRMATION,
            PaymentIntentStatus.REQUIRES_VERIFICATION,
        )
        and _now() > intent.expires_at
    ):
        intent.status = PaymentIntentStatus.EXPIRED
    return intent


def cancel_intent(intent_id: str) -> PaymentIntent:
    """Decline the purchase. Always available while nothing has been charged."""
    intent = get_intent(intent_id)
    if intent.status == PaymentIntentStatus.SUCCEEDED:
        raise PaymentError(409, "this payment has already been authorized")
    intent.status = PaymentIntentStatus.CANCELLED
    log.info("payment intent %s cancelled by user", intent.id)
    return intent


def start_verification(intent_id: str) -> VerificationChallenge:
    """Issue a step-up challenge in the shape of a 3-D Secure prompt."""
    intent = get_intent(intent_id)
    if intent.status == PaymentIntentStatus.EXPIRED:
        raise PaymentError(410, "this payment preview has expired - please re-price")
    if intent.status not in (
        PaymentIntentStatus.REQUIRES_VERIFICATION,
        PaymentIntentStatus.REQUIRES_CONFIRMATION,
    ):
        raise PaymentError(409, f"intent is {intent.status.value}")

    challenge = VerificationChallenge(
        intent_id=intent.id,
        challenge_id=f"vch_{secrets.token_hex(6)}",
        # DEMO ONLY: a real challenge sends this out of band and never returns
        # it. Returned here so the flow is runnable without a phone.
        demo_code=f"{secrets.randbelow(1_000_000):06d}",
        expires_at=_now() + CHALLENGE_TTL_SECONDS,
    )
    CHALLENGES[challenge.challenge_id] = {
        "intent_id": intent.id,
        "code": challenge.demo_code,
        "expires_at": challenge.expires_at,
        "attempts": 3,
        "verified": False,
    }
    intent.status = PaymentIntentStatus.REQUIRES_VERIFICATION
    return challenge


def verify(intent_id: str, challenge_id: str, code: str) -> PaymentIntent:
    """Check a step-up code. Wrong answers burn an attempt, not the intent."""
    intent = get_intent(intent_id)
    record = CHALLENGES.get(challenge_id)
    if record is None or record["intent_id"] != intent_id:
        raise PaymentError(404, "unknown verification challenge")
    if _now() > record["expires_at"]:
        raise PaymentError(410, "verification code expired - request a new one")
    if record["attempts"] <= 0:
        raise PaymentError(429, "too many attempts - request a new code")

    # Constant-time compare. The code is fake, but a demo people read should
    # not model the comparison wrongly.
    if not secrets.compare_digest(str(code).strip(), record["code"]):
        record["attempts"] -= 1
        raise PaymentError(
            401, f"incorrect code - {record['attempts']} attempt(s) remaining"
        )

    record["verified"] = True
    intent.status = PaymentIntentStatus.REQUIRES_CONFIRMATION
    log.info("payment intent %s: identity verified", intent.id)
    return intent


def _verified(intent_id: str) -> bool:
    return any(
        r["intent_id"] == intent_id and r["verified"] for r in CHALLENGES.values()
    )


def authorize(
    intent_id: str, idempotency_key: str, confirmed_total_cents: int
) -> AuthorizationReceipt:
    """Charge the intent. The only function in this module that moves money.

    Every refusal below is a case where charging would mean the user paid for
    something other than what they agreed to.
    """
    # Replay first: a retried request must return the original outcome rather
    # than authorizing a second time.
    existing = RECEIPTS.get(idempotency_key)
    if existing is not None:
        log.info("idempotent replay of %s -> order %s", idempotency_key, existing.order_id)
        return existing

    intent = get_intent(intent_id)

    if intent.status == PaymentIntentStatus.SUCCEEDED:
        raise PaymentError(409, "this payment has already been authorized")
    if intent.status == PaymentIntentStatus.EXPIRED:
        raise PaymentError(410, "this payment preview has expired - please re-price")
    if intent.status == PaymentIntentStatus.CANCELLED:
        raise PaymentError(409, "this payment was cancelled")

    # The user authorizes a number they read. If the intent no longer totals
    # that number, the thing they agreed to no longer exists.
    if confirmed_total_cents != intent.total_cents:
        raise PaymentError(
            409,
            f"the total changed since you reviewed it "
            f"({_money(confirmed_total_cents)} -> {_money(intent.total_cents)}). "
            "Please review the updated preview.",
        )

    if intent.requires_step_up and not _verified(intent_id):
        raise PaymentError(401, "identity verification required before authorizing")

    intent.status = PaymentIntentStatus.AUTHORIZING

    approved = declined = 0
    for charge in intent.charges:
        method = get_method(charge.payment_method_id) or default_method()
        # Deterministic simulated decline, so the partial-failure path is
        # demonstrable rather than a coin flip nobody can reproduce.
        if method.last4 == ALWAYS_DECLINE_LAST4:
            charge.status = "declined"
            charge.decline_reason = (
                f"Issuer declined {method.display} (simulated). "
                "Try a different card for this merchant."
            )
            declined += charge.total_cents
        else:
            charge.status = "approved"
            charge.auth_code = f"VISA·AUTH·{secrets.token_hex(3).upper()}"
            approved += charge.total_cents

    order_id = f"SIM-{secrets.token_hex(4).upper()}"
    intent.order_id = order_id
    intent.status = (
        PaymentIntentStatus.SUCCEEDED if approved > 0 else PaymentIntentStatus.FAILED
    )

    now = _now()
    audit = [
        f"Order assembled by Room Hack agent from {len(intent.charges)} merchant(s).",
        f"Preview shown to user: {_money(intent.total_cents)} total.",
    ]
    if intent.requires_step_up:
        audit.append("Step-up identity verification completed by user.")
    audit.append(
        f"User authorized {_money(intent.total_cents)} at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}."
    )
    for charge in intent.charges:
        method = get_method(charge.payment_method_id) or default_method()
        if charge.status == "approved":
            audit.append(
                f"{charge.merchant}: {_money(charge.total_cents)} approved on "
                f"{method.display} ({charge.auth_code})."
            )
        else:
            audit.append(
                f"{charge.merchant}: {_money(charge.total_cents)} DECLINED on "
                f"{method.display}. Not charged."
            )

    receipt = AuthorizationReceipt(
        intent_id=intent.id,
        order_id=order_id,
        status=intent.status,
        charges=intent.charges,
        total_cents=intent.total_cents,
        approved_cents=approved,
        declined_cents=declined,
        authorized_at=now,
        audit=audit,
    )
    RECEIPTS[idempotency_key] = receipt
    log.info(
        "order %s: %s approved, %s declined",
        order_id,
        _money(approved),
        _money(declined),
    )
    return receipt
