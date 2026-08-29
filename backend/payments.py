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

  0. MANDATE    the agent presents a signed, scoped token. Anything outside
                its scope is refused here, before pricing means anything.
  1. INTENT     the agent prices everything and stops. Nothing is chargeable.
  2. PREVIEW    the user reads exactly what will be charged, by whom, on which
                card, before anything is authorized.
  3. STEP-UP    for anything unusual, the user proves they are present - with
                a device biometric bound to a hardware key, not a code.
  4. AUTHORIZE  the user, never the agent, releases the charge - once, under
                an idempotency key, against the total they actually read.
  5. RECEIPT    an audit trail of what was agreed and when.

The invariant every function here preserves: an agent can reach state 1 and
nothing further. States 3-5 require a user action that the agent cannot
synthesise.

Layered on top, the Visa Agentic Payments Stack (see `vts.py` and
`webauthn.py`) adds three properties this rail could not have on its own:

  SCOPE       The agent spends against a network token, never a PAN, and that
              token carries a mandate - category lock, per-purchase cap,
              cumulative cap, merchant limit, expiry. Enforced in
              `create_intent` and re-checked in `authorize`.
  PRESENCE    Step-up is a FIDO2 passkey assertion. The signature is bound to
              this intent AND this amount, so proof-of-presence for a small
              purchase cannot authorize a large one.
  REVOCABILITY The user can revoke the mandate at any moment without touching
              the card. A revocation between preview and authorization stops
              the charge, which is why the mandate is checked twice.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time

from . import merchants
from . import vts
from . import webauthn
from .models import (
    AuthorizationReceipt,
    Cart,
    CardNetwork,
    CartLine,
    MandateEvaluation,
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
#
# This must name merchants the catalog actually sells, or the signal fires on
# every purchase and step-up stops meaning anything - a challenge that always
# appears is one people learn to click through. The catalog is currently all
# IKEA; a second supplier is "new" until it is added here.
KNOWN_MERCHANTS: frozenset[str] = frozenset({"IKEA"})

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
# (intent_id, idempotency_key) -> receipt. Replaying a key against the same
# intent returns the original outcome instead of charging again. Keying on the
# pair rather than the key alone keeps a client that reuses a key across two
# purchases from being handed the wrong receipt.
RECEIPTS: dict[tuple[str, str], AuthorizationReceipt] = {}
# intent_id -> the exact mandate credential the agent presented when pricing.
# Held here rather than on the PaymentIntent because the intent is serialized
# to the client and this is a bearer credential. Keeping the original string
# lets `authorize` re-verify what the agent actually holds, rather than
# re-signing the server's current state and checking its own arithmetic.
PRESENTED_MANDATES: dict[str, str] = {}


def _now() -> float:
    return time.time()


def _money(cents: int) -> str:
    # Explicit "S$": these strings appear on the authorization screen, where a
    # bare "$" invites the user to read Singapore prices as US dollars.
    return f"S${cents / 100:,.2f}"


# --- Pricing ---------------------------------------------------------------


def _price_merchant(
    merchant: str,
    lines: list[CartLine],
    method: PaymentMethod,
    token: vts.NetworkToken | None = None,
) -> MerchantCharge:
    """Fully price one merchant's basket, shipping and tax included.

    When an agent token is presented, the charge records the token's last4 and
    the merchant's category code rather than only the funding card - because
    those are what actually travel on the authorization, and what the mandate
    is checked against.
    """
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
        mcc=vts.mcc_for_merchant(merchant) or "",
        category_label=vts.label_for_mcc(vts.mcc_for_merchant(merchant) or ""),
        token_last4=token.token_last4 if token else "",
    )


def _assess_risk(
    charges: list[MerchantCharge],
    total_cents: int,
    budget_cents: int,
    methods: dict[str, PaymentMethod],
    initiated_by: str,
    mandate: MandateEvaluation | None = None,
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

    # Mandate signals. A violation is reported as its own signal rather than
    # folded into the step-up decision, because the two are different kinds of
    # answer: a step-up says "prove it is you", a violation says "you already
    # told me not to". No amount of verification clears the second.
    if mandate is not None:
        if mandate.violations:
            for violation in mandate.violations:
                signals.append(
                    RiskSignal(
                        code="mandate_violation",
                        detail=violation["detail"],
                        triggers_step_up=False,
                    )
                )
        else:
            signals.append(
                RiskSignal(
                    code="mandate_scoped",
                    detail=(
                        f"Within the agent's mandate: {_money(mandate.amount_cents)} "
                        f"of the {_money(mandate.per_transaction_cap_cents)} "
                        f"per-purchase limit, "
                        f"{_money(mandate.remaining_cents)} left of the total budget. "
                        "Furniture & home decor only."
                    ),
                )
            )
        signals.append(
            RiskSignal(
                code="token_presented",
                detail=(
                    "Paying with a single-use Visa agent token, not your card "
                    "number. Your 16-digit card is never shared with any "
                    "merchant, and you can revoke this agent's access without "
                    "cancelling the card."
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


def _evaluate_mandate(
    mandate_credential: str, merchants: list[str], total_cents: int
) -> tuple[MandateEvaluation | None, vts.NetworkToken | None, str]:
    """Measure a proposed purchase against the agent's mandate.

    A malformed or forged credential is not treated as "no mandate" - that
    would make presenting garbage strictly better for an attacker than
    presenting nothing, since an unscoped purchase faces no caps at all. It is
    a violation, and it blocks.
    """
    if not mandate_credential:
        # No credential presented. If the user has granted a live mandate, the
        # agent is REQUIRED to spend under it: otherwise omitting the
        # credential would be a trivial bypass, and every cap the user set
        # would be advisory. Falling back to the active mandate rather than
        # refusing outright keeps an honest client that forgot to attach it
        # working, while still enforcing the caps.
        active = next(
            (t for t in vts.list_tokens() if t.status == vts.TokenStatus.ACTIVE),
            None,
        )
        if active is None:
            return None, None, ""
        mandate_credential = vts.sign_mandate(active.mandate, active.token_id)
    try:
        check = vts.evaluate(
            mandate_credential=mandate_credential,
            merchants=merchants,
            amount_cents=total_cents,
        )
    except vts.MandateViolation as exc:
        return (
            MandateEvaluation(
                ok=False,
                amount_cents=total_cents,
                violations=[{"code": exc.code, "detail": exc.detail}],
            ),
            None,
            mandate_credential,
        )

    token = vts.TOKENS.get(check.token_id)
    return (
        MandateEvaluation(
            ok=check.ok,
            token_id=check.token_id,
            amount_cents=check.amount_cents,
            per_transaction_cap_cents=check.per_transaction_cap_cents,
            cumulative_cap_cents=check.cumulative_cap_cents,
            spent_cents=check.spent_cents,
            remaining_cents=check.remaining_cents,
            allowed_mccs=check.allowed_mccs,
            merchant_mccs=check.merchant_mccs,
            expires_at=check.expires_at,
            violations=check.violations,
        ),
        token,
        mandate_credential,
    )


def create_intent(
    lines: list[CartLine],
    *,
    session_id: str | None = None,
    budget_cents: int = 0,
    payment_method_ids: dict[str, str] | None = None,
    initiated_by: str = "agent",
    mandate_credential: str = "",
) -> PaymentIntent:
    """Price a purchase and park it awaiting human approval.

    This is everything the agent is allowed to do. The returned intent is
    inert: it names a total, but nothing in this module will move against it
    without a subsequent user-supplied authorization.

    When a mandate is presented, it is evaluated here and the result is
    attached to the intent. Pricing still completes on a violation rather than
    raising: the user is better served by a preview that shows exactly which
    guardrail fired and by how much than by an opaque refusal. What a
    violation does is set `mandate_blocked`, which `authorize` refuses on.
    """
    assignments = payment_method_ids or {}

    grouped: dict[str, list[CartLine]] = {}
    for line in lines:
        grouped.setdefault(line.merchant, []).append(line)

    merchants = sorted(grouped)

    # Price once with no token so the mandate is evaluated against the real
    # total (shipping and tax included). A cap checked against the subtotal
    # would let an agent slip past it on delivery charges.
    charges: list[MerchantCharge] = []
    methods: dict[str, PaymentMethod] = {}
    for merchant in merchants:
        method = get_method(assignments.get(merchant, "")) or default_method()
        methods[merchant] = method
        charges.append(_price_merchant(merchant, grouped[merchant], method))

    subtotal = sum(c.subtotal_cents for c in charges)
    shipping = sum(c.shipping_cents for c in charges)
    tax = sum(c.tax_cents for c in charges)
    total = subtotal + shipping + tax

    # `effective_credential` is what was actually evaluated - the presented one,
    # or the active mandate synthesised when the agent omitted it. Either way it
    # is the string `authorize` must later re-verify.
    mandate, token, effective_credential = _evaluate_mandate(
        mandate_credential, merchants, total
    )

    # Stamp the token onto each leg now that we know which one was presented.
    if token is not None:
        for charge in charges:
            charge.token_last4 = token.token_last4

    risk, step_up = _assess_risk(
        charges, total, budget_cents, methods, initiated_by, mandate
    )

    blocked = mandate is not None and not mandate.ok
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
        agent_token_id=token.token_id if token else "",
        mandate=mandate,
        mandate_blocked=blocked,
        # A passkey is the step-up whenever one is registered. The OTP path
        # survives only for a device with no authenticator, because a flow
        # that silently downgrades to a weaker factor is one an attacker
        # simply asks for.
        step_up_method="passkey" if webauthn.CREDENTIALS else "sms_otp",
    )
    INTENTS[intent.id] = intent
    if effective_credential:
        PRESENTED_MANDATES[intent.id] = effective_credential
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
        # This endpoint is the OTP path specifically. The passkey step-up does
        # not go through here, so labelling it "passkey" because the model
        # defaults that way would misreport what the user is being asked for.
        method="sms_otp",
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


# Passkey assertions accepted for an intent: assertion_id -> record. A
# separate store from the OTP challenges because the two prove different
# things and expire on different clocks.
ASSERTIONS: dict[str, dict] = {}

# How long a passkey assertion counts as evidence the user is present. Short:
# the whole value of a biometric is that it was performed *now*, and an
# assertion that stayed valid for an hour would be a session token wearing a
# biometric's clothes.
ASSERTION_TTL_SECONDS = 3 * 60


def record_assertion(
    result: webauthn.AssertionResult,
    intent_id: str,
    amount_cents: int,
    purpose: str = "payment",
) -> None:
    """Bank a verified passkey assertion against one intent and amount.

    `purpose` is recorded and checked on use. Without it, the biometric a user
    performed to approve one small purchase could be presented to mint a
    standing spending mandate instead - the two ceremonies ask the user for
    the same gesture but mean entirely different things, so the server must
    distinguish what each one was for.
    """
    ASSERTIONS[result.assertion_id] = {
        "purpose": purpose,
        "intent_id": intent_id,
        "amount_cents": amount_cents,
        "credential_id": result.credential_id,
        "verified_at": result.verified_at,
        "expires_at": result.verified_at + ASSERTION_TTL_SECONDS,
        "used": False,
    }


def _consume_assertion(assertion_id: str, intent_id: str, amount_cents: int) -> dict:
    """Spend a passkey assertion. Single-use, intent-bound, amount-bound.

    Single-use matters as much as the binding: without it, one biometric would
    authorize every subsequent charge for as long as it stayed fresh, which is
    precisely the standing authority the mandate exists to prevent.
    """
    record = ASSERTIONS.get(assertion_id)
    if record is None:
        raise PaymentError(401, "no verified passkey for this payment")
    if record.get("purpose") != "payment":
        raise PaymentError(
            403, "that identity check was not performed to approve a payment"
        )
    if record["used"]:
        raise PaymentError(409, "this authorization was already used")
    if _now() > record["expires_at"]:
        raise PaymentError(
            410, "your identity check expired - please verify with Face ID again"
        )
    if record["intent_id"] != intent_id:
        raise PaymentError(403, "this authorization was for a different payment")
    if record["amount_cents"] != amount_cents:
        raise PaymentError(
            403,
            "the total changed after you verified. Please review and verify again.",
        )
    record["used"] = True
    return record


def consume_provisioning_assertion(assertion_id: str) -> dict:
    """Spend an assertion that was performed to grant a mandate.

    Kept beside `_consume_assertion` so both ceremonies burn their proof under
    the same rules, and so neither can spend the other's.
    """
    record = ASSERTIONS.get(assertion_id)
    if record is None:
        raise PaymentError(401, "that identity check is not valid")
    if record.get("purpose") != "provisioning":
        raise PaymentError(
            403,
            "that identity check was performed to approve a payment, not to "
            "grant this agent a spending mandate. Verify again.",
        )
    if record["used"]:
        raise PaymentError(409, "that identity check was already used")
    if _now() > record["expires_at"]:
        raise PaymentError(410, "that identity check expired - verify again")
    record["used"] = True
    return record


def _verified(intent_id: str) -> bool:
    """Whether a still-valid step-up challenge was answered for this intent.

    Expiry is checked here too: a challenge answered an hour ago is not
    evidence that a human is present now, and without this a verification
    would hold for the lifetime of the process.
    """
    now = _now()
    if any(
        r["intent_id"] == intent_id and r["verified"] and now <= r["expires_at"]
        for r in CHALLENGES.values()
    ):
        return True
    # A fresh, unspent passkey assertion is the stronger form of the same
    # evidence and satisfies the same requirement.
    #
    # `purpose` is checked here for the same reason `_consume_assertion`
    # checks it: the biometric a user performed to grant the agent a spending
    # mandate is not approval of a payment, even though both ceremonies ask
    # for the identical gesture. Without this the two are interchangeable on
    # the no-assertion_id path, and the separation the other function enforces
    # would rest on an id-naming convention in main.py rather than on a check.
    return any(
        r.get("purpose") == "payment"
        and r["intent_id"] == intent_id
        and not r["used"]
        and now <= r["expires_at"]
        for r in ASSERTIONS.values()
    )


def authorize(
    intent_id: str,
    idempotency_key: str,
    confirmed_total_cents: int,
    assertion_id: str = "",
) -> AuthorizationReceipt:
    """Charge the intent. The only function in this module that moves money.

    Every refusal below is a case where charging would mean the user paid for
    something other than what they agreed to.
    """
    # Replay first: a retried request must return the original outcome rather
    # than authorizing a second time.
    #
    # Scoped to (intent, key), not the key alone. A client that reuses one key
    # across two different purchases would otherwise be handed the FIRST
    # purchase's receipt for the second - reporting success for an order that
    # was never authorized, and leaving the real one uncharged.
    existing = RECEIPTS.get((intent_id, idempotency_key))
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

    # A mandate violation is fatal and cannot be verified away. Re-checked
    # here rather than trusted from pricing time, because the user may have
    # revoked the mandate in the seconds between reading the preview and
    # authorizing - and a revocation that only took effect on the next
    # purchase would not be a revocation.
    if intent.agent_token_id:
        try:
            token = vts.get_token(intent.agent_token_id)
        except vts.MandateViolation as exc:
            raise PaymentError(exc.status, exc.detail) from exc
        if token.status != vts.TokenStatus.ACTIVE:
            raise PaymentError(
                403,
                f"this agent's spending mandate is {token.status.value}"
                + (
                    f" ({token.mandate.revocation_reason})"
                    if token.mandate.revocation_reason
                    else ""
                )
                + ". Your card itself is unaffected.",
            )
        # Re-verify the credential the AGENT PRESENTED, not one re-signed from
        # current server state. Re-signing would make the signature check a
        # no-op - the server verifying an HMAC it computed a line earlier - and
        # would silently pick up any later widening of the in-memory mandate
        # instead of the scope the user actually approved.
        presented = PRESENTED_MANDATES.get(intent.id)
        if not presented:
            raise PaymentError(
                403,
                "this purchase was priced against an agent mandate that is no "
                "longer available. Please re-price the order.",
            )
        try:
            recheck = vts.evaluate(
                mandate_credential=presented,
                merchants=[c.merchant for c in intent.charges],
                amount_cents=intent.total_cents,
            )
        except vts.MandateViolation as exc:
            raise PaymentError(exc.status, exc.detail) from exc
        if not recheck.ok:
            intent.mandate_blocked = True
            raise PaymentError(
                403,
                "outside the agent's mandate: "
                + "; ".join(v["detail"] for v in recheck.violations),
            )
    elif intent.mandate_blocked:
        raise PaymentError(
            403,
            "this purchase is outside the spending mandate you granted the agent.",
        )

    # Proof of presence. A passkey assertion is consumed here - single-use and
    # bound to this intent and this amount - so that the biometric the user
    # performed authorizes this charge and nothing else.
    #
    # `require_user_presence` is a claim the mandate makes to the user, so it
    # has to be a claim the rail keeps. Without this branch an agent-initiated
    # purchase that happened to trip no risk signal - a familiar merchant, a
    # small amount, inside budget - would charge with no human anywhere in the
    # loop, while the UI displayed "requires your approval". A guarantee shown
    # to the user and not enforced here is worse than no guarantee at all.
    presence_required = intent.requires_step_up
    if intent.agent_token_id:
        token = vts.TOKENS.get(intent.agent_token_id)
        if token is not None and token.mandate.require_user_presence:
            presence_required = True

    if assertion_id:
        _consume_assertion(assertion_id, intent_id, intent.total_cents)
    elif presence_required and not _verified(intent_id):
        raise PaymentError(
            401,
            "verify with Face ID / Touch ID before authorizing - this agent's "
            "mandate requires you to approve every purchase.",
        )

    # Claim the cumulative headroom before charging, not after. Between the
    # recheck above and the commit below there is otherwise a window in which
    # several intents priced against the same budget each pass - which is
    # exactly the split-the-order attack the cumulative cap exists to stop.
    if intent.agent_token_id:
        if not vts.reserve_spend(intent.agent_token_id, intent.total_cents):
            raise PaymentError(
                403,
                "this would take the agent past the total budget you set for it.",
            )

    intent.status = PaymentIntentStatus.AUTHORIZING

    approved = declined = 0
    for charge in intent.charges:
        method = get_method(charge.payment_method_id) or default_method()

        # Split settlement: each merchant leg gets its own single-use
        # cryptogram. One per leg, not one per order, so that a cryptogram
        # captured from the merchant that received it cannot be replayed
        # against another merchant on the same order.
        if intent.agent_token_id:
            cryptogram = vts.issue_cryptogram(
                token_id=intent.agent_token_id,
                intent_id=intent.id,
                amount_cents=charge.total_cents,
                assertion_id=assertion_id or "user_confirmed",
            )
            charge.cryptogram_id = cryptogram["cryptogram_id"]

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
            if charge.cryptogram_id:
                # Burn the cryptogram on the leg that actually settled.
                try:
                    vts.consume_cryptogram(
                        charge.cryptogram_id,
                        intent_id=intent.id,
                        amount_cents=charge.total_cents,
                    )
                except vts.MandateViolation as exc:  # pragma: no cover - defensive
                    log.warning("cryptogram refused for %s: %s", charge.merchant, exc.detail)

    order_id = f"SIM-{secrets.token_hex(4).upper()}"
    intent.order_id = order_id
    intent.status = (
        PaymentIntentStatus.SUCCEEDED if approved > 0 else PaymentIntentStatus.FAILED
    )

    # Only the approved portion counts against the mandate's cumulative cap.
    # The full total was reserved above, so give back whatever did not settle:
    # holding a reservation for money that never moved would silently shrink
    # the budget the user has left.
    if intent.agent_token_id:
        if declined > 0:
            vts.release_spend(intent.agent_token_id, declined)
        if approved > 0:
            vts.record_spend(
                intent.agent_token_id,
                amount_cents=approved,
                intent_id=intent.id,
                order_id=order_id,
            )
        else:
            # Nothing settled at all - release the remainder too.
            vts.release_spend(intent.agent_token_id, intent.total_cents - declined)

    now = _now()
    audit = [
        f"Order assembled by Room Hack agent from {len(intent.charges)} merchant(s).",
        f"Preview shown to user: {_money(intent.total_cents)} total.",
    ]
    if intent.agent_token_id and intent.mandate:
        audit.append(
            f"Agent presented scoped Visa token {intent.agent_token_id} "
            f"(presentationType=AI_AGENT); no card number was shared."
        )
        audit.append(
            f"Mandate check passed: {_money(intent.total_cents)} within the "
            f"{_money(intent.mandate.per_transaction_cap_cents)} per-purchase cap, "
            f"category locked to {', '.join(intent.mandate.allowed_mccs) or 'any'}."
        )
    if assertion_id:
        record = ASSERTIONS.get(assertion_id, {})
        audit.append(
            "User verified by device biometric (FIDO2 passkey, "
            f"credential {str(record.get('credential_id', ''))[:12]}…). "
            "Signature bound to this order and amount; no biometric data left the device."
        )
    elif intent.requires_step_up:
        audit.append("Step-up identity verification completed by user.")
    audit.append(
        f"User authorized {_money(intent.total_cents)} at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}."
    )
    for charge in intent.charges:
        method = get_method(charge.payment_method_id) or default_method()
        if charge.status == "approved":
            instrument = (
                f"agent token ···· {charge.token_last4} on {method.display}"
                if charge.token_last4
                else method.display
            )
            audit.append(
                f"{charge.merchant}: {_money(charge.total_cents)} approved on "
                f"{instrument} ({charge.auth_code})."
            )
        else:
            audit.append(
                f"{charge.merchant}: {_money(charge.total_cents)} DECLINED on "
                f"{method.display}. Not charged."
            )

    # Record each approved leg against the selling merchant. Only approved
    # legs: a declined charge earns nobody anything, and crediting one would
    # put money in a payout ledger that was never collected.
    for charge in intent.charges:
        if charge.status == "approved":
            merchants.record_order_split(
                order_id=order_id,
                intent_id=intent.id,
                merchant_name=charge.merchant,
                gross_cents=charge.total_cents,
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
    RECEIPTS[(intent.id, idempotency_key)] = receipt
    log.info(
        "order %s: %s approved, %s declined",
        order_id,
        _money(approved),
        _money(declined),
    )
    return receipt
