# Payments

How money is authorized, split, and settled in Room Hack — and where the
simulation stops.

This document exists because the payment path is the part of this project most
likely to be believed without being read. Every claim below is enforced in
code, in `backend/payments.py`, `backend/vts.py`, `backend/webauthn.py`,
`backend/merchants.py` and `backend/visa_direct.py`. Where something is
simulated, this says so in the same sentence.

**The one-line version:** an AI agent assembles a multi-merchant purchase, a
human authorizes it with a device biometric, and each merchant is paid on
Visa's push rail without needing a payment processor. No shopper money moves;
payouts are real encrypted requests to Visa's sandbox.

---

## The invariant

Everything here is arranged around a single rule:

> The agent may **price** a purchase. Only a human may **authorize** one.

An agent that could authorize its own purchases is not an agent with a spending
limit — it is an agent with your card. Every mechanism below exists to keep
those two acts separate, and each one is enforced server-side. A client that
skips the UI gets the same refusals.

---

## The whole path

```
SHOPPER SIDE                                          MERCHANT SIDE

  agent prices a basket                                 merchant onboards
        │                                                     │
        ▼                                                     ▼
  PaymentIntent  ───────── inert, charges nothing        uploads catalogue
        │                                                     │
        ▼                                                     ▼
  risk signals + mandate check                          products in the index
        │                                                     │
        ▼                                                     │
  passkey step-up  ◄──── bound to intent AND amount           │
        │                                                     │
        ▼                                                     │
  authorize  ─────── the only call that moves money           │
        │                                                     │
        ├──────────► per-merchant split recorded ─────────────┤
        │            (gross − commission = net)               │
        ▼                                                     ▼
     receipt                                        balance: pending_settlement
                                                              │
                                                        KYC verified
                                                              │
                                                              ▼
                                                    Visa Direct push (OCT)
                                                              │
                                                              ▼
                                                        ledger: paid
```

Two halves, joined at authorization. The shopper's charge is **simulated**. The
merchant payout is a **real encrypted request to Visa's sandbox**.

---

## Part 1 — Authorizing a purchase

### The five states

```
POST /api/payment/intent         agent prices it → preview. CHARGES NOTHING.
POST /api/payment/verify/start   issue a step-up challenge (OTP fallback)
POST /api/payment/verify         answer it
POST /api/payment/authorize      user releases the charge
POST /api/payment/cancel         user declines
```

`POST /api/payment/intent` is the boundary of the agent's authority. It returns
a fully-priced `PaymentIntent` that is completely inert: holding one confers no
ability to charge anything. Reaching any later state requires a user action the
agent cannot synthesise.

An intent expires after **15 minutes**. Prices, stock and intent all go stale;
re-pricing is cheap and a surprise charge is not.

### What the user sees first

A single total hides how many separate charges are being authorized, so the
preview breaks them out per merchant — which shop, which items, which card,
shipping, tax, delivery window. A purchase you cannot itemise is one you cannot
meaningfully consent to.

Prices are resolved server-side against the live catalog, never trusted from
the client, so the total shown cannot differ from the total charged.

### Risk signals

Every intent carries `risk[]` — plain-language explanations of why this
purchase is or is not routine. They appear **every time, including when nothing
is wrong.** A warning that only shows on bad news trains people to click past
it.

| code | step-up | meaning |
|---|---|---|
| `agent_initiated` | no | assembled by the agent, not typed by the user |
| `multi_merchant` | no | *n* separate statement lines, named |
| `routine` | no | within your usual limits and merchants |
| `token_presented` | no | paying via a scoped agent token, not a card number |
| `mandate_scoped` | no | inside the mandate, with remaining headroom named |
| `amount_over_threshold` | **yes** | over the per-card limit the user set |
| `new_merchant` | **yes** | first purchase from this shop |
| `over_budget` | **yes** | over the budget set for this room |
| `mandate_violation` | **blocks** | outside the mandate — verification cannot clear it |

The last row is the important one. A mandate violation is **not** a step-up: no
amount of verification clears it, because the user already decided this
purchase was out of scope. Verification proves *who you are*, not *that you
changed your mind*.

### A real trace

A newly-onboarded merchant selling a S$1,699 sofa against a S$2,000 budget:

```
intent    total 184766  step_up=True
  [agent_initiated]        step_up=False
  [amount_over_threshold]  step_up=True
  [new_merchant]           step_up=True
```

`new_merchant` fires because `KNOWN_MERCHANTS` is `{"IKEA"}` — anyone
onboarded through the merchant flow is unfamiliar by definition, and the first
purchase from them asks for a biometric.

---

## Part 2 — The Visa Agentic Payments Stack

Three layers sit on the rail. Each answers a question the rail alone cannot.

### Layer 1 — Presence (FIDO2 passkey)

Step-up is a device biometric bound to a hardware-held key, not a code.

```
POST /api/passkey/register/options   begin enrolment
POST /api/passkey/register           verify + store the public key
POST /api/passkey/challenge          challenge bound to one intent AND amount
POST /api/passkey/verify             verify the assertion → single-use id
```

The signature is bound to **this origin** and **this amount**. That is what a
one-time code cannot do:

- A code can be read aloud to someone on the phone. A passkey signature cannot
  be relayed — it is worthless on a lookalike domain.
- A code authorizes whatever the attacker is doing at that moment. A passkey
  signature is bound to a specific total, so proof-of-presence for a small
  purchase cannot authorize a large one.

No biometric data reaches the server. What crosses the wire is a signature over
a server-issued challenge.

Assertions are also tagged with a **purpose** (`payment` vs `provisioning`) and
checked on use. Without that, the biometric a user performed to approve one
small purchase could be replayed to mint a standing spending mandate — the two
ceremonies ask for the same gesture and mean entirely different things.

### Layer 2 — Scope (network token + mandate)

The agent never holds a card number. It spends against a scoped token carrying
a mandate.

```
GET  /api/agent-token/defaults    suggested limits
POST /api/agent-token/provision   mint a scoped AI_AGENT token
GET  /api/agent-token             tokens + spend history
```

A PAN is a bearer credential with no scope: anyone holding those 16 digits can
charge any amount, anywhere, forever, and the only remedy is cancelling the
card. Handing that to an autonomous agent is indefensible — not because the
agent is malicious, but because a prompt-injected agent, a logged request or a
compromised process then has unlimited spending power.

A mandate is signed (HMAC-SHA256, JWS-style) and verified on every use, so the
agent holds a credential it cannot forge or widen. It carries:

| Constraint | Why it exists |
|---|---|
| Per-transaction cap | The agent cannot assemble a single order above it |
| Cumulative cap | Otherwise a per-transaction cap is defeated by making many transactions |
| Category lock (MCC) | Furniture money cannot become airline money |
| Merchant allowlist | Tighter than category, when the user wants it |
| Max merchants per order | An agent bundling twelve unknown shops is a signal |
| Expiry (default 24h) | A standing permission the user forgot they granted is the failure mode |
| `require_user_presence` | Explicit and auditable rather than implied by code path |

**Category lock, in practice.** The mandate is expressed in Merchant Category
Codes — the same 4-digit codes the card networks have used since the 1970s.
Customers never see the number; they see the label.

| MCC | Shown to the shopper |
|---|---|
| 5712 | Furniture |
| 5713 | Rugs & Flooring |
| 5719 | Home Furnishing |
| 5200 | Home Supply |
| 5065 | Lighting |

The lock is expressed in codes rather than merchant names because a name check
is defeated by a merchant renaming itself, and because MCC is what actually
travels on the authorization.

**Unknown fails closed.** A merchant with no known MCC is refused, not waved
through:

```
IKEA (5719)              → ok
Northstar Home (unknown) → category_locked: not a furniture or home-decor
                           merchant. This agent token is locked to that category.
```

**A merchant cannot choose its own MCC.** `MerchantOnboardRequest` deliberately
has no `mcc` field — the platform assigns it. A merchant that could declare its
own category could opt itself into any agent's mandate.

### Layer 3 — Revocability

```
POST /api/agent-token/revoke      kill the mandate. THE CARD KEEPS WORKING.
```

This is the property that makes the whole design defensible. A user who thinks
the agent is misbehaving revokes the mandate and is still paying for lunch with
the same physical card five seconds later.

Revocation is checked **at authorization time**, not only at pricing time — a
mandate revoked in the seconds between reading a preview and authorizing stops
the charge. A revocation that only took effect on the next purchase would not
be a revocation.

### Split settlement

Each merchant leg gets its own **single-use cryptogram** bound to that leg's
amount, so a cryptogram captured from one merchant cannot be replayed against
another. Cryptograms expire after 5 minutes.

---

## Part 3 — Safeguards on authorization itself

`POST /api/payment/authorize` is the only endpoint that moves money. Every
refusal below is a case where charging would mean the user paid for something
other than what they agreed to.

| Refusal | Why |
|---|---|
| Total drifted from the confirmed amount | The thing they agreed to no longer exists |
| Step-up required but not completed | Presence was demanded and not proven |
| Preview expired | 15 minutes; prices and stock go stale |
| Mandate revoked since pricing | Re-checked here, not trusted from earlier |
| Mandate violation | Cannot be verified away |
| Already authorized / cancelled | An intent is spent once |

**Idempotency.** Replaying an `idempotency_key` returns the original receipt
rather than charging twice. The key is scoped to `(intent, key)`, not the key
alone — a client reusing one key across two purchases would otherwise be handed
the *first* purchase's receipt for the second, reporting success for an order
that was never authorized while the real one went uncharged.

**Partial failure is modelled.** One card (`···· 5454`) always declines, so the
flow can demonstrate what happens when one of three charges does not go
through. A checkout that only ever succeeds teaches nothing. Declined legs earn
nobody anything — only approved legs are recorded against a merchant, because
crediting a declined charge would put money in a payout ledger that was never
collected.

---

## Part 4 — Paying merchants (Visa Direct)

A merchant needs **no payment processor** to be paid. They nominate an account,
and their share of each order is pushed to it with Visa Direct — an Original
Credit Transaction (OCT).

```
GET  /api/merchant/balance        owed, and whether it can settle (signed)
GET  /api/merchant/payouts        per-order breakdown (signed)
POST /api/merchant/{id}/kyc       platform-side KYC outcome
POST /api/merchant/{id}/settle    pay pending balance ({"dry_run": true} to preview)
GET  /api/payouts/status          whether live payouts are configured
```

### Selling and being paid are different gates

| Gate | Requires |
|---|---|
| **Can sell** | An account. Available immediately on onboarding. |
| **Can be paid** | KYC verified, account active, payout card on file. |

A merchant may sell while `PENDING` — that is what makes onboarding usable —
but `can_settle` stays false until KYC passes, and every accrual against an
unsettleable account is logged as a warning. The merchant UI says which state
they are in rather than letting them assume money follows automatically.

### The split

Commission (default **5%**, 500 bps) is deducted from the merchant's gross, not
added to the shopper's total. The shopper already agreed to a number; changing
it after the fact would make the preview they approved a lie.

```
split     gross 184766 − commission 9238 = net 175528
          status pending_settlement
settle    not_settleable: cannot pay out: account is pending, KYC is unverified
dry_run   would_pay_cents 175528, records 1
```

KYC is re-checked **at payout**, not trusted from order time — the answer may
have changed since.

### Message Level Encryption

Visa Direct refuses a plaintext body even over mutual TLS. The payload is
JWE-encrypted (RSA-OAEP-256 + A128GCM) with Visa's public key, and Visa
encrypts its response back to ours.

The two certificates are easy to confuse, and getting them backwards produces a
payload Visa cannot read — with a rejection that never mentions encryption:

| Certificate | Role |
|---|---|
| **Server encryption cert** (Visa's) | **Encrypts** our requests |
| **Client private key** (ours, with the Key-ID) | **Decrypts** Visa's responses |

The Key-ID travels both as the JWE `kid` **and** as a `keyId` HTTP header;
omitting the header fails exactly like a missing credential. Visa Direct uses
**Two-Way SSL + Basic auth, not x-pay-token** — the reverse costs hours.

### Four payload details the sandbox rejects if wrong

Each fails with an error naming nothing useful:

- **`transactionIdentifier`** — numeric only, ≤15 digits. Order ids like
  `SIM-20617CF3` are refused; the order id goes in `senderReference`, which
  accepts text and keeps the payout traceable.
- **`retrievalReferenceNumber`** — *not* 12 arbitrary digits. The format is
  `yddd` + `hh` + 6 free digits, where `y` is the last digit of the year and
  `ddd` the day of the year.
- **HTTP 200 is not an approval.** The outcome is in `actionCode` (`00` =
  approved); anything else is a decline. Eleven codes are mapped to plain
  language. `settle` refuses to mark records paid on a decline, so a
  frequency-limited card leaves the ledger `pending_settlement`.
- **Error bodies are also MLE-encrypted**, so they are decrypted before being
  raised — otherwise every failure is an opaque blob at exactly the moment the
  reason matters most.

### Idempotent payouts

This is the part most worth reading, because getting it wrong costs real money.

**The failure it prevents:** Visa approves a payout, the response is lost on the
way back, `settle` raises, and an operator retries. If the retry looks like a
new payment, the merchant is paid twice with nothing recording that it was one
payout.

Visa deduplicates on trace numbers. If those numbers are randomly generated per
call, **a retry is indistinguishable from a second payment.**

The fix has three parts:

1. **Trace numbers derive from an idempotency key.** The same key always
   produces the same `systemsTraceAuditNumber`, `transactionIdentifier` and
   `retrievalReferenceNumber`, so Visa's own dedup recognises a retry. Derived
   by hashing rather than counting, because a counter needs durable storage
   that survives the crash the retry is recovering from. The key itself derives
   from the pending records, so an operator retrying after a timeout reproduces
   it without having saved anything.

2. **The attempt is persisted *before* the network call.** If the process dies
   mid-request, the attempt and its trace numbers survive — and those numbers
   are what identify the payout to Visa afterwards.

3. **Three outcomes, not two.** Conflating "declined" with "no response" is the
   error that causes double payment:

| Outcome | Meaning | Behaviour |
|---|---|---|
| `paid` | Approved | Replays the original receipt. No second push. |
| `declined` | Definitely no money moved | Retryable once the cause is fixed |
| `unknown` | **No outcome returned** | Refuses, and names the transaction to reconcile |

The `unknown` case refuses rather than guessing:

```
settlement_unresolved: a previous payout attempt for these 1 record(s) did not
return an outcome and may or may not have been paid. Reconcile transaction
714886776510893 (RRN 624117869616) with Visa before retrying.
```

Refusing to act is the correct behaviour when the alternative is possibly
paying someone twice. A human reconciles against Visa's records using the trace
numbers the attempt actually used.

> **Known limitation.** The attempt store is an in-memory dict, so a process
> restart loses attempt records and a retry after that would re-push. What the
> implementation gets right is the *ordering* — persist, call, update — which is
> identical against a database; only the store changes.

---

## What is real and what is not

| Real | Simulated |
|---|---|
| Mandates signed and verified on every use | No shopper money moves; auth codes are random hex |
| FIDO2 assertions cryptographically verified (ES256/RS256) | No funding account exists behind any token |
| Cumulative spend tracked, so caps are real ceilings | Mandate signing key is process-local, not an HSM |
| Revocation checked at authorization time | KYC fields are collected, never verified |
| Merchant accounts, HMAC signing, replay protection | Merchant state is in-memory; a restart clears it |
| Catalogue ingestion into the live vector index | |
| Per-merchant splits and commission arithmetic | |
| **Visa Direct: real JWE-encrypted, mutually-TLS'd requests to Visa's sandbox** | Sandbox moves no real money |

`SANDBOX_BASE` is hardcoded to `sandbox.api.visa.com`. There is no environment
variable that points it at production.

Without live credentials, `settle` returns `simulated: true`, lists exactly what
is missing, and leaves every record `pending_settlement`. It never marks a
payout paid that did not happen.

---

## Where this stops, and why

Accepting other people's merchants and settling money to them makes an operator
a **payment facilitator** — a regulated activity requiring KYC/AML on every
merchant, an acquirer or PayFac licence, chargeback liability, and PCI scope.
None of that is code, and none of it is here.

Visa Direct removes the requirement that every *merchant* hold an acquirer
relationship — they need only an account that can receive a push. Money still
lands with the platform first and is pushed out afterwards, so the operator
remains merchant of record. A real simplification for the merchant; none at all
for the platform.

### What production would additionally require

Honest list, roughly in order of how much they matter:

1. **Operator auth on `/kyc` and `/settle`.** Both are currently
   unauthenticated. KYC approval is a money-movement decision.
2. **Tokenized payout accounts.** `payout_pan` is stored as a plain string.
   Production collects the card in a PCI-scoped vault and stores a token — the
   difference between SAQ A and SAQ D.
3. **Durable storage.** Payout ledgers must survive a restart.
4. **Real KYC.** The four states (`unverified` → `in_review` → `verified` /
   `rejected`) are modelled; nothing populates them but a manual call.
5. **KMS-held MLE keys**, rather than PEM files on disk.

---

## Verifying it yourself

```bash
# 1. the agent prices a basket -> a preview, inert
curl -s localhost:8000/api/payment/intent -H 'content-type: application/json' \
  -d '{"item_ids":["ikea-40618528"]}' \
  | jq '.id, .total_cents, .requires_step_up, .risk'

# 2. authorizing before verifying is refused
#    {"detail":"identity verification required before authorizing"}

# 3. step up, then answer (demo_code is returned only because this is a demo)

# 4. authorizing a total the user never saw is refused
#    {"detail":"the total changed since you reviewed it ..."}

# 5. authorize for real, then replay the SAME key -> one order, not two

# whether live payouts are configured
curl -s localhost:8000/api/payouts/status
```

For the merchant half, see the "Onboarding third-party merchants" section of
the [README](README.md#onboarding-third-party-merchants).
