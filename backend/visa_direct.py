"""Visa Direct payouts: settling each merchant's share to their own account.

This is the rail that answers "how does a merchant get paid without having a
payment processor". The shopper authorizes once; each merchant's share is then
pushed to a card or account they nominated, using Visa's push-payment rail.

WHAT THIS DOES AND DOES NOT CHANGE

It does NOT remove the facilitator question. Money still lands with the
platform first and is pushed out afterwards, which makes the operator the
merchant of record with the KYC and licensing that implies. What it removes is
the requirement that every MERCHANT hold an acquirer relationship - they need
only an account that can receive a push. That is a real simplification for the
merchant and no simplification at all for the platform, and the README says so
rather than implying the regulatory problem went away.

MESSAGE LEVEL ENCRYPTION

Visa Direct will not accept a plaintext body even over mutual TLS. The payload
is JWE-encrypted (RSA-OAEP-256 + A128GCM) with Visa's public key, and Visa
encrypts its response to ours. That is why this module needs a key pair and a
key ID that the rest of the VDP integration does not.

WHAT IS SIMULATED

Everything, unless `VISA_LIVE` is on AND the Visa Direct credentials are
present. A sandbox push still moves no real money, but it is a real signed,
encrypted request to Visa - which is the part worth demonstrating. Without
credentials, payouts are recorded exactly as they are today and marked
`pending_settlement`, because a payout ledger that claimed to have paid
someone would be the single most dishonest thing in this codebase.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from . import config, visa_client

log = logging.getLogger("roomhack.visadirect")

PUSH_FUNDS_PATH = "/visadirect/fundstransfer/v1/pushfundstransactions"

# ISO 8583 action codes Visa Direct actually returns, in plain language. Only
# "00" is an approval; the rest are the declines a payout realistically hits.
ACTION_CODES: dict[str, str] = {
    "00": "approved",
    "05": "do not honour - the receiving issuer declined",
    "12": "invalid transaction for this account",
    "14": "invalid account number",
    "31": "the recipient's bank does not support this transfer",
    "51": "insufficient funds in the funding account",
    "57": "this transaction type is not permitted for the recipient's card",
    "62": "restricted card",
    "65": "exceeds withdrawal frequency limit",
    "91": "the recipient's issuer is unavailable - retry later",
    "96": "system error at the network - retry later",
}


class VisaDirectUnavailable(Exception):
    """Live payouts are not configured. Carries what is missing, in order."""


def _requirements() -> list[str]:
    """Everything still needed before a live payout can be attempted."""
    missing: list[str] = []
    if not config.VISA_LIVE:
        missing.append("VISA_LIVE=true")
    if not config.VISA_MLE_KEY_ID:
        missing.append("VISA_MLE_KEY_ID (from the VDP dashboard, beside your uploaded public key)")
    if not Path(config.VISA_MLE_SERVER_CERT_PATH).expanduser().is_file():
        missing.append(f"Visa's MLE certificate at {config.VISA_MLE_SERVER_CERT_PATH}")
    if not config.VISA_DIRECT_ACQUIRING_BIN:
        missing.append("VISA_DIRECT_ACQUIRING_BIN (issued with a Visa Direct agreement)")
    if not config.VISA_DIRECT_SENDER_ACCOUNT:
        missing.append("VISA_DIRECT_SENDER_ACCOUNT (the funding account)")
    return missing


def available() -> bool:
    return not _requirements()


def status() -> dict[str, Any]:
    """What is configured, for /api/health. Never returns a secret's value."""
    missing = _requirements()
    return {
        "live_payouts": not missing,
        "missing": missing,
        "mle_key_id": bool(config.VISA_MLE_KEY_ID),
        "mle_private_key": Path(config.VISA_MLE_PRIVATE_KEY_PATH).expanduser().is_file(),
        "mle_server_cert": Path(config.VISA_MLE_SERVER_CERT_PATH).expanduser().is_file(),
        "acquiring_bin": bool(config.VISA_DIRECT_ACQUIRING_BIN),
    }


# --- Message Level Encryption ----------------------------------------------


def encrypt_payload(payload: dict[str, Any]) -> str:
    """JWE-encrypt a request body with Visa's public key.

    RSA-OAEP-256 to wrap a fresh content key, A128GCM to encrypt the body -
    the algorithms Visa's MLE specifies. A new content key per request, so two
    payouts never share key material.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.x509 import load_pem_x509_certificate

    # THE TWO CERTIFICATES, since the naming invites exactly one mistake:
    #
    #   SERVER encryption certificate  Visa's public key, downloaded from the
    #       VDP portal. Encrypts REQUESTS. This is the one used here.
    #   CLIENT private key             Ours, issued with the Key-ID. Decrypts
    #       RESPONSES Visa encrypted to us. Used in `decrypt_payload`.
    #
    # Encrypting with our own key produces a payload Visa cannot read, and the
    # rejection says nothing about encryption - so the mistake is invisible.
    cert_path = Path(config.VISA_MLE_SERVER_CERT_PATH).expanduser()
    raw = cert_path.read_bytes()
    try:
        public_key = load_pem_x509_certificate(raw).public_key()
    except ValueError:
        public_key = serialization.load_pem_public_key(raw)

    header = {
        "alg": "RSA-OAEP-256",
        "enc": "A128GCM",
        "kid": config.VISA_MLE_KEY_ID,
        # Visa requires a millisecond timestamp inside the protected header;
        # it bounds how long a captured ciphertext is accepted.
        "iat": int(time.time() * 1000),
    }
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())

    content_key = AESGCM.generate_key(bit_length=128)
    iv = os.urandom(12)
    ciphertext_with_tag = AESGCM(content_key).encrypt(
        iv, json.dumps(payload).encode(), header_b64.encode()
    )
    # AESGCM appends the 16-byte tag; JWE carries it as its own segment.
    ciphertext, tag = ciphertext_with_tag[:-16], ciphertext_with_tag[-16:]

    encrypted_key = public_key.encrypt(
        content_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return ".".join(
        [
            header_b64,
            _b64url(encrypted_key),
            _b64url(iv),
            _b64url(ciphertext),
            _b64url(tag),
        ]
    )


def decrypt_payload(jwe: str) -> dict[str, Any]:
    """Decrypt a JWE response with our MLE private key."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    parts = jwe.split(".")
    if len(parts) != 5:
        raise ValueError("malformed JWE response")
    header_b64, encrypted_key, iv, ciphertext, tag = parts

    private_key = serialization.load_pem_private_key(
        Path(config.VISA_MLE_PRIVATE_KEY_PATH).expanduser().read_bytes(), password=None
    )
    content_key = private_key.decrypt(
        _b64url_decode(encrypted_key),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    plaintext = AESGCM(content_key).decrypt(
        _b64url_decode(iv),
        _b64url_decode(ciphertext) + _b64url_decode(tag),
        header_b64.encode(),
    )
    return json.loads(plaintext)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * ((-len(data)) % 4))


# --- Payout ----------------------------------------------------------------


def _digits_from(key: str, salt: str, width: int) -> str:
    """`width` decimal digits derived deterministically from an idempotency key.

    The same key always yields the same digits, which is the entire point: Visa
    deduplicates on the trace numbers, so a retry MUST carry the ones the first
    attempt used or the network cannot tell a retry from a second payment.

    Derived by hashing rather than by counting, because a counter needs durable
    storage that survives the crash the retry is recovering from - and the key
    the caller already holds is enough.
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return f"{int.from_bytes(digest, 'big') % (10 ** width):0{width}d}"


def _transaction_id(idempotency_key: str) -> str:
    """The 15-digit numeric transaction identifier, stable per idempotency key."""
    return _digits_from(idempotency_key, "txnid", 15)


def _systems_trace(idempotency_key: str) -> str:
    """A 6-digit systems-trace audit number (STAN), stable per idempotency key.

    Visa uses this to deduplicate: retrying with the same STAN is a retry,
    while a new one is a second payment.
    """
    return _digits_from(idempotency_key, "stan", 6)


def _retrieval_reference(idempotency_key: str) -> str:
    """A 12-digit RRN in Visa's expected shape: yddd hh nnnnnn.

    The date and hour segments are real - Visa validates their shape - and the
    trailing six digits are derived from the idempotency key so a retry within
    the same hour reproduces the original RRN exactly.

    A retry that crosses an hour boundary yields a different RRN, which is why
    the STAN and transaction identifier above carry no time component: between
    the three, the network still sees a duplicate. This is also why the caller
    must not rely on the RRN alone for deduplication.
    """
    now = time.gmtime()
    year_digit = str(now.tm_year)[-1]
    day_of_year = f"{now.tm_yday:03d}"
    hour = f"{now.tm_hour:02d}"
    return f"{year_digit}{day_of_year}{hour}{_digits_from(idempotency_key, 'rrn', 6)}"


def build_push_request(
    *,
    recipient_pan: str,
    amount_cents: int,
    currency: str = "SGD",
    recipient_name: str = "",
    order_id: str = "",
    idempotency_key: str,
) -> dict[str, Any]:
    """Build an OCT (Original Credit Transaction) push request.

    Amounts cross the wire in major units with two decimals, not cents -
    sending "1234" where Visa expects "12.34" is a 100x error, so the
    conversion happens exactly once, here.

    `idempotency_key` is REQUIRED and has no default. Every trace number Visa
    deduplicates on is derived from it, so the same key rebuilds a byte-similar
    request and the network recognises a retry; a fresh key is a new payment.
    Defaulting it would silently reintroduce the double-payment it exists to
    prevent, so the caller must decide.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    return {
        "acquirerCountryCode": config.VISA_DIRECT_ACQUIRER_COUNTRY,
        "acquiringBin": config.VISA_DIRECT_ACQUIRING_BIN,
        "amount": f"{amount_cents / 100:.2f}",
        "businessApplicationId": "AA",  # account-to-account
        "cardAcceptor": {
            "address": {"country": "SGP", "county": "SG", "state": "SG", "zipCode": "000000"},
            "idCode": "RH-PLATFORM",
            "name": "Room Hack",
            "terminalId": "RH0001",
        },
        "localTransactionDateTime": now,
        "recipientName": recipient_name[:24],
        "recipientPrimaryAccountNumber": recipient_pan,
        "senderAccountNumber": config.VISA_DIRECT_SENDER_ACCOUNT,
        "senderCountryCode": "SG",
        "senderName": "Room Hack Platform",
        # Free-text, so this is where the order id can actually live.
        "senderReference": order_id[:16],
        "systemsTraceAuditNumber": _systems_trace(idempotency_key),
        "transactionCurrencyCode": currency,
        # Numeric only, max 15 digits. Our order ids look like "SIM-20617CF3",
        # so they cannot be used directly - Visa rejects them as "Invalid
        # content". The order id travels in `senderReference` instead, which
        # accepts text, keeping the payout traceable back to the order.
        "transactionIdentifier": _transaction_id(idempotency_key),
        # RRN is NOT 12 arbitrary digits. Visa's format is yddd + hh + 6 free
        # digits, where y is the last digit of the year and ddd the day of the
        # year; a random 12-digit string is rejected as invalid content. The
        # trailing digits must also make the RRN unique within the day, so
        # they are random rather than sequential.
        "retrievalReferenceNumber": _retrieval_reference(idempotency_key),
        "pointOfServiceData": {
            "panEntryMode": "90",
            "posConditionCode": "00",
            "motoECIIndicator": "0",
        },
    }


def push_payout(
    *,
    recipient_pan: str,
    amount_cents: int,
    currency: str = "SGD",
    recipient_name: str = "",
    order_id: str = "",
    idempotency_key: str,
) -> dict[str, Any]:
    """Push one merchant's share to their account. Raises if not configured.

    Deliberately raises rather than silently simulating: a caller asking for a
    live payout must not be told one happened when it did not. The SIMULATED
    path lives in the caller, which decides based on `available()`.
    """
    missing = _requirements()
    if missing:
        raise VisaDirectUnavailable(
            "live payouts are not configured. Still needed: " + "; ".join(missing)
        )

    payload = build_push_request(
        recipient_pan=recipient_pan,
        amount_cents=amount_cents,
        currency=currency,
        recipient_name=recipient_name,
        order_id=order_id,
        idempotency_key=idempotency_key,
    )
    log.info(
        "visa direct push: %s %s to ····%s (order %s)",
        f"{amount_cents / 100:.2f}",
        currency,
        recipient_pan[-4:],
        order_id,
    )
    # Two-Way SSL + Basic auth + MLE. Visa Direct's authentication page is
    # explicit that it uses mutual TLS with a username and password - NOT
    # x-pay-token - and its encryption guide requires MLE on every call.
    response = visa_client.call(
        "POST",
        PUSH_FUNDS_PATH,
        body=payload,
        creds=visa_client.load_credentials(),
        use_mle=True,
        timeout=30,
    )
    # `visa_client.call` already decrypts MLE responses.
    #
    # An HTTP 200 is NOT an approval. Visa returns the outcome in `actionCode`
    # ("00" approved) and anything else is a decline with a reason - so a
    # caller that treated a 200 as success would report money moved when the
    # issuer refused it.
    action = str(response.get("actionCode", ""))
    response["approved"] = action == "00"
    response["decline_reason"] = "" if action == "00" else ACTION_CODES.get(
        action, f"declined (action code {action})"
    )
    return response
