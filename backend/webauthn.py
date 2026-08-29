"""FIDO2 / WebAuthn verification: the Visa Payment Passkey (VPP) step-up.

This replaces the SMS OTP as the primary way a user proves they are present.
The difference is not cosmetic. An OTP is a shared secret that travels over a
channel an attacker can intercept, and the user can be socially engineered
into reading it aloud. A passkey is a private key held in the device's secure
enclave that never leaves it, unlocked by a local biometric, and the signature
it produces is bound to this origin - so a phished user on a lookalike domain
produces a signature the real server rejects.

WHAT IS REAL HERE, and what is not:

  REAL   The challenge is server-generated, single-use and time-boxed.
  REAL   The signature is verified with the public key registered earlier,
         using ES256 / RS256 over the exact bytes the spec defines.
  REAL   clientDataJSON is checked for type, challenge and origin.
  REAL   The RP ID hash and the User Present / User Verified flags are
         checked against the authenticator data.
  REAL   The signature counter is checked for cloned-authenticator rollback.
  REAL   The transaction amount is bound into the challenge, so a signature
         for a S$40 purchase cannot authorize a S$4,000 one.

  NOT    Attestation is not verified against a trusted root. A production
         relying party would check the attestation statement against the FIDO
         Metadata Service to know WHICH authenticator model it is talking to.
         We accept any authenticator ("none" attestation), which is the normal
         choice for consumer passkeys anyway, but it means we know the key is
         held in *a* secure enclave without knowing whose.
  NOT    Nothing is registered with Visa. There is no relying party at Visa,
         no real VTS enrollment. The shape is faithful; the counterparty is
         this process.

Biometric data never reaches this module - that is the whole point of the
design. Face ID happens on the device, unlocks the private key locally, and
only a signature crosses the wire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

log = logging.getLogger("roomhack.webauthn")

# A challenge is evidence that a human touched the sensor *just now*. Ten
# minutes is the spec's usual ceiling; we use two because a payment
# authorization the user walked away from should not still be signable.
CHALLENGE_TTL_SECONDS = 120

# WebAuthn authenticator data flag bits (spec section 6.1).
FLAG_USER_PRESENT = 0x01   # someone physically touched the device
FLAG_USER_VERIFIED = 0x04  # ...and proved it was *them* (biometric / PIN)
FLAG_ATTESTED_DATA = 0x40  # attested credential data follows

# COSE key/algorithm constants (RFC 8152). Only the two algorithms real
# platform authenticators actually produce.
COSE_ALG_ES256 = -7
COSE_ALG_RS256 = -257
COSE_KTY_EC2 = 2
COSE_KTY_RSA = 3


# --- base64url -------------------------------------------------------------


def b64url_decode(data: str) -> bytes:
    """Decode base64url, tolerating the padding browsers omit."""
    if isinstance(data, bytes):  # pragma: no cover - defensive
        data = data.decode()
    padding_needed = (-len(data)) % 4
    return base64.urlsafe_b64decode(data + "=" * padding_needed)


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


# --- Minimal CBOR ----------------------------------------------------------
#
# An attestation object is CBOR, and a COSE public key is CBOR. Rather than
# take a dependency for two shapes, we decode the subset the spec permits
# here: unsigned/negative ints, byte strings, text strings, arrays and maps.
# Anything outside that subset raises rather than guessing, because silently
# mis-parsing a key is how you end up verifying signatures against garbage.


class CborError(ValueError):
    """Malformed or unsupported CBOR in an authenticator response."""


def _cbor_read(buf: bytes, pos: int) -> tuple[Any, int]:
    if pos >= len(buf):
        raise CborError("truncated CBOR")
    initial = buf[pos]
    major, info = initial >> 5, initial & 0x1F
    pos += 1

    # Argument decoding, shared by every major type.
    if info < 24:
        arg = info
    elif info == 24:
        arg = buf[pos]; pos += 1
    elif info == 25:
        arg = struct.unpack_from(">H", buf, pos)[0]; pos += 2
    elif info == 26:
        arg = struct.unpack_from(">I", buf, pos)[0]; pos += 4
    elif info == 27:
        arg = struct.unpack_from(">Q", buf, pos)[0]; pos += 8
    else:
        raise CborError(f"unsupported CBOR additional info {info}")

    if major == 0:
        return arg, pos
    if major == 1:
        return -1 - arg, pos
    if major == 2:
        end = pos + arg
        if end > len(buf):
            raise CborError("truncated CBOR byte string")
        return buf[pos:end], end
    if major == 3:
        end = pos + arg
        if end > len(buf):
            raise CborError("truncated CBOR text string")
        return buf[pos:end].decode("utf-8", "replace"), end
    if major == 4:
        items = []
        for _ in range(arg):
            item, pos = _cbor_read(buf, pos)
            items.append(item)
        return items, pos
    if major == 5:
        out: dict[Any, Any] = {}
        for _ in range(arg):
            key, pos = _cbor_read(buf, pos)
            val, pos = _cbor_read(buf, pos)
            out[key] = val
        return out, pos
    if major == 7 and info in (20, 21, 22):
        return {20: False, 21: True, 22: None}[info], pos
    raise CborError(f"unsupported CBOR major type {major}")


def cbor_decode(buf: bytes) -> Any:
    """Decode one CBOR item. Trailing bytes are allowed and ignored."""
    value, _ = _cbor_read(buf, 0)
    return value


def cbor_decode_prefix(buf: bytes) -> tuple[Any, int]:
    """Decode one CBOR item, returning how many bytes it consumed.

    Needed for COSE keys embedded in attested credential data, where the key
    is followed by extension bytes and the spec gives no explicit length.
    """
    return _cbor_read(buf, 0)


# --- Registered credentials ------------------------------------------------


@dataclass
class StoredCredential:
    """A registered passkey. Public key only - the private half never leaves
    the user's secure enclave, which is the property that makes this stronger
    than any secret we could hold on their behalf."""

    credential_id: str          # base64url
    public_key_cose: bytes      # COSE_Key, as the authenticator gave it
    alg: int
    sign_count: int = 0
    user_handle: str = ""
    label: str = "This device"
    transports: list[str] = field(default_factory=list)
    created_at: float = 0.0
    # Whether the authenticator verified the *user* (biometric/PIN) at
    # registration, not merely that someone was present. A credential
    # registered without it cannot satisfy a payment step-up.
    user_verified_at_registration: bool = False
    backed_up: bool = False     # synced to a keychain (iCloud/Google)


CREDENTIALS: dict[str, StoredCredential] = {}
CHALLENGES: dict[str, dict[str, Any]] = {}


class WebAuthnError(Exception):
    """A refusal the user should see. Carries an HTTP status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _now() -> float:
    return time.time()


def _purge_expired() -> None:
    now = _now()
    for key in [k for k, v in CHALLENGES.items() if v["expires_at"] < now]:
        CHALLENGES.pop(key, None)


# --- Challenge issuance ----------------------------------------------------


def create_challenge(
    purpose: str,
    *,
    intent_id: str | None = None,
    amount_cents: int = 0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mint a single-use challenge bound to what it authorizes.

    `amount_cents` and `intent_id` are recorded alongside the challenge, not
    merely displayed. Verification refuses a signature whose challenge was
    issued for a different intent or a different amount - so a signature
    captured for a small purchase cannot be replayed against a large one.
    This is the transaction-binding property that makes a passkey a payment
    instrument rather than just a login.
    """
    _purge_expired()
    raw = secrets.token_bytes(32)
    challenge = b64url_encode(raw)
    CHALLENGES[challenge] = {
        "purpose": purpose,
        "intent_id": intent_id,
        "amount_cents": amount_cents,
        "context": context or {},
        "expires_at": _now() + CHALLENGE_TTL_SECONDS,
        "used": False,
    }
    return {
        "challenge": challenge,
        "expires_at": CHALLENGES[challenge]["expires_at"],
        "purpose": purpose,
    }


def _consume_challenge(
    challenge: str, purpose: str, intent_id: str | None, amount_cents: int
) -> dict[str, Any]:
    """Validate a challenge and burn it. Single-use, always."""
    record = CHALLENGES.get(challenge)
    if record is None:
        raise WebAuthnError(400, "unknown or already-used challenge")
    if record["used"]:
        raise WebAuthnError(400, "this challenge was already used")
    if _now() > record["expires_at"]:
        CHALLENGES.pop(challenge, None)
        raise WebAuthnError(410, "challenge expired - please try again")
    if record["purpose"] != purpose:
        raise WebAuthnError(400, "challenge was issued for a different purpose")
    if intent_id is not None and record["intent_id"] != intent_id:
        raise WebAuthnError(400, "challenge was issued for a different payment")
    # The binding that matters: a signature over "pay S$40" is not a signature
    # over "pay S$4,000", even if everything else about the request matches.
    if record["amount_cents"] != amount_cents:
        raise WebAuthnError(
            400,
            "the amount changed since this authorization was requested - "
            "please review the updated total",
        )
    record["used"] = True
    return record


# --- Signature verification ------------------------------------------------


def _load_public_key(cose: dict[Any, Any]):
    """Turn a COSE_Key into a cryptography public key object."""
    kty = cose.get(1)
    alg = cose.get(3)

    if kty == COSE_KTY_EC2:
        if cose.get(-1) != 1:  # P-256 only, which is what ES256 means
            raise WebAuthnError(400, "unsupported elliptic curve")
        x, y = cose.get(-2), cose.get(-3)
        if not isinstance(x, bytes) or not isinstance(y, bytes):
            raise WebAuthnError(400, "malformed EC2 public key")
        numbers = ec.EllipticCurvePublicNumbers(
            int.from_bytes(x, "big"), int.from_bytes(y, "big"), ec.SECP256R1()
        )
        return numbers.public_key(), alg

    if kty == COSE_KTY_RSA:
        n, e = cose.get(-1), cose.get(-2)
        if not isinstance(n, bytes) or not isinstance(e, bytes):
            raise WebAuthnError(400, "malformed RSA public key")
        numbers = rsa.RSAPublicNumbers(
            int.from_bytes(e, "big"), int.from_bytes(n, "big")
        )
        return numbers.public_key(), alg

    raise WebAuthnError(400, f"unsupported key type {kty}")


def _verify_signature(cose_key: bytes, alg: int, signed: bytes, signature: bytes) -> None:
    """Verify `signature` over `signed`. Raises on any failure."""
    try:
        key, key_alg = _load_public_key(cbor_decode(cose_key))
    except CborError as exc:
        raise WebAuthnError(400, f"malformed credential public key: {exc}") from exc

    # Trust the algorithm recorded at registration, not one the client asserts
    # now - otherwise an attacker picks the weakest algorithm the key allows.
    effective = key_alg if key_alg in (COSE_ALG_ES256, COSE_ALG_RS256) else alg

    try:
        if effective == COSE_ALG_ES256:
            if not isinstance(key, ec.EllipticCurvePublicKey):
                raise WebAuthnError(400, "key/algorithm mismatch")
            key.verify(signature, signed, ec.ECDSA(hashes.SHA256()))
        elif effective == COSE_ALG_RS256:
            if not isinstance(key, rsa.RSAPublicKey):
                raise WebAuthnError(400, "key/algorithm mismatch")
            key.verify(signature, signed, padding.PKCS1v15(), hashes.SHA256())
        else:
            raise WebAuthnError(400, f"unsupported algorithm {effective}")
    except InvalidSignature as exc:
        raise WebAuthnError(401, "signature did not verify") from exc


def _parse_auth_data(auth_data: bytes) -> dict[str, Any]:
    """Split authenticator data into its fixed header and optional payload."""
    if len(auth_data) < 37:
        raise WebAuthnError(400, "authenticator data too short")
    rp_id_hash = auth_data[:32]
    flags = auth_data[32]
    sign_count = struct.unpack_from(">I", auth_data, 33)[0]

    parsed: dict[str, Any] = {
        "rp_id_hash": rp_id_hash,
        "flags": flags,
        "user_present": bool(flags & FLAG_USER_PRESENT),
        "user_verified": bool(flags & FLAG_USER_VERIFIED),
        "backed_up": bool(flags & 0x10),
        "sign_count": sign_count,
        "credential_id": None,
        "public_key_cose": None,
    }

    if flags & FLAG_ATTESTED_DATA:
        # aaguid(16) || credIdLen(2) || credId || COSE key
        if len(auth_data) < 55:
            raise WebAuthnError(400, "attested credential data truncated")
        cred_len = struct.unpack_from(">H", auth_data, 53)[0]
        start = 55
        end = start + cred_len
        if end > len(auth_data):
            raise WebAuthnError(400, "credential id length overruns buffer")
        parsed["credential_id"] = auth_data[start:end]
        try:
            key, consumed = cbor_decode_prefix(auth_data[end:])
        except CborError as exc:
            raise WebAuthnError(400, f"malformed COSE key: {exc}") from exc
        # Re-encode is not needed: we keep the exact bytes the authenticator
        # sent so verification later parses precisely what was registered.
        parsed["public_key_cose"] = auth_data[end : end + consumed]
        parsed["cose"] = key
    return parsed


def _check_client_data(
    client_data_json: bytes,
    expected_type: str,
    expected_challenge: str,
    allowed_origins: list[str],
) -> dict[str, Any]:
    """Validate the browser's account of what it was asked to sign.

    The origin check is the anti-phishing property of WebAuthn: a passkey on
    evil-lookalike.com produces clientDataJSON naming that origin, and this
    rejects it. Without this check a passkey is no better than a password.
    """
    try:
        client_data = json.loads(client_data_json.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise WebAuthnError(400, "malformed clientDataJSON") from exc

    if client_data.get("type") != expected_type:
        raise WebAuthnError(
            400,
            f"wrong ceremony type: expected {expected_type}, "
            f"got {client_data.get('type')!r}",
        )

    # Constant-time: the challenge is a secret the client must echo exactly.
    got = str(client_data.get("challenge") or "")
    if not secrets.compare_digest(got, expected_challenge):
        raise WebAuthnError(400, "challenge mismatch - possible replay")

    origin = str(client_data.get("origin") or "")
    if origin not in allowed_origins:
        raise WebAuthnError(
            403,
            f"origin {origin!r} is not an allowed origin for this relying party",
        )

    if client_data.get("crossOrigin") is True:
        raise WebAuthnError(403, "cross-origin authentication is not accepted")

    return client_data


# --- Registration ----------------------------------------------------------


def registration_options(
    *, rp_id: str, rp_name: str, user_id: str, user_name: str, user_display: str
) -> dict[str, Any]:
    """Options for navigator.credentials.create().

    `userVerification: required` and `residentKey: required` together are what
    make this a *payment* passkey rather than a login passkey: the credential
    must be discoverable on the device and must demand a biometric or PIN each
    time, so possession of an unlocked laptop is never sufficient.
    """
    challenge = create_challenge("registration")
    return {
        "challenge": challenge["challenge"],
        "rp": {"id": rp_id, "name": rp_name},
        "user": {
            "id": b64url_encode(user_id.encode()),
            "name": user_name,
            "displayName": user_display,
        },
        "pubKeyCredParams": [
            {"type": "public-key", "alg": COSE_ALG_ES256},
            {"type": "public-key", "alg": COSE_ALG_RS256},
        ],
        "timeout": CHALLENGE_TTL_SECONDS * 1000,
        "attestation": "none",
        "authenticatorSelection": {
            "authenticatorAttachment": "platform",   # Touch ID / Face ID, not a USB key
            "residentKey": "required",
            "requireResidentKey": True,
            "userVerification": "required",
        },
        "excludeCredentials": [
            {"type": "public-key", "id": c.credential_id}
            for c in CREDENTIALS.values()
        ],
        "expires_at": challenge["expires_at"],
    }


def verify_registration(
    *,
    credential_id: str,
    client_data_json_b64: str,
    attestation_object_b64: str,
    transports: list[str] | None = None,
    label: str = "",
    rp_id: str,
    allowed_origins: list[str],
    user_handle: str = "",
) -> StoredCredential:
    """Verify a create() response and store the public key."""
    client_data_json = b64url_decode(client_data_json_b64)
    try:
        attestation = cbor_decode(b64url_decode(attestation_object_b64))
    except CborError as exc:
        raise WebAuthnError(400, f"malformed attestation object: {exc}") from exc

    auth_data_raw = attestation.get("authData")
    if not isinstance(auth_data_raw, bytes):
        raise WebAuthnError(400, "attestation object missing authData")
    auth_data = _parse_auth_data(auth_data_raw)

    # Read the challenge out of clientData first so we can validate it against
    # our own store; _check_client_data then re-compares in constant time.
    try:
        stated_challenge = str(json.loads(client_data_json).get("challenge") or "")
    except ValueError as exc:
        raise WebAuthnError(400, "malformed clientDataJSON") from exc

    _consume_challenge(stated_challenge, "registration", None, 0)
    _check_client_data(
        client_data_json, "webauthn.create", stated_challenge, allowed_origins
    )

    if auth_data["rp_id_hash"] != hashlib.sha256(rp_id.encode()).digest():
        raise WebAuthnError(403, "credential was created for a different site")
    if not auth_data["user_present"]:
        raise WebAuthnError(400, "authenticator did not report user presence")
    if not auth_data["user_verified"]:
        # Refused rather than downgraded: a credential registered without user
        # verification can never satisfy a payment step-up, so registering it
        # would only produce a passkey that fails at checkout.
        raise WebAuthnError(
            400,
            "this device did not verify your identity (Face ID / Touch ID / PIN). "
            "A payment passkey requires it.",
        )
    if auth_data["credential_id"] is None or auth_data["public_key_cose"] is None:
        raise WebAuthnError(400, "attestation contained no credential")

    stored_id = b64url_encode(auth_data["credential_id"])
    # Trust the authenticator's own credential id over the client's echo of it:
    # the former is inside the signed-over authenticator data.
    if credential_id and credential_id != stored_id:
        log.warning("client credential id disagreed with authData; using authData")

    cose = auth_data.get("cose") or {}
    credential = StoredCredential(
        credential_id=stored_id,
        public_key_cose=auth_data["public_key_cose"],
        alg=int(cose.get(3) or COSE_ALG_ES256),
        sign_count=auth_data["sign_count"],
        user_handle=user_handle,
        label=label or "This device",
        transports=transports or [],
        created_at=_now(),
        user_verified_at_registration=True,
        backed_up=auth_data["backed_up"],
    )
    CREDENTIALS[stored_id] = credential
    log.info("passkey registered: %s (%s)", credential.label, stored_id[:12])
    return credential


# --- Authentication --------------------------------------------------------


def authentication_options(
    *,
    rp_id: str,
    intent_id: str | None = None,
    amount_cents: int = 0,
    purpose: str = "payment",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Options for navigator.credentials.get(), bound to this transaction."""
    if not CREDENTIALS:
        raise WebAuthnError(409, "no passkey is registered on this account yet")
    challenge = create_challenge(
        purpose, intent_id=intent_id, amount_cents=amount_cents, context=context
    )
    return {
        "challenge": challenge["challenge"],
        "rpId": rp_id,
        "timeout": CHALLENGE_TTL_SECONDS * 1000,
        "userVerification": "required",
        "allowCredentials": [
            {
                "type": "public-key",
                "id": c.credential_id,
                "transports": c.transports or ["internal"],
            }
            for c in CREDENTIALS.values()
        ],
        "expires_at": challenge["expires_at"],
    }


@dataclass
class AssertionResult:
    """Proof that the user was present and verified, for this exact amount."""

    credential_id: str
    user_verified: bool
    sign_count: int
    amount_cents: int
    intent_id: str | None
    verified_at: float
    # An opaque handle the payment rail exchanges for a token cryptogram. It
    # is the *result* of the biometric, never the biometric itself.
    assertion_id: str


def verify_assertion(
    *,
    credential_id: str,
    client_data_json_b64: str,
    authenticator_data_b64: str,
    signature_b64: str,
    rp_id: str,
    allowed_origins: list[str],
    purpose: str = "payment",
    intent_id: str | None = None,
    amount_cents: int = 0,
) -> AssertionResult:
    """Verify a get() response. This is the moment "it is really them" is proven.

    Every check below is one an attacker has to defeat: the signature (needs
    the enclave), the origin (needs the real domain), the challenge (needs to
    be live, unused, and issued for this exact amount), the UV flag (needs the
    biometric, not just the device), and the counter (needs the original key,
    not a clone).
    """
    credential = CREDENTIALS.get(credential_id)
    if credential is None:
        raise WebAuthnError(404, "unknown passkey - please register this device")

    client_data_json = b64url_decode(client_data_json_b64)
    authenticator_data = b64url_decode(authenticator_data_b64)
    signature = b64url_decode(signature_b64)

    try:
        stated_challenge = str(json.loads(client_data_json).get("challenge") or "")
    except ValueError as exc:
        raise WebAuthnError(400, "malformed clientDataJSON") from exc

    record = _consume_challenge(stated_challenge, purpose, intent_id, amount_cents)
    _check_client_data(
        client_data_json, "webauthn.get", stated_challenge, allowed_origins
    )

    auth_data = _parse_auth_data(authenticator_data)
    if auth_data["rp_id_hash"] != hashlib.sha256(rp_id.encode()).digest():
        raise WebAuthnError(403, "assertion was signed for a different site")
    if not auth_data["user_present"]:
        raise WebAuthnError(401, "authenticator did not report user presence")
    if not auth_data["user_verified"]:
        raise WebAuthnError(
            401,
            "Face ID / Touch ID did not verify you. A payment requires "
            "biometric or device-PIN verification, not just an unlocked device.",
        )

    # The signed bytes are exactly authenticatorData || SHA256(clientDataJSON).
    signed = authenticator_data + hashlib.sha256(client_data_json).digest()
    _verify_signature(credential.public_key_cose, credential.alg, signed, signature)

    # Counter regression means two authenticators share one private key - i.e.
    # the credential was extracted and cloned. Authenticators that always
    # report 0 (common for passkeys synced through a keychain) are exempt,
    # since for those the counter carries no information.
    new_count = auth_data["sign_count"]
    if new_count != 0 or credential.sign_count != 0:
        if new_count <= credential.sign_count:
            raise WebAuthnError(
                401,
                "this passkey's signature counter went backwards, which can "
                "mean it has been cloned. Re-register the device.",
            )
        credential.sign_count = new_count

    result = AssertionResult(
        credential_id=credential_id,
        user_verified=True,
        sign_count=credential.sign_count,
        amount_cents=record["amount_cents"],
        intent_id=record["intent_id"],
        verified_at=_now(),
        assertion_id=f"vpp_{secrets.token_hex(8)}",
    )
    log.info(
        "passkey assertion verified: %s for intent %s (%d cents)",
        credential_id[:12],
        intent_id,
        amount_cents,
    )
    return result


def list_credentials() -> list[StoredCredential]:
    return list(CREDENTIALS.values())


def delete_credential(credential_id: str) -> bool:
    return CREDENTIALS.pop(credential_id, None) is not None
