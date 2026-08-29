"""Visa Developer Platform client: mutual TLS, Basic auth, x-pay-token.

This is the real network edge. Everything else in this app that says "Visa"
is a simulation; this module is the one place an HTTP request actually leaves
for developer.visa.com.

Two authentication schemes, because VDP uses different ones per product:

  MUTUAL TLS + BASIC AUTH   Visa Token Service, Visa Direct. The client
      certificate proves who we are at the transport layer; the User ID and
      Password prove it again at the HTTP layer. Both are required - the certs
      alone will not authenticate a request.

  X-PAY-TOKEN               Click to Pay / VDES. An HMAC-SHA256 over the
      resource path, query string, and body, keyed by a shared secret. Still
      sent over the same mTLS channel.

DESIGN NOTE - why this sits behind the same seam as the simulation:

The rest of the app talks to `vts.py`, which is entirely local and fake. This
module does not replace it. It sits beside it, and `config.VISA_LIVE` decides
which one the app uses. That matters for a reason beyond tidiness: VDP's
sandbox routinely returns 403 for products a project is not entitled to, and
token provisioning in particular usually requires a BIN sponsor. An
integration that assumed live access would break the demo for everyone who
does not have it. Simulation stays the default; live mode is opt-in and
degrades to simulation rather than to an error.

NOTHING HERE MOVES REAL MONEY EITHER. The VDP sandbox is a test environment
with test PANs. But unlike the rest of the app, the requests are real, the
credentials are real, and a misconfigured call will really fail - so the
errors this module raises are written to say exactly which credential is
wrong.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from . import config

log = logging.getLogger("roomhack.visa")

# VDP sandbox. The base path differs per product family; each method below
# names its own resource path relative to this.
SANDBOX_BASE = "https://sandbox.api.visa.com"

# A sandbox call that hangs is worse than one that fails: the checkout flow is
# in front of a user waiting for a spinner.
DEFAULT_TIMEOUT_SECONDS = 20


class VisaCredentialError(Exception):
    """A credential is missing or unusable. Raised at startup, not mid-payment.

    Separate from VisaAPIError because the two need different responses: this
    one means the deployment is misconfigured and no request should be
    attempted, while an API error means the request was made and refused.
    """


class VisaAPIError(Exception):
    """VDP refused a request. Carries the status and Visa's own error body."""

    def __init__(self, status: int, detail: str, body: Any = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.body = body


@dataclass
class VisaCredentials:
    """Everything needed to authenticate to VDP, validated up front."""

    cert_path: Path
    key_path: Path
    ca_path: Path
    user_id: str
    password: str
    api_key: str = ""
    shared_secret: str = ""

    @property
    def has_mtls(self) -> bool:
        return bool(self.user_id and self.password)

    @property
    def has_xpay(self) -> bool:
        return bool(self.api_key and self.shared_secret)


def load_credentials() -> VisaCredentials:
    """Read credentials from config and check they are actually usable.

    Every failure below names the specific file or variable at fault. A
    generic "authentication failed" against mutual TLS is close to
    undebuggable - the handshake can fail for at least four unrelated reasons,
    and the caller cannot tell them apart from the exception alone.
    """
    cert = Path(config.VISA_CERT_PATH).expanduser()
    key = Path(config.VISA_KEY_PATH).expanduser()
    ca = Path(config.VISA_CA_PATH).expanduser()

    missing = [
        name
        for name, path in (("certificate", cert), ("private key", key), ("CA bundle", ca))
        if not path.is_file()
    ]
    if missing:
        raise VisaCredentialError(
            f"missing Visa {', '.join(missing)}. Expected: "
            f"cert={cert}, key={key}, ca={ca}. "
            "Download the client certificate from your VDP project "
            "(Credentials -> Certificates) after submitting a CSR."
        )

    if not config.VISA_USER_ID or not config.VISA_PASSWORD:
        raise VisaCredentialError(
            "VISA_USER_ID and VISA_PASSWORD are required in addition to the "
            "certificates - VDP checks both mutual TLS and HTTP Basic auth."
        )

    creds = VisaCredentials(
        cert_path=cert,
        key_path=key,
        ca_path=ca,
        user_id=config.VISA_USER_ID,
        password=config.VISA_PASSWORD,
        api_key=config.VISA_API_KEY,
        shared_secret=config.VISA_SHARED_SECRET,
    )
    _assert_cert_matches_key(creds)
    return creds


def _assert_cert_matches_key(creds: VisaCredentials) -> None:
    """Check the certificate and private key are a pair, before any request.

    A mismatched cert and key fail the TLS handshake with an error that names
    neither file, and it is a genuinely easy mistake: VDP hands out a key at
    project creation and a certificate later, and regenerating either one
    silently invalidates the pair. Catching it here turns a baffling network
    error into a sentence that says what to do.
    """
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_cert_chain(certfile=str(creds.cert_path), keyfile=str(creds.key_path))
    except ssl.SSLError as exc:
        raise VisaCredentialError(
            f"the certificate at {creds.cert_path} does not match the private "
            f"key at {creds.key_path}. They must be the pair VDP issued "
            f"together - regenerating either one invalidates the other. ({exc})"
        ) from exc
    except OSError as exc:
        raise VisaCredentialError(
            f"could not read the Visa certificate or key: {exc}"
        ) from exc


def _ssl_context(creds: VisaCredentials) -> ssl.SSLContext:
    """Build the mutual-TLS context.

    TWO SEPARATE TRUST PATHS, which is the thing that makes this confusing:

      Verifying VISA'S server   sandbox.api.visa.com presents a DigiCert-issued
          certificate, verified against the system's public CA store like any
          other HTTPS site. Visa's own sandbox CA bundle is NOT involved and
          loading it here causes "unable to get local issuer certificate".

      Proving WE are us          Our client certificate, signed by the Payment
          Sandbox Issuing CA. That is what `creds.ca_path` chains, and it goes
          in `load_cert_chain`, not in the verification store.

    Verification stays ON. Disabling it is the standard workaround people reach
    for when a handshake fails, and it silently converts an authenticated
    channel into an unauthenticated one - on the exact connection carrying
    payment credentials.
    """
    # Public CA store: verifies Visa's server.
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    # Our identity: client cert + key. The sandbox chain is offered alongside
    # the leaf so Visa can build the path to its own issuing CA.
    ctx.load_cert_chain(certfile=str(creds.cert_path), keyfile=str(creds.key_path))
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def _basic_auth(creds: VisaCredentials) -> str:
    raw = f"{creds.user_id}:{creds.password}".encode()
    return "Basic " + base64.b64encode(raw).decode()


# The dashboard displays the shared secret RSA-encrypted under the MLE public
# key you registered ("Shared Secret is Encrypted using RSA/ECB/OAEPWith
# SHA-256AndMGF1Padding"), so what a developer copies is ciphertext, not the
# secret. Decrypting here means the value in .env can be exactly what the
# dashboard showed - pasting ciphertext is the expected case, not a mistake.
_SECRET_CACHE: dict[str, str] = {}


def _plaintext_shared_secret(creds: VisaCredentials) -> str:
    """Return the usable shared secret, decrypting it if it is encrypted.

    A plaintext secret is short and printable; the encrypted form is base64
    over RSA-2048 ciphertext (344 chars -> 256 bytes). Length is a reliable
    discriminator, and an unnecessary decrypt attempt is harmless because a
    plaintext value is not valid base64 ciphertext anyway.
    """
    secret = creds.shared_secret
    if len(secret) < 100:
        return secret  # already plaintext

    cached = _SECRET_CACHE.get(secret)
    if cached:
        return cached

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding

    # A project can hold several MLE key pairs - one registered on the X-Pay
    # Token tab, another issued for Visa Direct - and only the one whose
    # PUBLIC half encrypted this secret can decrypt it. Rather than make the
    # operator work out which, try each key we hold. Wrong keys fail fast and
    # locally, so this costs nothing and removes a genuinely confusing
    # failure: registering a second MLE key silently breaks the first one's
    # secret.
    candidates = [
        Path(config.VISA_MLE_PRIVATE_KEY_PATH).expanduser(),
        Path("./secrets/visa_mle_vdp.pem"),
        Path("./secrets/visa_mle_private.pem"),
    ]
    seen: set[Path] = set()
    tried: list[str] = []
    for key_path in candidates:
        if key_path in seen or not key_path.is_file():
            continue
        seen.add(key_path)
        tried.append(key_path.name)
        try:
            private_key = serialization.load_pem_private_key(
                key_path.read_bytes(), password=None
            )
            plaintext = private_key.decrypt(
                base64.b64decode(secret),
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            ).decode()
        except Exception:  # noqa: BLE001 - a wrong key is expected here
            continue
        _SECRET_CACHE[secret] = plaintext
        return plaintext

    raise VisaCredentialError(
        "could not decrypt VISA_SHARED_SECRET with any MLE private key on "
        f"disk (tried: {', '.join(tried) or 'none found'}). The shared secret "
        "is encrypted under whichever public key was registered on the X-Pay "
        "Token tab, so that key's private half must be present. Re-copy the "
        "shared secret after registering a new key, or restore the old key."
    )


def x_pay_token(
    resource_path: str, query_string: str, body: str, creds: VisaCredentials
) -> str:
    """Build an x-pay-token header for Click to Pay / VDES.

    HMAC-SHA256 over timestamp + resource path + query + body, keyed by the
    shared secret. The resource path excludes the leading "/vdp/", which is
    the detail every first integration gets wrong.
    """
    if not creds.has_xpay:
        raise VisaCredentialError(
            "VISA_API_KEY and VISA_SHARED_SECRET are required for x-pay-token "
            "calls. Both come from the X-Pay Token tab of your VDP project's "
            "Credentials page."
        )
    timestamp = str(int(time.time()))
    message = f"{timestamp}{resource_path}{query_string}{body}"
    digest = hmac.new(
        _plaintext_shared_secret(creds).encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return f"xv2:{timestamp}:{digest}"


def call(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    creds: VisaCredentials | None = None,
    use_xpay: bool = False,
    use_mle: bool = False,
    query: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Make one authenticated VDP request.

    `use_xpay` selects the x-pay-token signing scheme; everything else uses
    Basic auth. Both travel over the same mutual-TLS channel.

    `use_mle` additionally JWE-encrypts the request body. Some products -
    Visa Direct among them - reject a plaintext body even over mutual TLS and
    even with valid credentials, which surfaces as a bare 9123/9125 rather
    than anything mentioning encryption. The signature is computed over the
    ENCRYPTED body, because that is what is actually transmitted.
    """
    creds = creds or load_credentials()
    payload = json.dumps(body) if body is not None else ""

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if use_mle and body is not None:
        from . import visa_direct

        payload = json.dumps({"encData": visa_direct.encrypt_payload(body)})
        # Per Visa's encryption guide the Key-ID travels as an HTTP header as
        # well as the JWE `kid`. Omitting the header is rejected the same way a
        # missing credential is, with no mention of encryption - which is what
        # makes it hard to find.
        headers["keyId"] = config.VISA_MLE_KEY_ID
    if use_xpay:
        # Two details Visa's docs are quiet about, both verified against the
        # sandbox:
        #   1. The resource path excludes the leading "/vdp/".
        #   2. The API key travels in the QUERY STRING, not a header, and is
        #      therefore part of the signed message. Sent as a header instead,
        #      every request fails 9123 with no indication why.
        resource = path[len("/vdp/") :] if path.startswith("/vdp/") else path.lstrip("/")
        api_key_qs = f"apikey={creds.api_key}" if creds.api_key else ""
        query = f"{query}&{api_key_qs}" if query and api_key_qs else (query or api_key_qs)
        headers["x-pay-token"] = x_pay_token(resource, query, payload, creds)
    else:
        headers["Authorization"] = _basic_auth(creds)

    # Built AFTER the auth block, because x-pay-token appends the API key to
    # the query string and the signed query must match the one actually sent.
    url = f"{SANDBOX_BASE}{path}" + (f"?{query}" if query else "")

    req = request.Request(
        url, data=payload.encode() if payload else None, headers=headers, method=method
    )

    try:
        with request.urlopen(req, timeout=timeout, context=_ssl_context(creds)) as res:
            raw = res.read().decode()
            result = json.loads(raw) if raw else {}
            # Unwrap MLE responses here so callers always see plain JSON,
            # whichever product they called.
            if use_mle and isinstance(result, dict) and "encData" in result:
                from . import visa_direct

                result = visa_direct.decrypt_payload(result["encData"])
            return result
    except error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = raw
        # Error bodies come back MLE-encrypted too, so an unhelpful blob is
        # what a caller would otherwise see at exactly the moment they most
        # need the reason.
        if isinstance(parsed, dict) and "encData" in parsed:
            try:
                from . import visa_direct

                parsed = visa_direct.decrypt_payload(parsed["encData"])
                raw = json.dumps(parsed)
            except Exception:  # noqa: BLE001 - keep the ciphertext if we cannot
                pass
        # 401/403 dominate first-time integrations and mean different things:
        # one is a bad credential, the other an unentitled product. Saying
        # which saves a long hunt through the wrong documentation.
        if exc.code == 401:
            detail = (
                "VDP rejected the credentials (401). Check VISA_USER_ID and "
                "VISA_PASSWORD, and that the client certificate belongs to "
                "this project."
            )
        elif exc.code == 403:
            detail = (
                "VDP refused this product (403). The project is authenticated "
                "but not entitled to this API - Visa Token Service "
                "provisioning in particular usually needs a partner agreement "
                "or BIN sponsor. This is not a configuration error you can fix "
                "locally."
            )
        else:
            detail = f"VDP returned {exc.code}: {raw[:300]}"
        raise VisaAPIError(exc.code, detail, parsed) from exc
    except ssl.SSLError as exc:
        raise VisaAPIError(
            0,
            f"TLS handshake with VDP failed: {exc}. This usually means the "
            "certificate/key pair is wrong, or the CA bundle does not include "
            "the intermediate certificate.",
        ) from exc
    except error.URLError as exc:
        raise VisaAPIError(0, f"could not reach VDP: {exc.reason}") from exc


# --- Connectivity check ----------------------------------------------------


def ping() -> dict[str, Any]:
    """Verify credentials end to end against VDP's own test endpoint.

    Run this before debugging anything else. It exercises the full path -
    certificate, key, CA chain, Basic auth - and every distinct failure mode
    returns a different message.
    """
    creds = load_credentials()
    return call("GET", "/vdp/helloworld", creds=creds)


def status() -> dict[str, Any]:
    """Report what is configured, without making a request.

    Deliberately never returns a secret's value - only whether it is set.
    This surfaces in /api/health, and a health endpoint that echoed a
    password would be a credential leak on an unauthenticated route.
    """
    cert = Path(config.VISA_CERT_PATH).expanduser()
    key = Path(config.VISA_KEY_PATH).expanduser()
    ca = Path(config.VISA_CA_PATH).expanduser()
    ready = all(p.is_file() for p in (cert, key, ca)) and bool(
        config.VISA_USER_ID and config.VISA_PASSWORD
    )
    return {
        "live_mode": config.VISA_LIVE,
        "ready": ready,
        "certificate": cert.is_file(),
        "private_key": key.is_file(),
        "ca_bundle": ca.is_file(),
        "basic_auth": bool(config.VISA_USER_ID and config.VISA_PASSWORD),
        "x_pay_token": bool(config.VISA_API_KEY and config.VISA_SHARED_SECRET),
    }
