"""Verify Visa sandbox credentials. Run: python -m backend.visa_check

Exists as a standalone command because credential problems are much easier to
diagnose away from a running server, and because the first question when a
payment call fails is always "are the credentials even right".
"""

from __future__ import annotations

import sys

from . import config, visa_client


def main() -> int:
    print("Visa Developer Platform - credential check\n")

    st = visa_client.status()
    rows = [
        ("Client certificate", st["certificate"], config.VISA_CERT_PATH),
        ("Private key", st["private_key"], config.VISA_KEY_PATH),
        ("CA bundle", st["ca_bundle"], config.VISA_CA_PATH),
        ("User ID / Password", st["basic_auth"], "VISA_USER_ID, VISA_PASSWORD"),
        ("API key / secret", st["x_pay_token"], "VISA_API_KEY, VISA_SHARED_SECRET"),
    ]
    for label, present, where in rows:
        print(f"  [{'x' if present else ' '}] {label:22} {where}")

    print(f"\n  live mode: {'ON' if config.VISA_LIVE else 'OFF (simulation)'}")

    if not st["ready"]:
        # The x-pay-token pair is deliberately not required here: it is only
        # needed for Click to Pay, and everything else works without it.
        print("\nNot ready. Missing pieces above must be filled in first.")
        if not st["certificate"]:
            # Most VDP projects already HAVE a certificate - it is generated
            # with the keypair at project creation and simply needs
            # downloading. Leading with "submit a CSR" sends people hunting
            # for a step they have already passed.
            print(
                "\nThe client certificate is downloaded from your VDP project:"
                "\n  Credentials -> Two-Way SSL -> Inbound -> Download (in the"
                "\n  Actions column of the active certificate row), then save it"
                f"\n  to {config.VISA_CERT_PATH}."
                "\n\n  If no certificate row exists there, use Add Credential for"
                "\n  Inbound - and if it asks for a CSR, one is ready at"
                "\n  secrets/visa_request.csr."
            )
        return 1

    print("\nAll credentials present. Testing the connection to VDP...\n")
    try:
        result = visa_client.ping()
    except visa_client.VisaCredentialError as exc:
        print(f"  CREDENTIAL PROBLEM: {exc}")
        return 1
    except visa_client.VisaAPIError as exc:
        print(f"  VDP REFUSED ({exc.status}): {exc.detail}")
        # 403 is an entitlement wall, not a broken setup - worth saying so,
        # because it is not something to keep debugging.
        return 0 if exc.status == 403 else 1

    print(f"  Connected. VDP replied: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
