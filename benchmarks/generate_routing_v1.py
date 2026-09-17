from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "data"

CALIBRATION_PATH = DATA_DIR / "routing_v1_calibration.json"
TEST_PATH = DATA_DIR / "routing_v1_test.json"

CANDIDATES = [
    "billing",
    "technical support",
    "sales",
    "account access",
    "cancellation",
    "shipping",
]

INTENT_DEFINITIONS = {
    "billing": (
        "Charges, invoices, payments, receipts, "
        "refunds, and unexpected fees."
    ),
    "technical support": (
        "Product failures, crashes, errors, broken "
        "features, synchronization, or other technical problems."
    ),
    "sales": (
        "Pre-purchase questions about pricing, quotes, "
        "demos, plans, licenses, or volume purchases."
    ),
    "account access": (
        "Login, password, authentication, locked-account, "
        "or two-factor-access problems."
    ),
    "cancellation": (
        "Requests to end, discontinue, or stop renewal "
        "of an existing subscription or service."
    ),
    "shipping": (
        "Delivery, parcels, tracking, courier status, "
        "delivery address, or arrival-time questions."
    ),
}


CALIBRATION_FAMILIES = {
    "billing": [
        (
            "duplicate_charge",
            "I was charged {value} twice on my card.",
            ["$19", "$42", "$73", "€35", "£28"],
        ),
        (
            "wrong_invoice",
            "The invoice for order {value} has the wrong total.",
            ["A104", "B227", "C318", "D441", "E592"],
        ),
        (
            "missing_receipt",
            "A payment of {value} was taken, but no receipt arrived.",
            ["$16", "$31", "$58", "€44", "£67"],
        ),
        (
            "refund_charge",
            "I need a refund for the duplicate charge on order {value}.",
            ["F120", "G245", "H366", "J487", "K508"],
        ),
    ],
    "technical support": [
        (
            "app_crash",
            "The app crashes every time I try to {value}.",
            [
                "open a project",
                "save a document",
                "export a file",
                "start a call",
                "upload an image",
            ],
        ),
        (
            "error_code",
            "The program shows error code {value} and stops working.",
            ["E17", "E42", "A09", "X31", "R88"],
        ),
        (
            "feature_loading",
            "The {value} screen never finishes loading.",
            [
                "dashboard",
                "reports",
                "settings",
                "analytics",
                "workspace",
            ],
        ),
        (
            "upload_failure",
            "Files fail every time I try to upload a {value}.",
            [
                "PDF",
                "spreadsheet",
                "photo",
                "presentation",
                "text document",
            ],
        ),
    ],
    "sales": [
        (
            "seat_quote",
            "Can you give me a price quote for {value} seats?",
            ["12", "25", "40", "75", "120"],
        ),
        (
            "demo_request",
            "I'd like to see a demo for our {value} team.",
            [
                "engineering",
                "finance",
                "design",
                "operations",
                "research",
            ],
        ),
        (
            "enterprise_price",
            "What would the enterprise plan cost for a company of {value} people?",
            ["30", "60", "90", "150", "300"],
        ),
        (
            "volume_discount",
            "Do you offer a discount if we purchase {value} licenses?",
            ["20", "50", "80", "100", "250"],
        ),
    ],
    "account access": [
        (
            "password_loop",
            "The password reset keeps sending me back to the {value} page.",
            [
                "sign-in",
                "verification",
                "welcome",
                "profile",
                "security",
            ],
        ),
        (
            "locked_out",
            "I'm locked out after {value} failed login attempts.",
            ["three", "four", "five", "six", "several"],
        ),
        (
            "two_factor",
            "My two-factor code is not arriving on my {value}.",
            [
                "phone",
                "email",
                "authenticator",
                "new device",
                "tablet",
            ],
        ),
        (
            "email_change",
            "I cannot sign in after changing my email to {value}.",
            [
                "a new work address",
                "a personal address",
                "my company address",
                "a new domain",
                "another mailbox",
            ],
        ),
    ],
    "cancellation": [
        (
            "end_subscription",
            "I want to end my subscription {value}.",
            [
                "today",
                "this week",
                "at the end of the month",
                "before the next cycle",
                "immediately",
            ],
        ),
        (
            "stop_renewal",
            "Please stop my plan from renewing {value}.",
            [
                "next month",
                "next week",
                "after this cycle",
                "on the next date",
                "after the current term",
            ],
        ),
        (
            "close_plan",
            "I no longer need the service and want the plan closed {value}.",
            [
                "now",
                "after Friday",
                "at month end",
                "after this period",
                "before renewal",
            ],
        ),
        (
            "terminate_service",
            "Please terminate my current service {value}.",
            [
                "as soon as possible",
                "after this month",
                "after the paid period",
                "before renewal",
                "at the end of today",
            ],
        ),
    ],
    "shipping": [
        (
            "package_missing",
            "My package was supposed to arrive {value}, but it is still missing.",
            [
                "yesterday",
                "Monday",
                "two days ago",
                "this morning",
                "last Friday",
            ],
        ),
        (
            "tracking_stuck",
            "The tracking page has been stuck on {value} for days.",
            [
                "in transit",
                "label created",
                "at depot",
                "processing",
                "outbound facility",
            ],
        ),
        (
            "address_update",
            "Can I change the delivery address to {value} before the parcel arrives?",
            [
                "my office",
                "my new apartment",
                "a nearby pickup point",
                "my parents' house",
                "another street",
            ],
        ),
        (
            "arrival_date",
            "When should parcel {value} reach me?",
            ["P104", "P225", "P347", "P468", "P589"],
        ),
    ],
}


TEST_FAMILIES = {
    "billing": [
        (
            "unexpected_debit",
            "Why did my card get debited {value} more than I expected?",
            ["$11", "$26", "$49", "€32", "£54"],
        ),
        (
            "unknown_fee",
            "The receipt for order {value} contains a fee I do not recognize.",
            ["L611", "M732", "N843", "Q954", "R165"],
        ),
        (
            "failed_payment",
            "Payment for order {value} failed, but the money still left my bank.",
            ["S216", "T327", "U438", "V549", "W650"],
        ),
        (
            "invoice_extra",
            "Please explain the extra {value} on my latest invoice.",
            ["$7", "$13", "$24", "€18", "£21"],
        ),
    ],
    "technical support": [
        (
            "screen_freeze",
            "The application freezes whenever I open the {value}.",
            [
                "editor",
                "calendar",
                "search panel",
                "history view",
                "media viewer",
            ],
        ),
        (
            "sync_failure",
            "Synchronization stops when it reaches {value}.",
            [
                "my laptop",
                "the cloud folder",
                "the final file",
                "my phone",
                "the shared workspace",
            ],
        ),
        (
            "api_error",
            "The API returns {value} whenever my integration sends a request.",
            [
                "status 500",
                "status 502",
                "an invalid response",
                "a timeout",
                "an empty response",
            ],
        ),
        (
            "desktop_closes",
            "The desktop client closes by itself while I {value}.",
            [
                "edit a document",
                "join a meeting",
                "open settings",
                "search files",
                "switch workspaces",
            ],
        ),
    ],
    "sales": [
        (
            "compare_plans",
            "We're evaluating the product and want to compare plans for {value} users.",
            ["15", "35", "55", "85", "140"],
        ),
        (
            "annual_quote",
            "Could you prepare an annual quote for our {value} department?",
            [
                "marketing",
                "legal",
                "engineering",
                "support",
                "data",
            ],
        ),
        (
            "buy_licenses",
            "Our company is considering buying {value} user licenses.",
            ["18", "45", "70", "110", "200"],
        ),
        (
            "team_trial",
            "Can we arrange a product trial for a team of {value} people before buying?",
            ["10", "20", "50", "80", "160"],
        ),
    ],
    "account access": [
        (
            "new_phone",
            "I changed phones and can no longer authenticate on my {value}.",
            [
                "profile",
                "workspace",
                "company login",
                "user page",
                "member portal",
            ],
        ),
        (
            "reset_expired",
            "The password recovery link says it expired after {value}.",
            [
                "a few minutes",
                "one attempt",
                "I opened it",
                "I switched browsers",
                "I changed devices",
            ],
        ),
        (
            "credentials_rejected",
            "My credentials are rejected even though I can use them on {value}.",
            [
                "another browser",
                "my old laptop",
                "the mobile app",
                "a second device",
                "the web portal",
            ],
        ),
        (
            "verification_block",
            "I cannot get past the identity verification step on my {value}.",
            [
                "new laptop",
                "work computer",
                "phone",
                "tablet",
                "home PC",
            ],
        ),
    ],
    "cancellation": [
        (
            "membership_end",
            "I don't need the membership anymore; please end it {value}.",
            [
                "today",
                "after this month",
                "after the current period",
                "next Friday",
                "before the next charge",
            ],
        ),
        (
            "disable_renewal",
            "Please turn off the next renewal {value}.",
            [
                "for my plan",
                "for this subscription",
                "on my current service",
                "for my membership",
                "on this contract",
            ],
        ),
        (
            "discontinue_plan",
            "I would like to discontinue my current plan {value}.",
            [
                "immediately",
                "after this cycle",
                "at the end of the month",
                "after Sunday",
                "before it renews",
            ],
        ),
        (
            "end_after_date",
            "Keep the service until {value}, then end it.",
            [
                "Friday",
                "month end",
                "the renewal date",
                "tomorrow",
                "the paid period finishes",
            ],
        ),
    ],
    "shipping": [
        (
            "parcel_delayed",
            "The parcel is already {value} late. When will it arrive?",
            [
                "one day",
                "two days",
                "three days",
                "a week",
                "several days",
            ],
        ),
        (
            "tracking_no_update",
            "Tracking number {value} has not updated since it left the warehouse.",
            ["ZX101", "ZX202", "ZX303", "ZX404", "ZX505"],
        ),
        (
            "wrong_location",
            "The courier says parcel {value} was delivered somewhere else.",
            ["YT116", "YT227", "YT338", "YT449", "YT550"],
        ),
        (
            "delivery_progress",
            "Where is parcel {value}, and how long until it reaches my address?",
            ["RV121", "RV232", "RV343", "RV454", "RV565"],
        ),
    ],
}


def expand_split(
    split: str,
    families: dict,
    seed: int,
) -> list[dict]:
    examples = []

    for intent in CANDIDATES:
        intent_families = families[intent]

        if len(intent_families) != 4:
            raise RuntimeError(
                f"{intent}: expected exactly 4 template families."
            )

        for family_name, template, values in intent_families:
            if len(values) != 5:
                raise RuntimeError(
                    f"{intent}/{family_name}: expected 5 values."
                )

            for variant_index, value in enumerate(
                values,
                start=1,
            ):
                text = template.format(
                    value=value
                )

                examples.append(
                    {
                        "id": (
                            f"{split}_"
                            f"{intent.replace(' ', '_')}_"
                            f"{family_name}_"
                            f"{variant_index:02d}"
                        ),
                        "split": split,
                        "intent": intent,
                        "family": family_name,
                        "text": text,
                    }
                )

    random.Random(seed).shuffle(
        examples
    )

    return examples


def validate_split(
    split: str,
    examples: list[dict],
) -> None:
    if len(examples) != 120:
        raise RuntimeError(
            f"{split}: expected 120 examples, got {len(examples)}."
        )

    counts = Counter(
        example["intent"]
        for example in examples
    )

    for candidate in CANDIDATES:
        if counts[candidate] != 20:
            raise RuntimeError(
                f"{split}/{candidate}: expected 20, got {counts[candidate]}."
            )

    texts = [
        example["text"].strip().lower()
        for example in examples
    ]

    if len(texts) != len(set(texts)):
        raise RuntimeError(
            f"{split}: duplicate text detected."
        )

    for example in examples:
        lowered = example["text"].lower()

        for candidate in CANDIDATES:
            if candidate.lower() in lowered:
                raise RuntimeError(
                    "Exact candidate phrase leaked into input: "
                    f'{example["id"]}: {candidate!r}'
                )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_dataset(
    path: Path,
    split: str,
    seed: int,
    examples: list[dict],
) -> None:
    payload = {
        "benchmark": "routing_v1",
        "split": split,
        "seed": seed,
        "candidates": CANDIDATES,
        "intent_definitions": (
            INTENT_DEFINITIONS
        ),
        "examples": examples,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = expand_split(
        split="calibration",
        families=CALIBRATION_FAMILIES,
        seed=20260919,
    )

    test = expand_split(
        split="test",
        families=TEST_FAMILIES,
        seed=20260920,
    )

    validate_split(
        "calibration",
        calibration,
    )

    validate_split(
        "test",
        test,
    )

    calibration_texts = {
        example["text"].strip().lower()
        for example in calibration
    }

    test_texts = {
        example["text"].strip().lower()
        for example in test
    }

    overlap = (
        calibration_texts
        & test_texts
    )

    if overlap:
        raise RuntimeError(
            "Calibration/test exact text overlap detected."
        )

    calibration_families = {
        (
            example["intent"],
            example["family"],
        )
        for example in calibration
    }

    test_families = {
        (
            example["intent"],
            example["family"],
        )
        for example in test
    }

    if (
        calibration_families
        & test_families
    ):
        raise RuntimeError(
            "Calibration/test template-family overlap detected."
        )

    write_dataset(
        CALIBRATION_PATH,
        "calibration",
        20260919,
        calibration,
    )

    write_dataset(
        TEST_PATH,
        "test",
        20260920,
        test,
    )

    print("ROUTING V1 DATASET")
    print("==================")
    print()
    print(
        "Candidates:",
        ", ".join(CANDIDATES),
    )
    print()

    for split, examples, path in (
        (
            "calibration",
            calibration,
            CALIBRATION_PATH,
        ),
        (
            "test",
            test,
            TEST_PATH,
        ),
    ):
        counts = Counter(
            example["intent"]
            for example in examples
        )

        print(split.upper())
        print("-" * len(split))

        print(
            "Examples:",
            len(examples),
        )

        for candidate in CANDIDATES:
            print(
                f"  {candidate:<20}"
                f"{counts[candidate]}"
            )

        print(
            "SHA256:",
            sha256_file(path),
        )

        print(
            "Path:",
            path.relative_to(ROOT),
        )

        print()

    print(
        "Exact text overlap:",
        len(
            calibration_texts
            & test_texts
        ),
    )

    print(
        "Template-family overlap:",
        len(
            calibration_families
            & test_families
        ),
    )


if __name__ == "__main__":
    main()
