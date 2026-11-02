import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_provenance_records_exact_legacy_anchors() -> None:
    provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")

    for revision in (
        "4861aab3af39e721c1b5a8b27b72c4f6bebda888",
        "13e695325072d7ac02578823541d18cbc325b8eb",
        "a1d9f146f8327bb336f6348486c625a85155278f",
        "fc6f8f4c3d8ca7a1849a2ef67ff5fca8d285a6f0",
        "7b361031cbbde258b13d50142f3a817c1ee897c5",
    ):
        assert revision in provenance


def test_functional_checks_do_not_promote_performance_qualification() -> None:
    metadata = json.loads((ROOT / "MOD_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["evidence"]["qualification"] == "unqualified"
    assert metadata["evidence"]["performance_claims"] == []
    assert metadata["lifecycle"]["default_enabled"] is False
