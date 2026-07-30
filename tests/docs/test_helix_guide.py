from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
HELIX_GUIDE = PROJECT_ROOT / "docs" / "guides" / "editors" / "helix.md"


def test_helix_recommended_recipe_routes_supported_einf_features() -> None:
    guide = HELIX_GUIDE.read_text(encoding="utf-8")

    assert "display-inlay-hints = true" in guide
    assert '{ name = "pyright", except-features = ["inlay-hints"] }' in guide
    assert (
        '{ name = "einf-lsp", only-features = ["diagnostics", "inlay-hints"] }' in guide
    )


def test_helix_guide_records_unroutable_feature_tradeoffs() -> None:
    guide = HELIX_GUIDE.read_text(encoding="utf-8")

    assert 'except-features = ["hover", "inlay-hints"]' in guide
    assert "disables\nhover from the primary Python server" in guide
    assert "does not currently support the LSP semantic-token feature" in guide
