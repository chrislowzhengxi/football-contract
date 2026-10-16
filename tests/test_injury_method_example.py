"""Worked injury examples: picked by fixed rules from the saved episodes and consistent with them."""
import pandas as pd
import pytest

from src.analysis import injury_method_example as ime


def test_examples_follow_the_rules_and_exist():
    if not ime.EPISODES.exists() or not ime.EXAMPLE_MD.exists():
        pytest.skip("outputs not built")
    E = pd.read_csv(ime.EPISODES)
    ex = ime.pick(E)
    assert len(ex) == 3 and all(ep.usable for _, _, ep in ex)
    assert any(ep.confirmed_injury_games >= 2 for _, _, ep in ex)
    assert any(ep.appearance_conflicts >= 1 for _, _, ep in ex)
    md = ime.EXAMPLE_MD.read_text()
    for _, _, ep in ex:
        assert f"{ep.current_share:.3f}" in md and f"{ep.fit_adjusted_share:.3f}" in md
