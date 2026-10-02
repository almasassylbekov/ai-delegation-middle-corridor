"""Sanity checks: python tests.py"""
from dataclasses import replace

from model import Params, run_once


def total(rows):
    return sum(r["cost"] for r in rows)


def test_reproducible():
    p = Params(horizon_h=24 * 120)
    assert run_once(p, "D1", 7) == run_once(p, "D1", 7)


def test_common_random_numbers():
    """Same seed -> same set of movements across architectures."""
    p = Params(horizon_h=24 * 120)
    ids = {a: sorted(r["id"] for r in run_once(p, a, 3)) for a in ("D0", "D2", "ORACLE")}
    assert ids["D0"] == ids["D2"] == ids["ORACLE"]


def test_near_perfect_ai_matches_oracle():
    """With (almost) no prediction error and no bias, autonomy ~ oracle."""
    p = replace(Params(horizon_h=24 * 365), sigma_base=1e-4, sigma_heterogeneity=0.0,
                wait_min_sigma=1.0)
    for s in range(3):
        d2, orc = total(run_once(p, "D2", s)), total(run_once(p, "ORACLE", s))
        assert abs(d2 - orc) / orc < 0.01, (d2, orc)


def test_adaptation_beats_no_adaptation():
    p = Params(horizon_h=24 * 365)
    for s in range(3):
        assert total(run_once(p, "ORACLE", s)) < total(run_once(p, "D0", s))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
