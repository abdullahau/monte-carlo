"""
Scenario-calendar peak-demand simulation
=======================================================

Nested Monte Carlo for sizing an event-rental fleet when only Year-1 is real.

    OUTER loop  ─ placement sampling : build plausible Y2..Y5 calendars by
                  perturbing the Y1 spine (new events placed by seasonal weight).
    INNER loop  ─ win-rate trials    : per calendar, flip a weighted coin per
                  event and read the peak concurrent demand.

The peak is is re-derived from each constructed calendar, so it moves in steps
(a new event either collides with the Oct–Jan cluster or lands in a trough) —
which is the whole point.

For a fixed calendar with an (n_events x horizon) on-hire mask M and per-asset
quantities Q, a batch of T win-trials is one Bernoulli draw  W (T x n_events)
followed by a single BLAS matmul per asset type:
        daily_demand = (W * Q[:,a]) @ M          # (T x horizon)
        peaks_a      = daily_demand.max(axis=1)  # (T,)

MODELLING CHOICE (roll-forward)
-------------------------------
Each year's distribution samples placement uncertainty around the *current*
spine. To advance the spine to the next year we COMMIT this year's new events
deterministically at their highest-weight months (the modal calendar). Thus the
spine grows realistically while the per-year peak distribution still reflects
the uncertainty of where the *incremental* events land.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Order matters: every quantity vector is indexed by this list.
ASSET_TYPES: list[str] = ["cabins", "buggies", "barriers", "fencing", "flooring"]
N_ASSETS: int = len(ASSET_TYPES)


# Data model
@dataclass
class Event:
    """An event's on-hire footprint. Days are integer offsets from season start
    (day 0 = 1 Oct of FY1). on_hire spans mobilization -> demobilization."""

    name: str
    on_hire_start: int
    on_hire_end: int  # inclusive
    qty: np.ndarray  # shape (N_ASSETS,), float32

    @property
    def duration(self) -> int:
        return self.on_hire_end - self.on_hire_start + 1


def make_event(
    name: str,
    start: int,
    end: int,
    cabins: int = 0,
    buggies: int = 0,
    barriers: int = 0,
    fencing: int = 0,
    flooring: int = 0,
) -> Event:
    return Event(
        name,
        start,
        end,
        np.array([cabins, buggies, barriers, fencing, flooring], dtype=np.float32),
    )


# Calendar -> matrices
def build_matrices(events: Sequence[Event], horizon: int):
    """Return on-hire mask M (n_events x horizon, float32) and quantity matrix
    Q (n_events x N_ASSETS, float32). M[i,d]=1 iff event i is on hire on day d."""
    n: int = len(events)
    M = np.zeros((n, horizon), dtype=np.float32)
    Q = np.empty((n, N_ASSETS), dtype=np.float32)
    for i, ev in enumerate(events):
        s = max(0, ev.on_hire_start)
        e = min(horizon - 1, ev.on_hire_end)
        if e >= s:
            M[i, s : e + 1] = 1.0
        Q[i] = ev.qty
    return M, Q


# Inner engine: win-rate trials for ONE fixed calendar
def precompute_asset_masks(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Fold quantities into the on-hire mask, per asset:
        QM[a] = Q[:,a,None] * M     -> (N_ASSETS x n_events x horizon)
    Built once per calendar; reused for every trial batch. float32."""
    # (n_events x N_ASSETS x 1) * (n_events x 1 x horizon) -> (n_events x N_ASSETS x horizon)
    QM = Q[:, :, None] * M[:, None, :]
    return np.ascontiguousarray(QM.transpose(1, 0, 2))  # (A x n_events x horizon)


def peaks_for_calendar(QM: np.ndarray, win_rate: float, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Return peaks array of shape (n_trials, N_ASSETS) for a pre-folded QM.

    All assets in ONE batched contraction + ONE max reduction:
        W   : (T x n_events)                         Bernoulli win mask
        D   : einsum('tn,anh->tah') -> (A x T x horizon)   daily demand, all assets
        peak: D.max over horizon     -> (A x T)
    No per-asset Python loop, no per-trial loop.
    """
    n_events = QM.shape[1]
    W = (rng.random((n_trials, n_events), dtype=np.float32) < win_rate).astype(np.float32)  # (T x n_events)
    # batched matmul over assets: (A x T x n_events) @ (A x n_events x horizon)
    # implemented as einsum to keep one BLAS-backed call.
    D = np.einsum("tn,anh->ath", W, QM, optimize=True)  # (A x T x horizon)
    return D.max(axis=2).T  # (T x A)


# Placement of incremental events
# Day offset of the 1st of each season month (FY1 = Oct..Sep), non-leap.
_MONTH_ORDER: list[int] = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_DAYS_IN: dict[int, int] = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def _month_start_offsets() -> dict:
    off, acc = {}, 0
    for m in _MONTH_ORDER:
        off[m] = acc
        acc += _DAYS_IN[m]
    return off


_MONTH_OFFSET = _month_start_offsets()


def place_new_events(
    spine: list[Event],
    n_new: int,
    month_weights: np.ndarray,
    archetype: Event,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> list[Event]:
    """Append n_new archetype-shaped events to a COPY of the spine, each placed
    in a month drawn from month_weights (indexed by _MONTH_ORDER).
    deterministic=True places every new event in the single highest-weight
    month (used to commit the modal calendar when rolling the spine forward)."""
    cal = list(spine)
    if n_new <= 0:
        return cal
    w = np.asarray(month_weights, dtype=np.float64)
    w /= w.sum()
    dur = archetype.duration
    for k in range(n_new):
        if deterministic:
            m = _MONTH_ORDER[int(w.argmax())]
        else:
            m = _MONTH_ORDER[rng.choice(len(_MONTH_ORDER), p=w)]
        span = _DAYS_IN[m]
        day_in = 0 if deterministic else int(rng.integers(0, max(1, span - 1)))
        start = _MONTH_OFFSET[m] + day_in
        cal.append(
            Event(
                f"{archetype.name}+{k + 1}",
                start,
                start + dur - 1,
                archetype.qty.copy(),
            )
        )
    return cal


# One scenario year: pool peaks across placements
def scenario_year(
    spine: list[Event],
    n_new: int,
    win_rate: float,
    month_weights: np.ndarray,
    archetype: Event,
    n_placements: int,
    n_trials: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return pooled peaks of shape (n_placements * n_trials, N_ASSETS).

    Outer loop = placements (cheap). Inner = vectorized win trials.
    Memory is capped at one (n_trials x horizon) buffer at a time.
    """
    pooled = np.empty((n_placements * n_trials, N_ASSETS), dtype=np.float32)
    for p in range(n_placements):
        cal = place_new_events(spine, n_new, month_weights, archetype, rng)
        M, Q = build_matrices(cal, horizon)
        QM = precompute_asset_masks(M, Q)
        pooled[p * n_trials : (p + 1) * n_trials] = peaks_for_calendar(QM, win_rate, n_trials, rng)
    return pooled


# Five-year roll-forward
def roll_forward(
    y1_events: list[Event],
    *,
    win_rate: float,
    growth: float,
    month_weights: np.ndarray,
    archetype: Event,
    coverage_pct: dict[str, float],
    horizon: int,
    years: int = 5,
    n_placements: int = 300,
    n_trials: int = 4000,
    percentiles: tuple[int, int, int] = (50, 85, 95),
    seed: int = 7,
) -> dict:
    """Run the scenario simulation for each year and size the fleet.

    coverage_pct: per-asset percentile to OWN (e.g. {"cabins":85, ...}).
    Returns a dict with, per year: the requested percentile bands and the
    required fleet (at each asset's coverage_pct), plus stepped capex units.
    """
    rng = np.random.default_rng(seed)
    spine = list(y1_events)
    count = float(len(y1_events))
    fleet_held = np.zeros(N_ASSETS, dtype=np.float64)

    out = {"asset_types": ASSET_TYPES, "years": []}
    for yr in range(1, years + 1):
        # whole-event increment (Y1 has no increment: spine already real)
        new_count = 0 if yr == 1 else int(round(count * (1 + growth)) - round(count))
        pooled = scenario_year(
            spine,
            new_count,
            win_rate,
            month_weights,
            archetype,
            n_placements,
            n_trials,
            horizon,
            rng,
        )

        # percentile bands (for reporting) and the owned-fleet point
        bands = {f"P{p}": np.percentile(pooled, p, axis=0) for p in percentiles}
        required = np.array(
            [np.percentile(pooled[:, a], coverage_pct[ASSET_TYPES[a]]) for a in range(N_ASSETS)],
            dtype=np.float64,
        )
        required = np.ceil(required)  # whole units
        capex_units = np.maximum(0.0, required - fleet_held)
        fleet_held = np.maximum(fleet_held, required)

        out["years"].append(
            {
                "year": yr,
                "new_events": new_count,
                "bands": {k: v.round(1) for k, v in bands.items()},
                "required_fleet": required.astype(int),
                "capex_units": capex_units.astype(int),
                "fleet_held": fleet_held.astype(int).copy(),
            }
        )

        # commit this year's incremental events to the spine (modal placement)
        if new_count > 0:
            spine = place_new_events(spine, new_count, month_weights, archetype, rng, deterministic=True)
        count = round(count * (1 + growth))
    return out


# Reporting helpers
def print_report(res: dict) -> None:
    at = res["asset_types"]
    for y in res["years"]:
        print(f"\n=== YEAR {y['year']}  (+{y['new_events']} new events) ===")
        hdr = "  " + "".join(f"{a:>11}" for a in at)
        print(hdr)
        for k, v in y["bands"].items():
            print(f"{k:<4}" + "".join(f"{x:>11.1f}" for x in v))
        print("req " + "".join(f"{x:>11d}" for x in y["required_fleet"]))
        print("buy " + "".join(f"{x:>11d}" for x in y["capex_units"]))


def to_dataframe(res: dict) -> pd.DataFrame:
    """Flatten to a tidy DataFrame ready to write to xlsx/csv for the workbook."""

    rows = []
    for y in res["years"]:
        for a, name in enumerate(res["asset_types"]):
            row = {
                "year": y["year"],
                "asset": name,
                "new_events": y["new_events"],
                "required_fleet": int(y["required_fleet"][a]),
                "capex_units": int(y["capex_units"][a]),
                "fleet_held": int(y["fleet_held"][a]),
            }
            for k, v in y["bands"].items():
                row[k] = float(v[a])
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time

    # Illustrative FY1 calendar (day 0 = 1 Oct). on_hire = mob start -> demob end.
    y1: list[Event] = [
        make_event("Saadiyat Nights", 1, 165, cabins=10, buggies=10, flooring=280),
        make_event("BRED X", 22, 44, cabins=8),
        make_event("MOTN DHA", 34, 60, cabins=8, buggies=6, barriers=350, fencing=550),
        make_event("No Art", 29, 62, cabins=24, buggies=6, flooring=200),
        make_event("Tanweer", 39, 60, buggies=8, barriers=900, fencing=50, flooring=200),
        make_event("MOTN Al Ain", 33, 74, cabins=18, buggies=6, barriers=350, fencing=550),
        make_event("F1 Afterparty", 34, 73, cabins=29, buggies=8, barriers=714, fencing=1180, flooring=178),
        make_event("MOTN Abu Dhabi", 42, 105, cabins=20, buggies=10, barriers=700, fencing=1100, flooring=200),
        make_event("Yas Winterfest", 57, 87, cabins=6, buggies=4, barriers=644, fencing=760, flooring=60),
        make_event("Animenia", 120, 142, cabins=8, buggies=4, barriers=288, fencing=192),
        make_event("DAZ", 165, 200, cabins=15, buggies=4, barriers=775, fencing=225, flooring=60),
        make_event("BRED Abu Dhabi", 193, 215, cabins=22, buggies=6, barriers=1500, fencing=500),
    ]

    # Seasonal placement weights for NEW events, indexed by _MONTH_ORDER
    # (Oct,Nov,Dec,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep) — concentrated Oct–Jan.
    weights: np.ndarray = np.array([0.10, 0.22, 0.24, 0.16, 0.08, 0.07, 0.07, 0.00, 0.00, 0.00, 0.00, 0.00])

    # Archetype for incremental events: a median-sized festival.
    archetype: Event = make_event("NewEvent", 0, 24, cabins=14, buggies=6, barriers=600, fencing=500, flooring=120)

    coverage: dict[str, float] = {
        "cabins": 85,
        "buggies": 85,
        "barriers": 85,
        "fencing": 85,
        "flooring": 85,
    }

    t0 = time.perf_counter()
    res = roll_forward(
        y1,
        win_rate=0.65,
        growth=0.10,
        month_weights=weights,
        archetype=archetype,
        coverage_pct=coverage,
        horizon=240,
        years=5,
        n_placements=300,
        n_trials=4000,
        seed=7,
    )
    dt = time.perf_counter() - t0

    print_report(res)
    total_samples = 300 * 4000 * 5
    print(f"\nDone in {dt:.2f}s  ({total_samples:,} peak evaluations across 5 years)")
