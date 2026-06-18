"""
Scenario-calendar peak-demand simulation
=========================================

The peak can't be grown by a rate — it's recomputed from a calendar.
We build a handful of plausible calendars per year by perturbing Y1, run the
win-rate Monte Carlo on each, pool the peaks, and read a percentile.

The code below follows the 9 steps directly. Events are held as plain arrays
(a small table), not objects — so you can see the data at every step.

A "calendar" is just three arrays of equal length (one row per event):
    start[i], end[i]  : on-hire day range (day 0 = 1 Oct of FY1), inclusive
    qty[i, :]         : units of each asset type that event needs
Asset columns are ordered by ASSETS.
"""

import numpy as np

ASSETS = ["cabins", "buggies", "barriers", "fencing", "flooring"]

# Months in season order (Oct..Sep) and their lengths, used to place new events.
MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MONTH_LEN = [31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30]
MONTH_START = np.cumsum([0] + MONTH_LEN[:-1])  # day offset of each month's 1st


# --------------------------------------------------------------------------- #
# The peak, for ONE calendar, under the win-rate Monte Carlo  (step 6)
# --------------------------------------------------------------------------- #
def simulate_peaks(start, end, qty, win_rate, n_trials, rng, horizon):
    """Coin-flip each event n_trials times; for the events that 'sign', walk the
    calendar and record the highest concurrent demand. Returns the peak for
    every trial: an array of shape (n_trials, n_assets).

    Vectorised, but the logic is literal:
      - on_hire[i, d]  = 1 if event i is on hire on day d
      - signed[t, i]   = 1 if event i won in trial t  (the coin flip)
      - demand[t, d]   = sum over signed events of their qty on that day
      - peak[t]        = max over days
    """
    n_events = len(start)

    # on_hire[i, d] : is event i on hire on day d?   (n_events x horizon)
    days = np.arange(horizon)
    on_hire = (days[None, :] >= start[:, None]) & (days[None, :] <= end[:, None])
    on_hire = on_hire.astype(np.float32)

    # signed[t, i] : did event i sign in trial t?   (n_trials x n_events)
    signed = (rng.random((n_trials, n_events)) < win_rate).astype(np.float32)

    # Per asset: daily demand = signed events' units summed by day; peak = max day.
    peaks = np.empty((n_trials, len(ASSETS)), dtype=np.float32)
    for a in range(len(ASSETS)):
        units_per_day = on_hire * qty[:, a][:, None]  # (n_events x horizon)
        demand = signed @ units_per_day  # (n_trials x horizon)
        peaks[:, a] = demand.max(axis=1)
    return peaks


# --------------------------------------------------------------------------- #
# Place new events on a copy of the spine  (steps 3, 4, 5)
# --------------------------------------------------------------------------- #
def add_new_events(start, end, qty, n_new, month_weights, archetype, rng):
    """Return a new calendar = spine + n_new archetype events, each placed in a
    month drawn from month_weights (heavily Oct–Jan) on a random day."""
    if n_new == 0:
        return start, end, qty

    arch_dur, arch_qty = archetype
    new_start, new_end, new_qty = [], [], []
    for _ in range(n_new):
        m = rng.choice(len(MONTHS), p=month_weights)  # pick a month
        day = MONTH_START[m] + rng.integers(0, MONTH_LEN[m])  # pick a day in it
        new_start.append(day)
        new_end.append(day + arch_dur - 1)
        new_qty.append(arch_qty)

    start = np.concatenate([start, new_start])
    end = np.concatenate([end, new_end])
    qty = np.vstack([qty, new_qty])
    return start, end, qty


# --------------------------------------------------------------------------- #
# One year: pool peaks over many placements  (steps 4–7)
# --------------------------------------------------------------------------- #
def simulate_year(start, end, qty, n_new, win_rate, month_weights, archetype, n_placements, n_trials, rng, horizon):
    """Build n_placements calendars (each with new events placed differently),
    run the win-rate sim on each, and pool all the peaks together."""
    pooled = []
    for _ in range(n_placements):
        s, e, q = add_new_events(start, end, qty, n_new, month_weights, archetype, rng)
        pooled.append(simulate_peaks(s, e, q, win_rate, n_trials, rng, horizon))
    return np.concatenate(pooled)  # (n_placements*n_trials x assets)


# --------------------------------------------------------------------------- #
# Five-year roll-forward  (steps 1, 2, 8, 9)
# --------------------------------------------------------------------------- #
def run(
    start,
    end,
    qty,
    *,
    win_rate,
    growth,
    month_weights,
    archetype,
    coverage_pct,
    horizon,
    years=5,
    n_placements=300,
    n_trials=4000,
    seed=7,
):
    """Roll the calendar forward, sizing fleet to the chosen percentile each year.

    coverage_pct : percentile to OWN, e.g. 85.
    Prints a small table per year and returns the per-year results.
    """
    rng = np.random.default_rng(seed)
    month_weights = np.asarray(month_weights, float)
    month_weights = month_weights / month_weights.sum()

    count = len(start)
    fleet_held = np.zeros(len(ASSETS))
    results = []

    for yr in range(1, years + 1):
        # step 2: how many whole new events this year (Y1 has none — it's real)
        n_new = 0 if yr == 1 else round(count * (1 + growth)) - round(count)

        # steps 4–7: pooled peak distribution
        peaks = simulate_year(
            start, end, qty, n_new, win_rate, month_weights, archetype, n_placements, n_trials, rng, horizon
        )

        # step 8: read the percentile -> required fleet -> stepped capex
        required = np.ceil(np.percentile(peaks, coverage_pct, axis=0))
        capex = np.maximum(0, required - fleet_held)
        fleet_held = np.maximum(fleet_held, required)

        results.append({"year": yr, "new_events": n_new, "peaks": peaks, "required": required, "capex": capex})

        # step 9: commit this year's new events to the spine (place them once)
        if n_new:
            start, end, qty = add_new_events(start, end, qty, n_new, month_weights, archetype, rng)
        count = round(count * (1 + growth))

    _print(results)
    return results


def _print(results):
    pct = lambda p, peaks: np.percentile(peaks, p, axis=0)
    for r in results:
        print(f"\n=== YEAR {r['year']}  (+{r['new_events']} new events) ===")
        print("        " + "".join(f"{a:>10}" for a in ASSETS))
        for p in (50, 85, 95):
            print(f"  P{p:<4} " + "".join(f"{x:>10.0f}" for x in pct(p, r["peaks"])))
        print("  fleet " + "".join(f"{x:>10.0f}" for x in r["required"]))
        print("  buy   " + "".join(f"{x:>10.0f}" for x in r["capex"]))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    # (name, hire start, hire end, cabin, buggies, barriers, fencing, flooring)
    rows = [
        ("Saadiyat Nights", 1, 165, 10, 10, 0, 0, 280),
        ("BRED X", 22, 44, 8, 0, 0, 0, 0),
        ("MOTN DHA", 34, 60, 8, 6, 350, 550, 0),
        ("No Art", 29, 62, 24, 6, 0, 0, 200),
        ("Tanweer", 39, 60, 0, 8, 900, 50, 200),
        ("MOTN Al Ain", 33, 74, 18, 6, 350, 550, 0),
        ("F1 Afterparty", 34, 73, 29, 8, 714, 1180, 178),
        ("MOTN Abu Dhabi", 42, 105, 20, 10, 700, 1100, 200),
        ("Yas Winterfest", 57, 87, 6, 4, 644, 760, 60),
        ("Animenia", 120, 142, 8, 4, 288, 192, 0),
        ("DAZ", 165, 200, 15, 4, 775, 225, 60),
        ("BRED Abu Dhabi", 193, 215, 22, 6, 1500, 500, 0),
    ]
    start = np.array([r[1] for r in rows])
    end = np.array([r[2] for r in rows])
    qty = np.array([r[3:] for r in rows], dtype=float)

    # Seasonal weights for placing NEW events (Oct..Sep) — concentrated Oct–Jan.
    weights = [0.10, 0.22, 0.24, 0.16, 0.08, 0.07, 0.07, 0, 0, 0, 0, 0]

    # A "typical" new event: (duration_days, qty array) — median festival.
    archetype = (25, np.array([14, 6, 600, 500, 120], dtype=float))

    t0 = time.perf_counter()
    run(
        start,
        end,
        qty,
        win_rate=0.65,
        growth=0.10,
        month_weights=weights,
        archetype=archetype,
        coverage_pct=85,
        horizon=240,
        years=5,
        n_placements=300,
        n_trials=4000,
        seed=7,
    )
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
