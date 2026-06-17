import random

import matplotlib.pyplot as plt

# The Year-1 "fingerprint"
# Each event:
# (name, on-hire start day, on-hire end day,
# cabins, buggies, barriers, fencing, flooring)
# day 0 = Oct 1
events = [
    ("Saadiyat Nights", 1, 157, 10, 10, 0, 0, 280),
    ("BRED X", 22, 43, 8, 0, 0, 0, 0),
    ("MOTN DHA", 34, 56, 8, 6, 350, 550, 0),
    ("No Art", 36, 57, 24, 6, 0, 0, 200),
    ("Tanweer Festival", 44, 56, 0, 8, 900, 50, 200),
    ("MOTN AAN", 40, 66, 18, 6, 350, 550, 0),
    ("Keinemusik", 57, 69, 29, 8, 714, 1180, 178),
    ("MOTN AUH", 50, 102, 20, 10, 700, 1100, 200),
    ("Yas Winterfest", 56, 88, 6, 4, 644, 760, 60),
    ("Animenia", 118, 140, 8, 4, 288, 192, 0),
    ("DAZ", 170, 200, 15, 4, 775, 225, 60),
    ("BRED", 196, 210, 22, 6, 1500, 500, 0),
]

ASSETS = [
    "Cabins",
    "Buggies",
    "Barriers",
    "Fencing",
    "Flooring",
]
HORIZON = 212  # 1st October to 30th April
WIN_P = 0.50  # probability of winning each event
TRIALS = 50_000


def peak_of_trial(active_events):
    """
    Returns the peak utilization for each asset during one trial.
    """
    peaks = {asset: 0 for asset in ASSETS}

    for d in range(HORIZON):
        loads = {
            "Cabins": 0,
            "Buggies": 0,
            "Barriers": 0,
            "Fencing": 0,
            "Flooring": 0,
        }

        for ev in active_events:
            _, start, end, cabins, buggies, barriers, fencing, flooring = ev

            if start <= d <= end:
                loads["Cabins"] += cabins
                loads["Buggies"] += buggies
                loads["Barriers"] += barriers
                loads["Fencing"] += fencing
                loads["Flooring"] += flooring

        for asset in ASSETS:
            peaks[asset] = max(peaks[asset], loads[asset])

    return peaks


# Store peak values from each trial
trial_peaks = {asset: [] for asset in ASSETS}

for _ in range(TRIALS):
    active = [ev for ev in events if random.random() < WIN_P]
    peaks = peak_of_trial(active)

    for asset in ASSETS:
        trial_peaks[asset].append(peaks[asset])


def pct(p):
    return peaks[int(p * TRIALS)]


def percentile(values, p):
    values = sorted(values)
    return values[int(p * len(values))]


all_win_peaks = peak_of_trial(events)

print(f"Win rate {WIN_P:.0%}, {TRIALS:,} trials")
print()

for asset in ASSETS:
    vals = sorted(trial_peaks[asset])

    mean_peak = sum(vals) / len(vals)

    print(asset.upper())
    print(f"  Mean : {mean_peak:8.0f}")
    print(f"  P50  : {percentile(vals, 0.50):8d}")
    print(f"  P60  : {percentile(vals, 0.60):8d}")
    print(f"  P65  : {percentile(vals, 0.65):8d}")
    print(f"  P75  : {percentile(vals, 0.75):8d}")
    print(f"  P85  : {percentile(vals, 0.85):8d}")
    print(f"  P95  : {percentile(vals, 0.95):8d}")
    print(f"  P100 : {all_win_peaks[asset]:8d}")
    print()

# Plotting
fig, axes = plt.subplots(
    1,
    len(ASSETS),
    figsize=(14, 2.8),
    constrained_layout=True,
)

for ax, asset in zip(axes, ASSETS):
    vals = trial_peaks[asset]

    ax.hist(vals, bins=30)

    p50 = percentile(vals, 0.50)
    p85 = percentile(vals, 0.85)
    p95 = percentile(vals, 0.95)

    ax.axvline(p50, linestyle="--", linewidth=1)
    ax.axvline(p85, linestyle="--", linewidth=1)
    ax.axvline(p95, linestyle=":", linewidth=1)

    ax.set_title(asset, fontsize=10)
    ax.tick_params(labelsize=8)

    ax.set_ylabel("")

    if asset != ASSETS[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Freq.", fontsize=8)

    ax.set_xlabel("Peak", fontsize=8)

fig.suptitle(
    f"Peak Demand Distribution ({TRIALS:,} trials, {WIN_P:.0%} win probability)",
    fontsize=12,
)

plt.show()
