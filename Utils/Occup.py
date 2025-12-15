import numpy as np
import matplotlib.pyplot as plt


def occupancy_probability(time_seconds):
    """Home occupancy profile P[occ=1|t] in [0,1] with only 2 blocks:

    - Week (Monday–Friday):
        * 0.3 between 8h and 18h
        * 0.9 the rest of the day

    - Weekend (Saturday–Sunday):
        * 0.6 between 8h and 18h
        * 0.9 the rest of the day
    """
    t = np.asarray(time_seconds, dtype=float)
    if t.ndim != 1:
        raise ValueError("time_seconds must be a 1D vector.")

    hours = (t / 3600.0) % 24.0
    days = (t / 86400.0).astype(int)

    # 0..4 = Monday–Friday, 5..6 = Saturday–Sunday
    is_weekend = (days % 7) >= 5
    in_work_hours = (hours >= 8.0) & (hours < 18.0)

    # Default: "outside 8–18h" block -> 0.9
    p = np.full_like(hours, 0.9, dtype=float)

    # Week, 8–18h block -> 0.3
    p[~is_weekend & in_work_hours] = 0.3

    # Weekend, 8–18h block -> 0.6
    p[is_weekend & in_work_hours] = 0.6

    return p


def sample_weekly_occupancy(time_seconds, prob, *, seed=None):
    """Sample an occ(t) ∈ {0,1} scenario with a single draw per constant-probability zone
    (8–18 and 18–8, weekday/weekend).
    """
    t = np.asarray(time_seconds, dtype=float)
    p = np.asarray(prob, dtype=float)
    if t.shape != p.shape:
        raise ValueError("time_seconds and prob must have the same shape.")

    n = len(t)
    occ = np.zeros_like(p, dtype=float)
    if n == 0:
        return occ

    rng = np.random.default_rng(seed)

    # Walk through intervals where p is constant
    start = 0
    for i in range(1, n):
        if not np.isclose(p[i], p[i - 1]):
            p_block = float(p[start])
            draw = 1.0 if rng.random() < p_block else 0.0
            occ[start:i] = draw
            start = i

    # Last block
    p_block = float(p[start])
    draw = 1.0 if rng.random() < p_block else 0.0
    occ[start:n] = draw

    return occ


def build_occupancy(time_seconds, *, seed=None):
    """Simple helper: return an occ(t) sample with the simplified scenario."""
    p = occupancy_probability(time_seconds)
    return sample_weekly_occupancy(time_seconds, p, seed=seed)


if __name__ == "__main__":
    # Small example over one week, 1h step
    t = np.arange(0, 5 * 24 * 3600, 3600)
    p = occupancy_probability(t)
    occ = sample_weekly_occupancy(t, p, seed=0)

    plt.step(t / (3600.0*24), p, where="post", label="P[occ=1|t]")
    plt.step(t / (3600.0*24), occ, where="post", label="Sample", alpha=0.7)
    plt.xlabel("Day")
    plt.ylabel("Occupancy / probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

