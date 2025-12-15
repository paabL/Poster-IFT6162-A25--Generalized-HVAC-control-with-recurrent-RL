import numpy as np
import matplotlib.pyplot as plt


def occupancy_probability(time_seconds):
    """Profil logement P[occ=1|t] dans [0,1] avec seulement 2 blocs :

    - Semaine (lundi–vendredi) :
        * 0.3 entre 8h et 18h
        * 0.9 le reste de la journée

    - Week-end (samedi–dimanche) :
        * 0.6 entre 8h et 18h
        * 0.9 le reste de la journée
    """
    t = np.asarray(time_seconds, dtype=float)
    if t.ndim != 1:
        raise ValueError("time_seconds doit être un vecteur 1D.")

    hours = (t / 3600.0) % 24.0
    days = (t / 86400.0).astype(int)

    # 0..4 = lundi–vendredi, 5..6 = samedi–dimanche
    is_weekend = (days % 7) >= 5
    in_work_hours = (hours >= 8.0) & (hours < 18.0)

    # Par défaut : bloc "hors 8–18h" -> 0.9
    p = np.full_like(hours, 0.9, dtype=float)

    # Semaine, bloc 8–18h -> 0.3
    p[~is_weekend & in_work_hours] = 0.3

    # Week-end, bloc 8–18h -> 0.6
    p[is_weekend & in_work_hours] = 0.6

    return p


def sample_weekly_occupancy(time_seconds, prob, *, seed=None):
    """Tire un scénario occ(t) ∈ {0,1} avec un seul tirage par zone
    de probabilité constante (8–18 et 18–8, en semaine/week-end).
    """
    t = np.asarray(time_seconds, dtype=float)
    p = np.asarray(prob, dtype=float)
    if t.shape != p.shape:
        raise ValueError("time_seconds et prob doivent avoir la même forme.")

    n = len(t)
    occ = np.zeros_like(p, dtype=float)
    if n == 0:
        return occ

    rng = np.random.default_rng(seed)

    # On balaye les intervalles où p est constante
    start = 0
    for i in range(1, n):
        if not np.isclose(p[i], p[i - 1]):
            p_block = float(p[start])
            draw = 1.0 if rng.random() < p_block else 0.0
            occ[start:i] = draw
            start = i

    # Dernier bloc
    p_block = float(p[start])
    draw = 1.0 if rng.random() < p_block else 0.0
    occ[start:n] = draw

    return occ


def build_occupancy(time_seconds, *, seed=None):
    """Helper simple : retourne un tirage occ(t) avec le scénario simplifié."""
    p = occupancy_probability(time_seconds)
    return sample_weekly_occupancy(time_seconds, p, seed=seed)


if __name__ == "__main__":
    # Petit exemple sur une semaine, pas de 1 h
    t = np.arange(0, 5 * 24 * 3600, 3600)
    p = occupancy_probability(t)
    occ = sample_weekly_occupancy(t, p, seed=0)

    plt.step(t / (3600.0*24), p, where="post", label="P[occ=1|t]")
    plt.step(t / (3600.0*24), occ, where="post", label="Tirage", alpha=0.7)
    plt.xlabel("Jour")
    plt.ylabel("Occupation / probabilité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



