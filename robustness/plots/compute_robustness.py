import numpy as np

def severity_weighted_tail(drops, T=20.0, L=100.0, alpha=1.0):
    depth = np.clip(-np.array(drops, dtype=float), 0, L)
    w = np.maximum(0.0, (depth - T) / (L - T)) ** alpha
    return w.mean()

def tail_fraction_and_mean_depth(drops, T=20.0, L=100.0, alpha=1.0):
    depth = np.clip(-np.array(drops, dtype=float), 0, L)
    mask = depth > T
    p = mask.mean()
    a = 0.0 if p == 0 else ((depth[mask] - T) / (L - T))**alpha .mean()
    return p, a, p*a

