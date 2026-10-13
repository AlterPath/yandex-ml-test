import random
import sys


def gen(seed, T, S, U, C, H, g_factor, l_factor, umax):
    rnd = random.Random(seed)
    upd = []
    for _ in range(U):
        c = rnd.randrange(C)
        r = rnd.randrange(T)
        s = rnd.randint(5, 100)
        h = max(1, int(10 + r * rnd.uniform(0.3, 2.0) + rnd.randint(-10, 20)))
        upd.append((c, s, h, r))
    dev = []
    per_house = [0] * H
    for _ in range(S):
        d = rnd.randrange(H)
        per_house[d] += 1
        u = rnd.randint(1, umax)
        w = [rnd.randint(1, 10) for _ in range(C)]
        dev.append((d, u, w))
    mean_s = sum(x[1] for x in upd) / U
    demand = sum(x[1] for x in dev) * mean_s / T
    G = max(100, int(g_factor * demand))
    L = []
    for m in range(H):
        base = max(1, per_house[m]) * umax * mean_s / T
        L.append(max(100, int(l_factor * base * rnd.uniform(0.5, 1.5))))
    lines = [f"{T} {S} {U} {C} {H} {G}", " ".join(map(str, L))]
    for d, u, w in dev:
        lines.append(f"{d} {u} " + " ".join(map(str, w)))
    for c, s, h, r in upd:
        lines.append(f"{c} {s} {h} {r}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]
    seed, T, S, U, C, H = map(int, args[:6])
    g_factor, l_factor = float(args[6]), float(args[7])
    umax = int(args[8])
    sys.stdout.write(gen(seed, T, S, U, C, H, g_factor, l_factor, umax))
