import random
import sys

SCEN = {
    1: dict(T=1, S=1, U=1, C=1, H=1, G=1000, l=(1000, 1000), u=(1, 1), w=(1, 30), s=(5, 40), h=(1, 100)),
    2: dict(T=3, S=4, U=1, C=1, H=1, G=40, l=(1000, 1000), u=(1, 1), w=(1, 30), s=(40, 40), h=(1, 100)),
    3: dict(T=14, S=8, U=30, C=2, H=2, G=100, l=(50, 50), u=(2, 4), w=(1, 30), s=(20, 50), h=(1, 100)),
    4: dict(T=30, S=50, U=100, C=5, H=10, G=500, l=(50, 100), u=(3, 8), w=(1, 50), s=(10, 100), h=(1, 100)),
    5: dict(T=90, S=1000, U=500, C=10, H=100, G=5000, l=(20, 200), u=(5, 15), w=(1, 100), s=(10, 200), h=(1, 100)),
    6: dict(T=365, S=100000, U=10000, C=30, H=10000, G=1000000, l=(10, 1000), u=(5, 30), w=(0, 100), s=(1, 1000), h=(1, 100)),
    7: dict(T=365, S=1000, U=10000, C=30, H=100, G=10000, l=(10, 1000), u=(5, 30), w=(0, 100), s=(1, 1000), h=(1, 100)),
    8: dict(T=365, S=3000, U=10000, C=30, H=300, G=30000, l=(10, 1000), u=(5, 30), w=(0, 100), s=(1, 1000), h=(1, 100)),
}


def gen(scen, seed, trend):
    p = SCEN[scen]
    rnd = random.Random(seed)
    T, S, U, C, H = p["T"], p["S"], p["U"], p["C"], p["H"]
    L = [rnd.randint(*p["l"]) for _ in range(H)]
    lines = [f"{T} {S} {U} {C} {H} {p['G']}", " ".join(map(str, L))]
    for _ in range(S):
        w = [rnd.randint(*p["w"]) for _ in range(C)]
        lines.append(f"{rnd.randrange(H)} {rnd.randint(*p['u'])} " + " ".join(map(str, w)))
    fridays = [t for t in range(T) if t % 7 == 4]
    for _ in range(U):
        if scen == 4 and rnd.random() < 0.6:
            r = rnd.choice(fridays)
        else:
            r = rnd.randrange(T)
        lo, hi = p["h"]
        if trend:
            base = lo + (hi - lo) * r / max(1, T - 1)
            h = int(min(hi, max(lo, rnd.gauss(base, (hi - lo) * 0.2))))
        else:
            h = rnd.randint(lo, hi)
        lines.append(f"{rnd.randrange(C)} {rnd.randint(*p['s'])} {h} {r}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    scen, seed, trend = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sys.stdout.write(gen(scen, seed, trend))
