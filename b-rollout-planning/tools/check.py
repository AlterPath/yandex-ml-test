import sys


def read_input(path):
    with open(path) as f:
        vals = list(map(int, f.read().split()))
    p = 0
    T, S, U, C, H, G = vals[p:p + 6]
    p += 6
    L = vals[p:p + H]
    p += H
    dev = []
    for _ in range(S):
        d, u = vals[p], vals[p + 1]
        w = vals[p + 2:p + 2 + C]
        p += 2 + C
        dev.append((d, u, w))
    upd = []
    for _ in range(U):
        upd.append(tuple(vals[p:p + 4]))
        p += 4
    return T, S, U, C, H, G, L, dev, upd


def check(inp_path, out_path):
    T, S, U, C, H, G, L, dev, upd = read_input(inp_path)
    with open(out_path) as f:
        vals = list(map(int, f.read().split()))
    A = vals[0]
    assert len(vals) == 1 + 3 * A, "bad output length"
    rows = [tuple(vals[1 + 3 * a:4 + 3 * a]) for a in range(A)]
    seen_pair = set()
    seen_day_class = set()
    cnt = [0] * S
    srv = {}
    house = {}
    events = {}
    for t, i, k in rows:
        assert 0 <= t < T and 0 <= i < S and 0 <= k < U, ("range", t, i, k)
        c, s, h, r = upd[k]
        assert r <= t, ("release", t, i, k)
        assert (i, k) not in seen_pair, ("repeat", t, i, k)
        seen_pair.add((i, k))
        assert (t, i, c) not in seen_day_class, ("same class same day", t, i, k)
        seen_day_class.add((t, i, c))
        cnt[i] += 1
        assert cnt[i] <= dev[i][1], ("device limit", i)
        srv[t] = srv.get(t, 0) + s
        assert srv[t] <= G, ("server limit", t)
        m = dev[i][0]
        house[(t, m)] = house.get((t, m), 0) + s
        assert house[(t, m)] <= L[m], ("house limit", t, m)
        events.setdefault((i, c), []).append((t, h))
    total = 0
    for (i, c), ev in events.items():
        ev.sort()
        w = dev[i][2][c]
        for a, (t, h) in enumerate(ev):
            nt = ev[a + 1][0] if a + 1 < len(ev) else T
            total += w * h * (nt - t)
    return total


if __name__ == "__main__":
    print(check(sys.argv[1], sys.argv[2]))
