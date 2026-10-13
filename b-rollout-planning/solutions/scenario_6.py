import sys
import time
import heapq
import random
from bisect import bisect_right
from collections import defaultdict

TIME_LIMIT = 20.0
MAX_FAILS = 200
SAMPLE_SHARE = 0.01
LAMBDAS = (0, 250, 500, 1000, 2000, 4000, 8000, 16000)


class Data:
    def __init__(self, raw):
        data = raw.split() 
        T, S, U, C, H, G = map(int, data[:6])
        self.T, self.S, self.U, self.C, self.H, self.G = T, S, U, C, H, G
        self.L = list(map(int, data[6:6 + H]))
        p = 6 + H
        row = C + 2
        self.row = row
        self.dev = list(map(int, data[p:p + S * row]))
        p += S * row
        upd = list(map(int, data[p:p + 4 * U]))
        self.cls, self.size, self.hap, self.rel = upd[0::4], upd[1::4], upd[2::4], upd[3::4]
        self.house, self.limit = self.dev[0::row], self.dev[1::row]


def pareto(d):
    by_class = [[] for _ in range(d.C)]
    for k in range(d.U):
        if d.rel[k] < d.T and d.size[k] <= d.G:
            by_class[d.cls[k]].append(k)
    fronts = []
    for ks in by_class:
        ks.sort(key=lambda k: (d.rel[k], -d.hap[k], d.size[k]))
        kept = []
        for k in ks:
            if all(d.hap[q] < d.hap[k] or d.size[q] > d.size[k] for q in kept):
                kept.append(k)
        kept.sort(key=lambda k: d.size[k])
        fronts.append(kept)
    return fronts


def second_step(k, k1, T, hap, rel):
    if rel[k] > rel[k1] and hap[k] > hap[k1]:
        return (hap[k] - hap[k1]) * (T - rel[k])
    if rel[k] < rel[k1] and hap[k] < hap[k1]:
        return hap[k] * (rel[k1] - rel[k])
    return 0


def build_tables(fronts, lam, d):
    T, size, hap, rel = d.T, d.size, d.hap, d.rel
    items_tab, ladder_tab = [], []
    for j, ks in enumerate(fronts):
        P = len(ks)
        items_c = [[()] * 101 for _ in range(P + 1)]
        ladder_c = [[None] * 101 for _ in range(P + 1)]
        for w in range(1, 101):
            k1, v1, k2, v2 = -1, 0, -1, 0
            for c in range(1, P + 1):
                k = ks[c - 1]
                v = w * hap[k] * (T - rel[k]) - lam * size[k]
                if v > v1:
                    k1, v1 = k, v
                    k2, v2 = -1, 0
                    for q in ks[:c]:
                        a = second_step(q, k1, T, hap, rel)
                        if a:
                            vq = w * a - lam * size[q]
                            if vq > v2:
                                k2, v2 = q, vq
                elif k1 >= 0:
                    a = second_step(k, k1, T, hap, rel)
                    if a:
                        vq = w * a - lam * size[k]
                        if vq > v2:
                            k2, v2 = k, vq
                if k1 < 0:
                    continue
                if k2 >= 0:
                    items_c[c][w] = (v1 * 64 + j * 2, v2 * 64 + j * 2 + 1)
                else:
                    items_c[c][w] = (v1 * 64 + j * 2,)
                ladder_c[c][w] = (k1, k2)
        items_tab.append(items_c)
        ladder_tab.append(ladder_c)
    return items_tab, ladder_tab


def plan(devices, tables, house_counts, d):
    T, C, row, dev, house, limit = d.T, d.C, d.row, d.dev, d.house, d.limit
    hap, rel = d.hap, d.rel
    items_tab, ladder_tab = tables
    groups = defaultdict(list)
    rows = {}
    for i in devices:
        m = house[i]
        hr = rows.get(m)
        if hr is None:
            counts = house_counts[m]
            hr = ([items_tab[j][counts[j]] for j in range(C)], [ladder_tab[j][counts[j]] for j in range(C)])
            rows[m] = hr
        trow, lrow = hr
        base = i * row + 2
        ws = dev[base:base + C]
        items = [x for tj, w in zip(trow, ws) for x in tj[w]]
        items.sort(reverse=True)
        left = limit[i]
        taken = {}
        for x in items:
            if left <= 0:
                break
            j = (x & 63) >> 1
            if x & 1:
                if j not in taken:
                    continue
                taken[j] = 2
            else:
                taken[j] = 1
            left -= 1
        for j, level in taken.items():
            w = ws[j]
            k1, k2 = lrow[j][w]
            if level == 1:
                groups[(k1 * 101 + w) * 101].append(i * 512 + T)
                continue
            if rel[k2] < rel[k1]:
                k1, k2 = k2, k1
            groups[(k1 * 101 + w) * 101].append(i * 512 + rel[k2])
            groups[(k2 * 101 + w) * 101 + hap[k1]].append(i * 512 + T)
    return groups


def schedule(groups, G, d, deadline, out):
    T, C = d.T, d.C
    size, hap, rel, house, L, cls = d.size, d.hap, d.rel, d.house, d.L, d.cls
    push, pop = heapq.heappush, heapq.heappop
    release = [[] for _ in range(T)]
    glist = []
    for key, devs in groups.items():
        prev_h = key % 101
        w = key // 101 % 101
        k = key // 10201
        release[rel[k]].append((-w * (hap[k] - prev_h) / size[k], len(glist)))
        glist.append((k, w, devs))
    cur_h = bytearray(d.S * C)
    heap = []
    backlog = {}
    value = 0
    for t in range(T):
        if time.time() > deadline:
            break
        for e in release[t]:
            push(heap, e)
        srv = G
        hl = L[:]
        for m in list(backlog):
            bl = backlog[m]
            while bl:
                negp, k, w, i, nxt = bl[0]
                if nxt <= t:
                    pop(bl)
                    continue
                s = size[k]
                if s > hl[m] or s > srv:
                    break
                pop(bl)
                idx = i * C + cls[k]
                h = hap[k]
                srv -= s
                hl[m] -= s
                value += w * (h - cur_h[idx]) * (T - t)
                cur_h[idx] = h
                if out is not None:
                    out.append((t, i, k))
            if not bl:
                del backlog[m]
        waiting = []
        fails = 0
        while heap:
            e = heap[0]
            k, w, devs = glist[e[1]]
            s = size[k]
            if s <= srv:
                h = hap[k]
                j = cls[k]
                while devs and s <= srv:
                    p = devs.pop()
                    i = p >> 9
                    nxt = p & 511
                    if nxt <= t:
                        continue
                    m = house[i]
                    if s > hl[m]:
                        bl = backlog.get(m)
                        if bl is None:
                            backlog[m] = [(e[0], k, w, i, nxt)]
                        else:
                            push(bl, (e[0], k, w, i, nxt))
                        continue
                    idx = i * C + j
                    srv -= s
                    hl[m] -= s
                    value += w * (h - cur_h[idx]) * (T - t)
                    cur_h[idx] = h
                    if out is not None:
                        out.append((t, i, k))
            pop(heap)
            if devs:
                waiting.append(e)
                fails += 1
                if fails > MAX_FAILS:
                    break
        for e in waiting:
            push(heap, e)
    return value


def tune_lambda(d, fronts, house_counts, rnd, deadline):
    houses = sorted(set(d.house))
    n = min(len(houses), max(30, int(len(houses) * SAMPLE_SHARE)))
    chosen = set(rnd.sample(houses, n))
    devices = [i for i in range(d.S) if d.house[i] in chosen]
    G = max(1, round(d.G * len(devices) / d.S))
    results = {}

    def evaluate(lam):
        if lam not in results:
            groups = plan(devices, build_tables(fronts, lam, d), house_counts, d)
            results[lam] = schedule(groups, G, d, float("inf"), None)
        return results[lam]

    for lam in LAMBDAS:
        evaluate(lam)
        if time.time() > deadline:
            break
    best = max(results, key=results.get)
    step = 1.4
    while time.time() < deadline and step > 1.05:
        for cand in (int(best * step), int(best / step)):
            evaluate(cand)
        best = max(results, key=results.get)
        step = step ** 0.5
    return best


def main():
    start = time.time()
    d = Data(sys.stdin.buffer.read())
    fronts = pareto(d)
    front_sizes = [[d.size[k] for k in ks] for ks in fronts]
    house_counts = [[bisect_right(fs, d.L[m]) for fs in front_sizes] for m in range(d.H)]
    lam = tune_lambda(d, fronts, house_counts, random.Random(7), start + TIME_LIMIT * 0.3)
    groups = plan(range(d.S), build_tables(fronts, lam, d), house_counts, d)
    out = []
    schedule(groups, d.G, d, start + TIME_LIMIT * 0.9, out)
    tstr = [f"{t} " for t in range(d.T)]
    istr = [f"{i} " for i in range(d.S)]
    kstr = [str(k) for k in range(d.U)]
    lines = [tstr[t] + istr[i] + kstr[k] for t, i, k in out]
    sys.stdout.write(str(len(lines)) + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
