import sys
from check import read_input


def class_curve(items, T, w, budget):
    items = sorted(items, key=lambda x: (x[0], -x[1]))
    stair = []
    for r, h in items:
        if not stair or h > stair[-1][1]:
            stair.append((r, h))
    items = stair
    n = len(items)
    NEG = float("-inf")
    dp = [[NEG] * (budget + 1) for _ in range(n)]
    best = [0] * (budget + 1)
    for a in range(n):
        ra, ha = items[a]
        dp[a][1] = w * ha * (T - ra)
        for b in range(a):
            rb, hb = items[b]
            if hb >= ha:
                continue
            gain = w * (ha - hb) * (T - ra)
            row_b = dp[b]
            row_a = dp[a]
            for q in range(2, budget + 1):
                if row_b[q - 1] != NEG and row_b[q - 1] + gain > row_a[q]:
                    row_a[q] = row_b[q - 1] + gain
        for q in range(1, budget + 1):
            if dp[a][q] > best[q]:
                best[q] = dp[a][q]
    for q in range(1, budget + 1):
        best[q] = max(best[q], best[q - 1])
    return best


def upper_bound(path):
    T, S, U, C, H, G, L, dev, upd = read_input(path)
    total = 0
    for i, (d, u, w) in enumerate(dev):
        cap = min(G, L[d])
        by_class = [[] for _ in range(C)]
        for c, s, h, r in upd:
            if s <= cap:
                by_class[c].append((r, h))
        knap = [0] * (u + 1)
        for c in range(C):
            if not by_class[c]:
                continue
            curve = class_curve(by_class[c], T, w[c], min(u, len(by_class[c])))
            new = knap[:]
            for q in range(u + 1):
                for x in range(1, len(curve)):
                    if q + x > u:
                        break
                    v = knap[q] + curve[x]
                    if v > new[q + x]:
                        new[q + x] = v
            knap = new
        total += max(knap)
    return total


if __name__ == "__main__":
    print(upper_bound(sys.argv[1]))
