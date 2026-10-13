import sys
from check import read_input


def brute(path):
    T, S, U, C, H, G, L, dev, upd = read_input(path)
    pairs = [(i, k) for i in range(S) for k in range(U)]
    srv = [G] * T
    free = [[L[m]] * T for m in range(H)]
    cnt = [0] * S
    busy = set()
    chosen = []
    best = [0]

    def score():
        ev = {}
        for t, i, k in chosen:
            ev.setdefault((i, upd[k][0]), []).append((t, upd[k][2]))
        total = 0
        for (i, c), e in ev.items():
            e.sort()
            for a, (t, h) in enumerate(e):
                nt = e[a + 1][0] if a + 1 < len(e) else T
                total += dev[i][2][c] * h * (nt - t)
        return total

    def dfs(p):
        if p == len(pairs):
            sc = score()
            if sc > best[0]:
                best[0] = sc
            return
        dfs(p + 1)
        i, k = pairs[p]
        if cnt[i] >= dev[i][1]:
            return
        c, s, h, r = upd[k]
        m = dev[i][0]
        for t in range(r, T):
            if (t, i, c) in busy or srv[t] < s or free[m][t] < s:
                continue
            busy.add((t, i, c))
            srv[t] -= s
            free[m][t] -= s
            cnt[i] += 1
            chosen.append((t, i, k))
            dfs(p + 1)
            chosen.pop()
            cnt[i] -= 1
            free[m][t] += s
            srv[t] += s
            busy.discard((t, i, c))

    dfs(0)
    return best[0]


if __name__ == "__main__":
    print(brute(sys.argv[1]))
