import sys
import time
import random
import heapq
from bisect import insort, bisect_left

TIME_LIMIT = 2.0
ALPHAS = (0.5, 1.0, 0.0)
NOISY_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


class Input:
    def __init__(self, vals):
        p = 0
        self.T, self.S, self.U, self.C, self.H, self.G = vals[p:p + 6]
        p += 6
        self.L = vals[p:p + self.H]
        p += self.H
        self.house = []
        self.limit = []
        self.w = []
        for _ in range(self.S):
            self.house.append(vals[p])
            self.limit.append(vals[p + 1])
            self.w.append(vals[p + 2:p + 2 + self.C])
            p += 2 + self.C
        self.cls = []
        self.size = []
        self.hap = []
        self.rel = []
        for _ in range(self.U):
            c, s, h, r = vals[p:p + 4]
            p += 4
            self.cls.append(c)
            self.size.append(s)
            self.hap.append(h)
            self.rel.append(r)
        self.devices_of = {}
        for i in range(self.S):
            self.devices_of.setdefault(self.house[i], []).append(i)


def useful_updates(inp):
    by_class = [[] for _ in range(inp.C)]
    for k in range(inp.U):
        if inp.rel[k] < inp.T and inp.size[k] <= inp.G:
            by_class[inp.cls[k]].append(k)
    result = []
    for ks in by_class:
        ks.sort(key=lambda k: (inp.rel[k], -inp.hap[k], inp.size[k]))
        kept = []
        for k in ks:
            if all(inp.hap[q] < inp.hap[k] or inp.size[q] > inp.size[k] for q in kept):
                kept.append(k)
        result.extend(kept)
    return result


def candidates_per_device(inp, useful):
    cands = []
    for i in range(inp.S):
        cap = inp.L[inp.house[i]]
        cands.append([k for k in useful if inp.size[k] <= cap])
    return cands


class Plan:
    def __init__(self, inp):
        self.inp = inp
        self.srv = [inp.G] * inp.T
        self.free = {}
        for m in inp.devices_of:
            self.free[m] = [inp.L[m]] * inp.T
        self.cnt = [0] * inp.S
        self.used = set()
        self.lines = [[] for _ in range(inp.S * inp.C)]
        self.on_day = [{} for _ in range(inp.T)]
        self.version = 0

    def best_slot(self, i, k, thr):
        inp = self.inp
        j = inp.cls[k]
        s = inp.size[k]
        h = inp.hap[k]
        w = inp.w[i][j]
        free = self.free[inp.house[i]]
        srv = self.srv
        line = self.lines[i * inp.C + j]
        n = len(line)
        best_g, best_t, bound = 0, -1, 0
        prev_t, prev_h = -1, 0
        for idx in range(n + 1):
            if idx < n:
                next_t, next_h = line[idx][0], line[idx][1]
            else:
                next_t, next_h = inp.T, 0
            dh = h - prev_h
            if dh > 0:
                t = max(inp.rel[k], prev_t + 1)
                while t < next_t:
                    g = w * dh * (next_t - t)
                    if g <= best_g:
                        break
                    if g <= thr:
                        if g > bound:
                            bound = g
                        break
                    if free[t] >= s and srv[t] >= s:
                        best_g, best_t = g, t
                        break
                    t += 1
            prev_t, prev_h = next_t, next_h
        if bound > best_g:
            return bound, -1
        return best_g, best_t

    def add(self, i, k, t):
        inp = self.inp
        s = inp.size[k]
        h = inp.hap[k]
        j = inp.cls[k]
        line = self.lines[i * inp.C + j]
        insort(line, (t, h, k))
        self.free[inp.house[i]][t] -= s
        self.srv[t] -= s
        self.cnt[i] += 1
        self.used.add((i, k))
        self.on_day[t][i] = self.on_day[t].get(i, 0) + 1
        self.version += 1
        idx = bisect_left(line, (t,))
        while idx + 1 < len(line) and line[idx + 1][1] <= h:
            self.remove(i, j, idx + 1)

    def remove(self, i, j, idx):
        inp = self.inp
        t, h, k = self.lines[i * inp.C + j].pop(idx)
        s = inp.size[k]
        self.free[inp.house[i]][t] += s
        self.srv[t] += s
        self.cnt[i] -= 1
        self.used.discard((i, k))
        if self.on_day[t][i] == 1:
            del self.on_day[t][i]
        else:
            self.on_day[t][i] -= 1
        self.version += 1
        return t, k

    def clear_device(self, i):
        removed = []
        for j in range(self.inp.C):
            line = self.lines[i * self.inp.C + j]
            while line:
                removed.append(self.remove(i, j, len(line) - 1))
        return removed

    def device_value(self, i):
        inp = self.inp
        total = 0
        for j in range(inp.C):
            line = self.lines[i * inp.C + j]
            w = inp.w[i][j]
            for a in range(len(line)):
                nt = line[a + 1][0] if a + 1 < len(line) else inp.T
                total += w * line[a][1] * (nt - line[a][0])
        return total

    def score(self):
        return sum(self.device_value(i) for i in range(self.inp.S))

    def rollouts(self):
        res = []
        for idx, line in enumerate(self.lines):
            i = idx // self.inp.C
            for t, h, k in line:
                res.append((t, i, k))
        res.sort()
        return res


def greedy(plan, cands, devices, weight, bias, deadline):
    inp = plan.inp
    T = inp.T
    heap = []
    open_devs = 0
    for i in devices:
        if plan.cnt[i] >= inp.limit[i]:
            continue
        open_devs += 1
        wi = inp.w[i]
        for k in cands[i]:
            if (i, k) in plan.used:
                continue
            g = wi[inp.cls[k]] * inp.hap[k] * (T - inp.rel[k])
            heap.append((-g * bias[i] / weight[k], i, k, -1, -1))
    heapq.heapify(heap)
    steps = 0
    while heap and open_devs > 0:
        steps += 1
        if steps % 1024 == 0 and time.time() > deadline:
            return
        _, i, k, ver, t = heapq.heappop(heap)
        if plan.cnt[i] >= inp.limit[i]:
            continue
        if ver != plan.version or t < 0:
            sc = bias[i] / weight[k]
            top = -heap[0][0] if heap else 0.0
            thr = top / sc if ver != plan.version else 0
            g, t = plan.best_slot(i, k, thr)
            if g <= 0:
                continue
            pr = g * sc
            if t < 0 or pr < top:
                heapq.heappush(heap, (-pr, i, k, plan.version, t))
                continue
        plan.add(i, k, t)
        if plan.cnt[i] >= inp.limit[i]:
            open_devs -= 1


def improve(plan, cands, weights, deadline, rnd):
    inp = plan.inp
    houses = list(inp.devices_of)
    bias = [1.0] * inp.S
    while time.time() < deadline:
        if rnd.random() < 0.5:
            group = []
            for m in rnd.sample(houses, min(len(houses), rnd.randint(1, 3))):
                group.extend(inp.devices_of[m])
        else:
            on_day = list(plan.on_day[rnd.randrange(inp.T)])
            if not on_day:
                continue
            group = rnd.sample(on_day, min(len(on_day), rnd.randint(2, 8)))
        before = sum(plan.device_value(i) for i in group)
        old = []
        for i in group:
            for t, k in plan.clear_device(i):
                old.append((t, i, k))
        for i in group:
            bias[i] = rnd.uniform(0.7, 1.3)
        greedy(plan, cands, group, rnd.choice(weights), bias, float("inf"))
        after = sum(plan.device_value(i) for i in group)
        if after < before:
            for i in group:
                plan.clear_device(i)
            for t, i, k in old:
                plan.add(i, k, t)


def main():
    start = time.time()
    vals = list(map(int, sys.stdin.buffer.read().split()))
    inp = Input(vals)
    cands = candidates_per_device(inp, useful_updates(inp))
    weights = [[s ** a for s in inp.size] for a in ALPHAS]
    noisy = [[s ** a for s in inp.size] for a in NOISY_ALPHAS]
    ones = [1.0] * inp.S
    best_plan, best_score = None, -1
    for n, weight in enumerate(weights):
        share = 0.6 if n == 0 else 0.3
        plan = Plan(inp)
        greedy(plan, cands, range(inp.S), weight, ones, start + TIME_LIMIT * share)
        sc = plan.score()
        if sc > best_score:
            best_plan, best_score = plan, sc
        if time.time() - start > TIME_LIMIT * 0.3:
            break
    improve(best_plan, cands, noisy, start + TIME_LIMIT, random.Random(1))
    res = best_plan.rollouts()
    out = [str(len(res))]
    for t, i, k in res:
        out.append(f"{t} {i} {k}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
