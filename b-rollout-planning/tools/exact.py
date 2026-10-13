import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from check import read_input


def solve_exact(path, time_limit=120):
    T, S, U, C, H, G, L, dev, upd = read_input(path)
    xs = []
    for i, (d, u, w) in enumerate(dev):
        for k, (c, s, h, r) in enumerate(upd):
            if s <= min(G, L[d]):
                for t in range(r, T):
                    xs.append((i, k, t))
    nx = len(xs)
    nv = 2 * nx
    idx = {key: n for n, key in enumerate(xs)}
    rows, cols, vals, lo, hi = [], [], [], [], []
    row = [0]

    def add_row(entries, lb, ub):
        for col, v in entries:
            rows.append(row[0])
            cols.append(col)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)
        row[0] += 1

    pair, day_class, dev_cnt, srv_day, house_day, y_day_class = {}, {}, {}, {}, {}, {}
    for n, (i, k, t) in enumerate(xs):
        c, s, h, r = upd[k]
        d = dev[i][0]
        pair.setdefault((i, k), []).append((n, 1))
        day_class.setdefault((i, c, t), []).append((n, 1))
        dev_cnt.setdefault(i, []).append((n, 1))
        srv_day.setdefault(t, []).append((n, s))
        house_day.setdefault((d, t), []).append((n, s))
        y_day_class.setdefault((i, c, t), []).append((nx + n, 1))
    for e in pair.values():
        add_row(e, -np.inf, 1)
    for e in day_class.values():
        add_row(e, -np.inf, 1)
    for i, e in dev_cnt.items():
        add_row(e, -np.inf, dev[i][1])
    for t, e in srv_day.items():
        add_row(e, -np.inf, G)
    for (d, t), e in house_day.items():
        add_row(e, -np.inf, L[d])
    for e in y_day_class.values():
        add_row(e, -np.inf, 1)
    for n, (i, k, t) in enumerate(xs):
        e = [(nx + n, 1)]
        for t2 in range(upd[k][3], t + 1):
            e.append((idx[(i, k, t2)], -1))
        add_row(e, -np.inf, 0)
    obj = np.zeros(nv)
    for n, (i, k, t) in enumerate(xs):
        c, s, h, r = upd[k]
        obj[nx + n] = -dev[i][2][c] * h
    A = coo_matrix((vals, (rows, cols)), shape=(row[0], nv)).tocsr()
    integrality = np.concatenate([np.ones(nx), np.zeros(nx)])
    res = milp(obj, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    if len(sys.argv) > 3 and res.x is not None:
        chosen = [xs[n] for n in range(nx) if res.x[n] > 0.5]
        with open(sys.argv[3], "w") as f:
            f.write(str(len(chosen)) + "\n")
            for i, k, t in sorted(chosen, key=lambda z: (z[2], z[0], z[1])):
                f.write(f"{t} {i} {k}" + "\n")
    best = -res.fun if res.x is not None else None
    bound = -res.mip_dual_bound if hasattr(res, "mip_dual_bound") and res.mip_dual_bound is not None else None
    return best, bound, res.status


if __name__ == "__main__":
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    print(*solve_exact(sys.argv[1], tl))
