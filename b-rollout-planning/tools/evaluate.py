import os
import re
import subprocess
import sys
import tempfile
import time
from check import check


def run(solution, tl, test):
    src = open(solution).read()
    if tl is not None:
        src = re.sub(r"^TIME_LIMIT = .*$", f"TIME_LIMIT = {tl}", src, flags=re.M)
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sol.py")
        op = os.path.join(d, "out.txt")
        open(sp, "w").write(src)
        t0 = time.time()
        with open(test) as fin, open(op, "w") as fout:
            subprocess.run([sys.executable, sp], stdin=fin, stdout=fout, check=True)
        wall = time.time() - t0
        return check(test, op), wall


if __name__ == "__main__":
    solution = sys.argv[1]
    tl = None if sys.argv[2] == "-" else sys.argv[2]
    for test in sys.argv[3:]:
        sc, wall = run(solution, tl, test)
        print(f"{os.path.basename(test)} score={sc} wall={wall:.2f}s", flush=True)
