from dataclasses import replace
import numpy as np
from model import Params, run_once

def regret(p, arch, seeds=range(10)):
    out = []
    for s in seeds:
        orc = {r["id"]: r["cost"] for r in run_once(p, "ORACLE", s)}
        aff = [r for r in run_once(p, arch, s) if r["affected"]]
        out.append(np.mean([r["cost"] - orc[r["id"]] for r in aff]))
    return np.mean(out)

base = Params(vary_window=True, review_mean_h=6.0)
for tau in (0.0, 4.0, 8.0):
    print("tau_auto", tau, "D2T regret", round(regret(replace(base, tau_auto_h=tau), "D2T")))
print("D1", round(regret(base, "D1")), "D2", round(regret(base, "D2")))