"""
Middle Corridor AI-delegation model (starter version)
=====================================================

A small discrete-event simulation (DES) of freight movements on a bounded
Kazakh segment of the Trans-Caspian route. It compares decision architectures
for one recurrent operational decision: when the Aktau ferry terminal is
disrupted, should a movement CONTINUE to Aktau, REROUTE to Kuryk, or WAIT
at the junction for better information?

Decision architectures
----------------------
D0      Predefined rule, no prediction-dependent adaptation (always continue).
D1      AI recommendation + human review before execution.
D2      AI decision executed immediately, no routine human review.
D2T     Selective autonomy: D2 when the AI's own uncertainty is below a
        threshold theta, otherwise escalated to D1 (human review).
ORACLE  Benchmark: perfect knowledge of the closure end, zero latency.
        Decision regret of an architecture = its cost minus ORACLE cost under
        common random numbers (same arrivals, disruptions and error draws).

Mechanisms represented (see README for the mapping to the research questions)
----------------------------------------------------------------------------
* Prediction quality: multiplicative log-normal error on the remaining closure
  time, with case-level difficulty (sigma_i), bias, and distribution shift.
* Information arrival: prediction error shrinks as time passes after the
  decision point, so waiting (for review or at the yard) buys information.
* Human review: a review delay L_i and a reviewer who holds an own, partly
  independent estimate (error human_sigma) but is anchored on the AI
  (anchoring; stronger under distribution shift = automation bias). The
  reviewer can both CATCH wrong recommendations and FALSELY OVERRIDE correct
  ones; both rates emerge from the model.
* Decision latency: the review must finish before the movement reaches the
  junction (commitment window tau). If it does not, the movement either holds
  at the junction until the review ends or proceeds on the default route.
* Reversibility: a reroute committed before the junction can be reversed at
  the junction if new information favours Aktau, at a cost that grows with the
  time elapsed since commitment.

All parameter values are ILLUSTRATIVE PLACEHOLDERS to be calibrated with
Middle Corridor data during the PhD. The model is pure Python + numpy
(no SimPy) so it runs anywhere.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

CONTINUE, REROUTE, WAIT = "continue", "reroute", "wait"
ARCHITECTURES = ("D0", "D1", "D2", "D2T", "ORACLE")


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    # --- horizon ---------------------------------------------------------
    horizon_h: float = 24 * 365 * 2      # generate arrivals for 2 years
    warmup_h: float = 24 * 30            # ignore first 30 days in statistics

    # --- demand and network (hours) --------------------------------------
    interarrival_mean_h: float = 8.0     # one block train every ~8 h
    origin_to_P_mean_h: float = 96.0     # rail: inland node -> decision point P
    origin_to_P_cv: float = 0.15
    tau_h: float = 12.0                  # P -> junction J (commitment window)
    J_to_aktau_h: float = 6.0
    J_to_kuryk_h: float = 10.0
    crossing_h: float = 20.0             # ferry to Alat (Baku)
    due_slack_h: float = 160.0           # due time = creation + slack

    # --- ferry terminals ---------------------------------------------------
    aktau_headway_h: float = 5.0
    aktau_cap: int = 1
    kuryk_headway_h: float = 8.0
    kuryk_cap: int = 1

    # --- disruptions at Aktau ---------------------------------------------
    closure_interarrival_mean_h: float = 240.0
    closure_median_h: float = 30.0
    closure_log_sd: float = 0.9
    joint_weather_prob: float = 0.3      # share of closures also hitting Kuryk
    distribution_shift: bool = False     # severe regime (see shift_* below)
    shift_duration_mult: float = 2.0
    shift_bias: float = -0.6             # AI underestimates under shift

    # --- AI prediction ------------------------------------------------------
    sigma_base: float = 0.5              # typical log-error of the prediction
    sigma_heterogeneity: float = 0.4     # case-to-case spread of difficulty
    bias: float = 0.0                    # log-bias in normal regime
    info_gain_per_h: float = 0.03        # error shrinks exp(-k * elapsed)

    # --- human review (D1 and escalated D2T cases) ------------------------
    review_mean_h: float = 6.0           # exponential review delay
    human_sigma: float = 0.6             # log-error of the reviewer's own estimate
    anchoring: float = 0.3               # weight the reviewer puts on the AI
    anchoring_under_shift: float = 0.6   # automation bias when the world shifts
    override_margin: float = 0.05        # override only if >5 % cost difference
    late_review: str = "hold"            # "hold" at J, or "default" (continue)

    # --- selective autonomy (D2T) -------------------------------------------
    theta: float = 0.45                  # autonomous if sigma_i <= theta

    # --- wait option ----------------------------------------------------------
    wait_h: float = 8.0
    wait_band: float = 0.10              # wait if predicted costs within 10 %
    wait_min_sigma: float = 0.6          # ... and the case is uncertain

    # --- costs (illustrative USD per block train) ---------------------------
    holding_per_h: float = 100.0
    reroute_cost: float = 3000.0
    late_penalty_per_h: float = 200.0
    reversal_cost_fixed: float = 1500.0
    reversal_cost_per_h: float = 100.0   # grows with time since commitment
    allow_reversal: bool = True


# --------------------------------------------------------------------------
# Minimal event engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.now = 0.0
        self._q: list = []
        self._seq = 0

    def at(self, t: float, fn, *args):
        self._seq += 1
        heapq.heappush(self._q, (t, self._seq, fn, args))

    def run(self):
        while self._q:
            t, _, fn, args = heapq.heappop(self._q)
            self.now = t
            fn(*args)


# --------------------------------------------------------------------------
# Terminals
# --------------------------------------------------------------------------
class Terminal:
    def __init__(self, sim: "Simulation", name, headway, cap, closures):
        self.sim, self.name, self.headway, self.cap = sim, name, headway, cap
        self.starts = [s for s, _ in closures]
        self.closures = closures
        self.queue: deque = deque()
        self.inbound = 0                 # movements committed and travelling here

    def closure_end(self, t):
        """End of the closure active at time t, or None if open."""
        i = bisect_right(self.starts, t) - 1
        if i >= 0 and self.closures[i][0] <= t < self.closures[i][1]:
            return self.closures[i][1]
        return None

    def start(self):
        self.sim.eng.at(self.headway, self.depart)

    def depart(self):
        eng = self.sim.eng
        if self.closure_end(eng.now) is None:
            for _ in range(min(self.cap, len(self.queue))):
                self.queue.popleft().on_ferry(eng.now)
        if self.sim.active > 0 or eng.now < self.sim.p.horizon_h:
            eng.at(eng.now + self.headway, self.depart)

    def expected_wait(self, arrival_t, closure_end_t):
        """Approximate FIFO port wait for a unit arriving at arrival_t.

        Only units ahead of it matter: those already queued, those already
        travelling to this terminal, and (for Aktau) the regular flow that
        reaches the terminal before it. Service resumes when the closure ends.
        """
        p, now = self.sim.p, self.sim.eng.now
        ahead = len(self.queue) + self.inbound
        if self.name == "aktau":
            ahead += max(0.0, arrival_t - now - p.J_to_aktau_h) / p.interarrival_mean_h
        rate = self.cap / self.headway
        reopen = now if closure_end_t is None else max(now, closure_end_t)
        # units served between reopening and our arrival
        served = max(0.0, arrival_t - reopen) * rate
        left = max(0.0, ahead - served)
        start_service = max(arrival_t, reopen)
        return (start_service - arrival_t) + left / rate + self.headway / 2


# --------------------------------------------------------------------------
# Movements
# --------------------------------------------------------------------------
@dataclass
class Movement:
    sim: "Simulation"
    mid: int
    created: float
    due: float
    rnd: dict
    affected: bool = False
    action: str = CONTINUE
    reviewed: bool = False
    missed_window: bool = False
    reversed: bool = False
    rerouted: bool = False
    commit_t: float | None = None
    extra_cost: float = 0.0
    done_t: float | None = None
    terminal: str = "aktau"
    ai_action: str | None = None
    oracle_action: str | None = None
    path: str = "none"
    tP: float = 0.0
    caught: bool = False
    false_override: bool = False

    # ---- life cycle ---------------------------------------------------------
    def start(self):
        self.sim.eng.at(self.created + self.rnd["T1"], self.at_P)

    def at_P(self):
        sim, t = self.sim, self.sim.eng.now
        self.tP = t
        end_A = sim.aktau.closure_end(t)
        if end_A is None or sim.arch == "D0":
            self.affected = end_A is not None
            sim.eng.at(t + sim.p.tau_h, self.at_J)
            return
        self.affected = True
        arch = sim.arch
        if arch == "D2T":
            arch = "D2" if self.rnd["sigma_i"] <= sim.p.theta else "D1"
        self.path = arch
        if arch == "ORACLE":
            self.execute(sim.decide(self, t, "oracle", allow_wait=False), t)
            sim.eng.at(t + sim.p.tau_h, self.at_J)
        elif arch == "D2":
            self.execute(sim.decide(self, t), t)
            sim.eng.at(t + sim.p.tau_h, self.at_J)
        else:  # D1: human review
            self.reviewed = True
            L = self.rnd["L"]
            if L <= sim.p.tau_h:
                sim.eng.at(t + L, self.review_done)
            sim.eng.at(t + sim.p.tau_h, self.at_J)

    def review_done(self):
        t = self.sim.eng.now
        self.execute(self.sim.review(self, t), t)

    def execute(self, action, t):
        self.action = action
        if action == REROUTE:
            self.commit_reroute(t)

    def commit_reroute(self, t):
        if not self.rerouted:
            self.rerouted, self.commit_t = True, t
            self.extra_cost += self.sim.p.reroute_cost
            self.sim.kuryk.inbound += 1

    def at_J(self):
        sim, p, t = self.sim, self.sim.p, self.sim.eng.now
        # review not finished before the junction
        if self.reviewed and self.rnd["L"] > p.tau_h:
            self.missed_window = True
            if p.late_review == "hold":
                sim.eng.at(self.tP + self.rnd["L"], self.review_done_at_J)
                return
            self.action = CONTINUE
        self.leave_J()

    def review_done_at_J(self):
        t = self.sim.eng.now
        self.execute(self.sim.review(self, t), t)
        self.leave_J()

    def leave_J(self):
        sim, p, t = self.sim, self.sim.p, self.sim.eng.now
        if self.action == WAIT:
            self.action = "waited"
            sim.eng.at(t + p.wait_h, self.after_wait)
            return
        # reversibility: undo an early reroute if new information favours Aktau
        if self.rerouted and p.allow_reversal and self.commit_t < t - 1e-9:
            c_cont, c_rr = sim.costs(self, t, "oracle" if self.path == "ORACLE" else "ai")
            c_rev = p.reversal_cost_fixed + p.reversal_cost_per_h * (t - self.commit_t)
            if c_rr - p.reroute_cost - c_cont > c_rev:
                self.reversed, self.rerouted = True, False
                self.extra_cost += c_rev
                sim.kuryk.inbound -= 1
        self.go_to_port(t)

    def after_wait(self):
        t = self.sim.eng.now
        who = "oracle" if self.path == "ORACLE" else "ai"
        a = self.sim.decide(self, t, who, allow_wait=False)
        if a == REROUTE:
            self.commit_reroute(t)
        self.go_to_port(t)

    def go_to_port(self, t):
        p = self.sim.p
        if self.rerouted:
            self.terminal = "kuryk"
            self.sim.eng.at(t + p.J_to_kuryk_h, self.arrive_port, self.sim.kuryk)
        else:
            self.sim.aktau.inbound += 1
            self.sim.eng.at(t + p.J_to_aktau_h, self.arrive_port, self.sim.aktau)

    def arrive_port(self, term: Terminal):
        term.inbound -= 1
        term.queue.append(self)

    def on_ferry(self, t):
        self.done_t = t + self.sim.p.crossing_h
        self.sim.active -= 1
        self.sim.finished.append(self)

    # ---- outcome -------------------------------------------------------------
    def total_cost(self):
        p = self.sim.p
        transit = self.done_t - self.created
        late = max(0.0, self.done_t - self.due)
        return p.holding_per_h * transit + p.late_penalty_per_h * late + self.extra_cost


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
class Simulation:
    def __init__(self, p: Params, arch: str, seed: int):
        assert arch in ARCHITECTURES
        self.p, self.arch, self.seed = p, arch, seed
        self.eng = Engine()
        self.finished: list[Movement] = []
        self.active = 0
        rng = np.random.default_rng(seed)
        self._draw_world(rng)

    # ---- common random numbers ---------------------------------------------------
    def _draw_world(self, rng):
        p = self.p
        # arrivals
        n_max = int(p.horizon_h / p.interarrival_mean_h * 1.5) + 50
        gaps = rng.exponential(p.interarrival_mean_h, n_max)
        arrivals = np.cumsum(gaps)
        self.arrivals = arrivals[arrivals < p.horizon_h]
        n = len(self.arrivals)
        # per-movement random numbers
        s = math.sqrt(math.log(1 + p.origin_to_P_cv ** 2))
        mu = math.log(p.origin_to_P_mean_h) - s * s / 2
        self.rnd = {
            "T1": rng.lognormal(mu, s, n),
            "eps": rng.standard_normal(n),
            "sig": rng.standard_normal(n),
            "uL": rng.random(n),
            "eh": rng.standard_normal(n),
        }
        # disruptions (Aktau; a share also closes Kuryk)
        dur_mult = p.shift_duration_mult if p.distribution_shift else 1.0
        closures_a, closures_k = [], []
        t = rng.exponential(p.closure_interarrival_mean_h)
        while t < p.horizon_h + 2000:
            d = rng.lognormal(math.log(p.closure_median_h * dur_mult), p.closure_log_sd)
            joint = rng.random() < p.joint_weather_prob
            frac = rng.uniform(0.5, 1.0)
            if closures_a and t < closures_a[-1][1]:
                t = closures_a[-1][1] + 1.0
            closures_a.append((t, t + d))
            if joint:
                closures_k.append((t, t + d * frac))
            t += d + rng.exponential(p.closure_interarrival_mean_h)
        self.aktau = Terminal(self, "aktau", p.aktau_headway_h, p.aktau_cap, closures_a)
        self.kuryk = Terminal(self, "kuryk", p.kuryk_headway_h, p.kuryk_cap, closures_k)

    def movement_rnd(self, i):
        p, r = self.p, self.rnd
        sigma_i = p.sigma_base * math.exp(p.sigma_heterogeneity * r["sig"][i]
                                          - p.sigma_heterogeneity ** 2 / 2)
        return {
            "T1": r["T1"][i], "eps": r["eps"][i], "sigma_i": sigma_i,
            "L": -p.review_mean_h * math.log(1 - r["uL"][i]),
            "eh": r["eh"][i],
        }

    # ---- prediction and decisions ----------------------------------------------
    def predicted_end(self, m: Movement, term: Terminal, t, who="ai"):
        """Closure end at `term` as seen by `who` in {"oracle", "ai", "human"}."""
        end = term.closure_end(t)
        if end is None or who == "oracle":
            return end
        p = self.p
        decay = math.exp(-p.info_gain_per_h * max(0.0, t - m.tP))
        sign = 1.0 if term.name == "aktau" else -1.0
        bias = p.bias + (p.shift_bias if p.distribution_shift else 0.0)
        ai_log_err = bias + m.rnd["sigma_i"] * decay * sign * m.rnd["eps"]
        if who == "ai":
            log_err = ai_log_err
        else:  # human reviewer: own information, partly anchored on the AI
            w = p.anchoring_under_shift if p.distribution_shift else p.anchoring
            own = p.human_sigma * decay * sign * m.rnd["eh"]
            log_err = w * ai_log_err + (1 - w) * own
        return t + (end - t) * math.exp(log_err)

    def costs(self, m: Movement, t, who="ai"):
        """Predicted remaining cost of (CONTINUE, REROUTE) from time t."""
        p = self.p
        to_J = max(0.0, m.tP + p.tau_h - t)
        res = []
        for term, leg, extra in ((self.aktau, p.J_to_aktau_h, 0.0),
                                 (self.kuryk, p.J_to_kuryk_h, p.reroute_cost)):
            arr = t + to_J + leg
            wait = term.expected_wait(arr, self.predicted_end(m, term, t, who))
            eta = arr + wait + p.crossing_h
            res.append(p.holding_per_h * (eta - t)
                       + p.late_penalty_per_h * max(0.0, eta - m.due) + extra)
        return res[0], res[1]

    def decide(self, m, t, who="ai", allow_wait=True):
        c_cont, c_rr = self.costs(m, t, who)
        best = CONTINUE if c_cont <= c_rr else REROUTE
        if (allow_wait and who == "ai"
                and abs(c_cont - c_rr) < self.p.wait_band * min(c_cont, c_rr)
                and m.rnd["sigma_i"] > self.p.wait_min_sigma):
            return WAIT
        return best

    def review(self, m, t):
        """Human review of the AI recommendation, completed at time t.

        The reviewer forms an own estimate (independent error human_sigma,
        partly anchored on the AI with weight `anchoring`) and overrides the
        recommendation only if the own estimate favours the other action by
        more than `override_margin`. Catch and false-override rates are
        therefore outcomes of the model, not inputs.
        """
        p = self.p
        ai = self.decide(m, t, "ai")
        truth = self.decide(m, t, "oracle", allow_wait=False)
        h_cont, h_rr = self.costs(m, t, "human")
        human = CONTINUE if h_cont <= h_rr else REROUTE
        confident = abs(h_cont - h_rr) > p.override_margin * min(h_cont, h_rr)
        final = human if (confident and human != ai) else ai
        m.ai_action, m.oracle_action = ai, truth
        m.caught = ai != truth and final == truth
        m.false_override = ai == truth and final != truth
        return final

    # ---- run ---------------------------------------------------------------------
    def run(self):
        self.aktau.start()
        self.kuryk.start()
        for i, t0 in enumerate(self.arrivals):
            m = Movement(self, i, t0, t0 + self.p.due_slack_h, self.movement_rnd(i))
            m.start()
        self.active = len(self.arrivals)
        self.eng.run()
        return self

    def results(self):
        rows = []
        for m in self.finished:
            if m.created < self.p.warmup_h:
                continue
            rows.append({
                "id": m.mid, "affected": m.affected, "cost": m.total_cost(),
                "transit_h": m.done_t - m.created, "on_time": m.done_t <= m.due,
                "terminal": m.terminal, "rerouted": m.rerouted, "reversed": m.reversed,
                "reviewed": m.reviewed, "missed_window": m.missed_window,
                "action": m.action, "caught": m.caught,
                "false_override": m.false_override,
                "ai_wrong": (m.ai_action is not None and m.ai_action != m.oracle_action),
            })
        return rows


def run_once(p: Params, arch: str, seed: int):
    return Simulation(p, arch, seed).run().results()
