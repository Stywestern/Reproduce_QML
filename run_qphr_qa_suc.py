"""
qubo.py
=======
A minimal, self-contained QUBO (Quadratic Unconstrained Binary Optimization)
representation used as the "quantum-native" objective format throughout this
package. Mirrors Eq. (4)-(6) of the paper:

    min  sum_i sum_j  B_ij * x_i * x_j  +  sum_i c_i * x_i

Kept as plain Python dicts keyed by *variable name* (not raw index) so the
pieces built in master.py, phr_alm.py and d_admm.py can be freely combined,
inspected, and swapped without index bookkeeping.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class QUBO:
    """min sum_i linear[i]*x_i + sum_{i<j} quad[(i,j)]*x_i*x_j + offset."""
    linear: dict[str, float] = field(default_factory=dict)
    quad: dict[tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0

    def add_linear(self, name: str, coeff: float) -> None:
        if coeff == 0.0:
            return
        self.linear[name] = self.linear.get(name, 0.0) + coeff

    def add_quadratic(self, name_i: str, name_j: str, coeff: float) -> None:
        if coeff == 0.0:
            return
        if name_i == name_j:
            # x_i^2 == x_i for a binary variable
            self.add_linear(name_i, coeff)
            return
        key = (name_i, name_j) if name_i < name_j else (name_j, name_i)
        self.quad[key] = self.quad.get(key, 0.0) + coeff

    def add_constant(self, c: float) -> None:
        self.offset += c

    def merge(self, other: "QUBO", weight: float = 1.0) -> "QUBO":
        """Return self + weight*other as a NEW QUBO (does not mutate self)."""
        out = QUBO(dict(self.linear), dict(self.quad), self.offset)
        for k, v in other.linear.items():
            out.add_linear(k, weight * v)
        for (i, j), v in other.quad.items():
            out.add_quadratic(i, j, weight * v)
        out.offset += weight * other.offset
        return out

    def variables(self) -> list[str]:
        names = set(self.linear) | {n for pair in self.quad for n in pair}
        return sorted(names)

    def energy(self, assignment: dict[str, int]) -> float:
        e = self.offset
        for name, c in self.linear.items():
            e += c * assignment.get(name, 0)
        for (i, j), c in self.quad.items():
            e += c * assignment.get(i, 0) * assignment.get(j, 0)
        return e


class BinaryEncoder:
    """
    Encodes one continuous variable as a weighted sum of binary variables
    (Sec. III-A1):   Upsilon = chi * sum_{j=0}^{J-1} 2^j * u_j
    """

    def __init__(self, name: str, num_bits: int, precision: float):
        self.name = name
        self.num_bits = num_bits
        self.precision = precision  # "chi" in the paper
        self.bit_names = [f"{name}__b{j}" for j in range(num_bits)]
        self.weights = {b: precision * (2 ** j) for j, b in enumerate(self.bit_names)}

    def linear_terms(self) -> dict[str, float]:
        """Coeffs so that sum_b linear_terms[b]*x_b == the encoded value."""
        return dict(self.weights)

    def decode(self, assignment: dict[str, int]) -> float:
        return sum(w * assignment.get(b, 0) for b, w in self.weights.items())

    def max_value(self) -> float:
        return self.precision * (2 ** self.num_bits - 1)

"""
backend.py
==========
Pluggable QUBO "solver" backends. This is the exact seam where a real
D-Wave QPU call (or a gate-based sampler) would plug in later -- swap
`SimulatedAnnealingBackend` for a `DWaveBackend` without touching
phr_alm.py / d_admm.py / benders.py at all.

No real QPU is reachable from this sandbox, so `SimulatedAnnealingBackend`
plays the role of the "QA sim": classical simulated annealing is the
standard classical proxy for a quantum annealer's behaviour on a QUBO
(it's literally what dimod's SimulatedAnnealingSampler does, minus the
dependency -- this is a small self-contained reimplementation so the
package has zero external deps beyond numpy/scipy).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import itertools
import numpy as np

from qubo import QUBO


class Backend(Protocol):
    def sample_qubo(self, qubo: QUBO, num_reads: int = 200) -> tuple[dict[str, int], float]:
        ...


@dataclass
class SimulatedAnnealingBackend:
    """Bit-flip Metropolis simulated annealer over a QUBO.

    `num_reads` independent anneals are run; the best (lowest-energy) one
    is returned -- mirroring how a D-Wave QPU call returns many samples
    per anneal cycle and the caller picks the best.
    """
    num_sweeps: int = 400
    beta_start: float = 0.1
    beta_end: float = 12.0
    seed: int | None = None

    def sample_qubo(self, qubo: QUBO, num_reads: int = 200) -> tuple[dict[str, int], float]:
        rng = np.random.default_rng(self.seed)
        names = qubo.variables()
        if not names:
            return {}, qubo.offset

        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        J = np.zeros((n, n))
        h = np.zeros(n)
        for name, c in qubo.linear.items():
            h[idx[name]] += c
        for (i, j), c in qubo.quad.items():
            J[idx[i], idx[j]] += c
            J[idx[j], idx[i]] += c

        betas = np.linspace(self.beta_start, self.beta_end, self.num_sweeps)

        best_x, best_e = None, np.inf
        for _ in range(num_reads):
            x = rng.integers(0, 2, size=n)
            for beta in betas:
                order = rng.permutation(n)
                for i in order:
                    local_field = h[i] + J[i] @ x
                    dE = (1 - 2 * x[i]) * local_field  # energy change of flipping bit i
                    if dE <= 0 or rng.random() < np.exp(-beta * dE):
                        x[i] = 1 - x[i]
            e = float(h @ x + (x @ J @ x) / 2 + qubo.offset)
            if e < best_e:
                best_e, best_x = e, x.copy()

        assignment = {name: int(best_x[idx[name]]) for name in names}
        return assignment, best_e


@dataclass
class ExactBackend:
    """Brute-force enumeration -- debug/verification use only (<=~20 vars)."""

    def sample_qubo(self, qubo: QUBO, num_reads: int = 1) -> tuple[dict[str, int], float]:
        names = qubo.variables()
        if len(names) > 22:
            raise ValueError("ExactBackend is for debugging only (<=22 binary vars)")
        best_x, best_e = None, np.inf
        for bits in itertools.product([0, 1], repeat=len(names)):
            assignment = dict(zip(names, bits))
            e = qubo.energy(assignment)
            if e < best_e:
                best_e, best_x = e, assignment
        return best_x, best_e


"""
master.py
=========
Builds the *master problem* (Eq. 2 in the paper) as a set of plain linear
inequalities over binary variables, ready to be fed into PHR-ALM
(phr_alm.py) or D-ADMM (d_admm.py) to become a QUBO -- with NO slack
variables, exactly as the paper's core idea (Sec. III-B).

We deliberately do NOT reuse SUCModelBuilder's LP/MIP machinery (problem.py):
the master problem only ever contains BINARY variables (u_g,t and the
binary encoding of Upsilon), so a dedicated, much smaller representation
is used here instead.

NOTE on fidelity: the min-up/min-down constraints below follow the
standard UC formulation (every generator, every hour, window truncated at
the horizon boundary). The paper restricts these to a slightly narrower
index range (t = 2 .. T-T^U_g+1); if you need bit-exact reproduction of
their results, tighten `build_min_up_down_constraints` accordingly -- the
PHR-ALM/D-ADMM machinery downstream doesn't care either way, it just
consumes a list of LinearIneq.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from problem import SUCProblemData
from qubo import BinaryEncoder


def u_name(g: int, t: int) -> str:
    return f"u_gen{g}_hr{t}"


@dataclass
class LinearIneq:
    """A single constraint g(x) <= 0, stored as  coeffs . x  <=  rhs."""
    coeffs: dict[str, float]
    rhs: float
    label: str = ""

    def g(self, assignment: dict[str, int]) -> float:
        """Evaluate g(x) = coeffs.x - rhs (violated when g(x) > 0)."""
        return sum(c * assignment.get(name, 0) for name, c in self.coeffs.items()) - self.rhs


def build_min_up_down_constraints(pd: SUCProblemData) -> list[LinearIneq]:
    """Standard min-up/min-down formulation, rewritten as g(x) <= 0."""
    cons: list[LinearIneq] = []
    N, T = pd.num_gens, pd.num_hours
    for g in range(N):
        TU, TD = int(pd.min_up[g]), int(pd.min_down[g])
        for t in range(T):
            prev_name = None if t == 0 else u_name(g, t - 1)
            prev_const = float(pd.initial_status[g]) if t == 0 else 0.0

            # --- Min-up: sum_{tau=t}^{t+TU-1} u_tau - TU*u_t + TU*u_{t-1} <= 0
            coeffs: dict[str, float] = {}
            for tau in range(t, min(t + TU, T)):
                coeffs[u_name(g, tau)] = coeffs.get(u_name(g, tau), 0.0) + 1.0
            coeffs[u_name(g, t)] = coeffs.get(u_name(g, t), 0.0) - TU
            rhs = 0.0
            if prev_name is not None:
                coeffs[prev_name] = coeffs.get(prev_name, 0.0) + TU
            else:
                rhs -= TU * prev_const
            cons.append(LinearIneq(coeffs, rhs, f"min_up_g{g}_t{t}"))

            # --- Min-down: sum_{tau=t}^{t+TD-1} (1-u_tau) - TD*u_{t-1} + TD*u_t <= 0
            window = list(range(t, min(t + TD, T)))
            coeffs = {}
            for tau in window:
                coeffs[u_name(g, tau)] = coeffs.get(u_name(g, tau), 0.0) - 1.0
            coeffs[u_name(g, t)] = coeffs.get(u_name(g, t), 0.0) + TD
            rhs = -len(window)
            if prev_name is not None:
                coeffs[prev_name] = coeffs.get(prev_name, 0.0) - TD
            else:
                rhs += TD * prev_const
            cons.append(LinearIneq(coeffs, rhs, f"min_down_g{g}_t{t}"))
    return cons


@dataclass
class MasterProblem:
    """Everything needed to evaluate/penalize the master problem (Eq. 2),
    without slack variables."""
    pd: SUCProblemData
    upsilon_encoder: BinaryEncoder
    alpha_lower: float = 0.0

    min_up_down: list[LinearIneq] = field(default_factory=list)
    benders_cuts: list[LinearIneq] = field(default_factory=list)

    def __post_init__(self):
        if not self.min_up_down:
            self.min_up_down = build_min_up_down_constraints(self.pd)

    def linear_objective(self) -> dict[str, float]:
        """Eq. (2a): sum C_cons_g * u_g,t + Upsilon."""
        obj: dict[str, float] = {}
        for g in range(self.pd.num_gens):
            for t in range(self.pd.num_hours):
                n = u_name(g, t)
                obj[n] = obj.get(n, 0.0) + self.pd.c_fixed[g]
        for name, w in self.upsilon_encoder.linear_terms().items():
            obj[name] = obj.get(name, 0.0) + w
        return obj

    def all_constraints(self) -> list[LinearIneq]:
        """Eq. (2b), (2d), (2e) and the accumulated Benders cuts (2c)."""
        cons = list(self.min_up_down) + list(self.benders_cuts)
        cons.append(LinearIneq(  # Upsilon >= alpha_lower  ->  alpha_lower - Upsilon <= 0
            coeffs={n: -w for n, w in self.upsilon_encoder.linear_terms().items()},
            rhs=-self.alpha_lower,
            label="alpha_lower",
        ))
        return cons

    def add_benders_cut(self, theta: dict[str, float], u_prev: dict[str, int],
                         const_term: float, label: str = "") -> None:
        """
        Eq. (2c):  Upsilon >= const_term + sum theta_{g,t}*(u_{g,t} - u_prev_{g,t})
        rewritten as   g(x) = [const_term - sum theta*u_prev] + sum theta*u - Upsilon <= 0
        i.e.           sum theta*u - Upsilon <= sum theta*u_prev - const_term
        """
        rhs_const = sum(theta.get(n, 0.0) * u_prev.get(n, 0) for n in theta) - const_term
        coeffs = dict(theta)
        for n, w in self.upsilon_encoder.linear_terms().items():
            coeffs[n] = coeffs.get(n, 0.0) - w
        self.benders_cuts.append(LinearIneq(coeffs=coeffs, rhs=rhs_const,
                                             label=label or f"cut_{len(self.benders_cuts)}"))



"""
phr_alm.py
==========
Powell-Hestenes-Rockafellar Augmented Lagrangian Multiplier method
(Sec. III-B / Algorithm 1 of the paper).

Turns a set of linear inequality constraints g_i(x) <= 0 into a QUBO
penalty -- WITHOUT slack variables -- using the previous iterate to decide,
per constraint, whether it is currently satisfied (Case 1, ignored, folds
into a constant) or violated (Case 2, penalized), exactly as Eq. (13)-(15).
This is the paper's core trick and the reason qubit count stops scaling
with the number of Benders cuts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from qubo import QUBO
from master import LinearIneq
from backend import Backend


def constraint_to_qubo(con: LinearIneq) -> QUBO:
    """g(x) = coeffs.x - rhs, expressed as an (affine) QUBO -- used as the
    building block for the squared penalty term below."""
    q = QUBO()
    for name, c in con.coeffs.items():
        q.add_linear(name, c)
    q.add_constant(-con.rhs)
    return q


def squared_qubo(q: QUBO) -> QUBO:
    """Return the QUBO representing (q(x))^2. Since x_i^2 = x_i for a
    binary variable, this stays a valid degree-2 QUBO -- exactly why the
    quadratic PHR penalty (Eq. 14/15) is QPU-native with no extra tricks."""
    names = q.variables()
    out = QUBO()
    out.add_constant(q.offset ** 2)
    for n, c in q.linear.items():
        out.add_linear(n, 2 * q.offset * c)
    for a in range(len(names)):
        i = names[a]
        ci = q.linear.get(i, 0.0)
        out.add_linear(i, ci * ci)  # x_i^2 == x_i
        for b in range(a + 1, len(names)):
            j = names[b]
            cj = q.linear.get(j, 0.0)
            out.add_quadratic(i, j, 2 * ci * cj)
    return out


@dataclass
class PHRState:
    lam: dict[str, float] = field(default_factory=dict)
    sigma: float = 1.0
    eta: float = 1.05     # penalty growth factor
    rho: float = 0.5      # residual-reduction factor ("varsigma" in Sec. III-E)
    tol: float = 1e-3
    prev_residual: float = np.inf


class PHRAugmentedLagrangian:
    """Iteratively builds and solves the PHR-augmented QUBO for a base
    objective + a list of LinearIneq constraints (Algorithm 1)."""

    def __init__(self, objective_linear: dict[str, float],
                 constraints: list[LinearIneq], backend: Backend,
                 sigma0: float = 1.0, eta: float = 1.05, rho: float = 0.5,
                 tol: float = 1e-3):
        self.objective_linear = objective_linear
        self.constraints = constraints
        self.backend = backend
        self.state = PHRState(lam={c.label: 0.0 for c in constraints},
                               sigma=sigma0, eta=eta, rho=rho, tol=tol)
        self.history: list[dict] = []

    def _base_qubo(self) -> QUBO:
        q = QUBO()
        for n, c in self.objective_linear.items():
            q.add_linear(n, c)
        return q

    def _penalty_qubo(self, x_prev: dict[str, int]) -> QUBO:
        """Eq. (14)/(15): only constraints violated by x_prev get penalized
        (Case 2); satisfied ones (Case 1) contribute only a constant and
        are skipped entirely -- line 5-9 of Algorithm 1."""
        s = self.state
        total = QUBO()
        for con in self.constraints:
            g_val = con.g(x_prev)
            lam_i = s.lam[con.label]
            if lam_i + s.sigma * g_val > 0:  # Case 2: violated
                shifted = constraint_to_qubo(con).merge(QUBO(offset=lam_i / s.sigma))
                total = total.merge(squared_qubo(shifted), weight=s.sigma / 2.0)
        return total

    def residual(self, x: dict[str, int]) -> float:
        """Eq. (16c)."""
        s = self.state
        terms = [max(-s.lam[con.label] / s.sigma, con.g(x)) ** 2 for con in self.constraints]
        return float(np.sqrt(sum(terms))) if terms else 0.0

    def step(self, x_prev: dict[str, int] | None, num_reads: int = 200) -> dict[str, int]:
        """One PHR-ALM iteration (Algorithm 1, lines 3-13)."""
        s = self.state
        x_prev = x_prev or {}
        qubo = self._base_qubo().merge(self._penalty_qubo(x_prev))
        x_new, energy = self.backend.sample_qubo(qubo, num_reads=num_reads)

        R = self.residual(x_new)
        for con in self.constraints:  # Eq. (16a)
            s.lam[con.label] = max(s.lam[con.label] + s.sigma * con.g(x_new), 0.0)
        s.sigma = s.sigma if R < s.rho * s.prev_residual else s.eta * s.sigma  # Eq. (16b)
        s.prev_residual = R

        self.history.append({"residual": R, "sigma": s.sigma, "energy": energy})
        return x_new

    def solve(self, max_iter: int = 50, num_reads: int = 200) -> dict[str, int]:
        x: dict[str, int] = {}
        for _ in range(max_iter):
            x = self.step(x, num_reads=num_reads)
            if self.history[-1]["residual"] <= self.state.tol:
                break
        return x