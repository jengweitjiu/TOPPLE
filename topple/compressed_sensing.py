"""
Compressed Interaction Sensing
===============================

Applies compressed sensing theory to recover sparse Möbius interaction
coefficients without exhaustive subset enumeration.

Key insight: In biological regulatory networks, most higher-order
interactions are negligible — only a few sparse high-order terms carry
signal. This sparsity assumption enables L1-regularized recovery from
a random sample of subset evaluations, analogous to how compressed
sensing recovers sparse signals from undersampled measurements.

Theory
------
Let I ∈ R^N be the vector of all Möbius interaction coefficients
(N = 2^p - 1 for p features). The stability loss Δ(T) for any subset
T is a linear function of the interaction coefficients:

    Δ(T) = Σ_{S: S⊇T} I(S)    (by Möbius inversion identity)

If we evaluate Δ for m randomly chosen subsets (m << N), we obtain:

    y = Φ · I

where y ∈ R^m are the observed stability losses, and Φ ∈ R^{m×N} is
the sensing matrix derived from the subset containment structure.

Under the sparsity assumption ||I||_0 = s << N, we recover I via:

    minimize ||I||_1  subject to  ||y - Φ·I||_2 ≤ ε

References
----------
- Candès, E.J. & Tao, T. (2006). Near-optimal signal recovery from
  random projections. IEEE Trans. Inf. Theory.
- Stobbe, P. & Krause, A. (2012). Learning Fourier-sparse set functions.
  AISTATS.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


class CompressedInteractionSensing:
    """
    Recover sparse interaction terms from random subset sampling.

    Parameters
    ----------
    n_features : int
        Number of features (regulons).
    max_order : int, default=3
        Maximum interaction order to recover.
    n_measurements : int, optional
        Number of random subsets to evaluate. Default: O(s · log(N/s))
        where s is the expected sparsity and N is the total number of
        interaction terms.
    sparsity_estimate : int, optional
        Expected number of non-zero interaction terms.
        If None, estimated as p * max_order (conservative).
    random_state : int, default=42
        Random seed for reproducibility.
    solver : str, default="scipy"
        Optimization solver. Options: "scipy" (linprog), "cvxpy" (if installed).

    Attributes
    ----------
    sensing_matrix_ : np.ndarray
        Binary sensing matrix Φ.
    sampled_subsets_ : list of frozenset
        Random subsets used as measurements.
    recovered_interactions_ : dict
        Recovered I(S) interaction terms.
    recovery_error_ : float
        L2 residual of the recovery.
    """

    def __init__(
        self,
        n_features: int,
        max_order: int = 3,
        n_measurements: Optional[int] = None,
        sparsity_estimate: Optional[int] = None,
        random_state: int = 42,
        solver: str = "scipy",
    ):
        self.n_features = n_features
        self.max_order = max_order
        self.random_state = random_state
        self.solver = solver

        # Enumerate all interaction terms up to max_order
        self.all_subsets_: List[FrozenSet[int]] = []
        for order in range(1, max_order + 1):
            for combo in combinations(range(n_features), order):
                self.all_subsets_.append(frozenset(combo))
        self.N_ = len(self.all_subsets_)
        self.subset_to_idx_ = {s: i for i, s in enumerate(self.all_subsets_)}

        # Determine number of measurements
        if sparsity_estimate is None:
            sparsity_estimate = n_features * max_order
        self.sparsity_estimate = sparsity_estimate

        if n_measurements is None:
            # O(s · log(N/s)) with safety factor
            self.n_measurements = min(
                int(3 * sparsity_estimate * np.log(max(self.N_ / sparsity_estimate, 2))),
                self.N_,  # don't exceed total subsets
            )
        else:
            self.n_measurements = n_measurements

    def design_measurements(self) -> List[FrozenSet[int]]:
        """
        Design the random measurement subsets.

        Uses a stratified random sampling strategy that ensures coverage
        across different subset sizes, with oversampling of small subsets
        (which contribute to more interaction terms).

        Returns
        -------
        list of frozenset
            Subsets to evaluate (compute Δ for).
        """
        rng = np.random.RandomState(self.random_state)

        # Stratified sampling: allocate measurements proportional to
        # the number of interaction terms at each order
        subsets_by_order = {}
        for order in range(1, self.max_order + 1):
            subsets_by_order[order] = [
                s for s in self.all_subsets_ if len(s) == order
            ]

        # Ensure all single features are measured (marginal contributions)
        sampled = [frozenset([i]) for i in range(self.n_features)]
        remaining = self.n_measurements - len(sampled)

        if remaining > 0:
            # Allocate remaining across orders 2..max_order
            pool = []
            for order in range(2, self.max_order + 1):
                pool.extend(subsets_by_order[order])

            if len(pool) <= remaining:
                sampled.extend(pool)
            else:
                indices = rng.choice(len(pool), size=remaining, replace=False)
                sampled.extend([pool[i] for i in indices])

        # Also add some random "mixed" subsets (not aligned with interaction terms)
        # These provide additional constraints for the recovery
        n_mixed = min(self.n_features, remaining // 4) if remaining > 0 else 0
        for _ in range(n_mixed):
            size = rng.randint(2, min(self.max_order + 1, self.n_features + 1))
            subset = frozenset(rng.choice(self.n_features, size=size, replace=False))
            if subset not in sampled:
                sampled.append(subset)

        self.sampled_subsets_ = sampled
        return sampled

    def build_sensing_matrix(self) -> np.ndarray:
        """
        Build the sensing matrix Φ from the measurement design.

        Φ[m, n] = 1 if interaction term n (subset S_n) is a superset of
        measurement subset T_m, because:

            Δ(T) = Σ_{S ⊇ T} I(S)

        Returns
        -------
        np.ndarray, shape (n_measurements, N)
        """
        m = len(self.sampled_subsets_)
        Phi = np.zeros((m, self.N_))

        for i, T in enumerate(self.sampled_subsets_):
            for j, S in enumerate(self.all_subsets_):
                if T.issubset(S):
                    Phi[i, j] = 1.0

        self.sensing_matrix_ = Phi
        return Phi

    def recover(
        self,
        delta_values: Dict[FrozenSet[int], float],
        regularization: float = 0.01,
    ) -> Dict[FrozenSet[int], float]:
        """
        Recover sparse interaction coefficients via L1 minimization.

        Parameters
        ----------
        delta_values : dict
            Mapping from sampled subsets to their Δ values.
        regularization : float, default=0.01
            Noise tolerance ε for the constraint ||y - Φ·I||_2 ≤ ε.

        Returns
        -------
        dict
            Recovered interaction terms {frozenset -> I(S)}.
        """
        if not hasattr(self, "sampled_subsets_"):
            self.design_measurements()
        if not hasattr(self, "sensing_matrix_"):
            self.build_sensing_matrix()

        # Build observation vector
        y = np.array([delta_values.get(s, 0.0) for s in self.sampled_subsets_])

        Phi = self.sensing_matrix_

        if self.solver == "cvxpy":
            coeffs = self._recover_cvxpy(Phi, y, regularization)
        else:
            coeffs = self._recover_scipy(Phi, y, regularization)

        # Map back to subsets
        self.recovered_interactions_ = {}
        for i, val in enumerate(coeffs):
            if abs(val) > 1e-6:  # Filter near-zero terms
                self.recovered_interactions_[self.all_subsets_[i]] = val

        # Compute recovery error
        y_pred = Phi @ coeffs
        self.recovery_error_ = np.linalg.norm(y - y_pred)

        return self.recovered_interactions_

    def _recover_scipy(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """
        L1 recovery via scipy linprog (LASSO-like reformulation).

        Reformulate: minimize ||I||_1 s.t. ||y - Φ·I||_2 ≤ ε
        As a linear program with variable splitting I = I+ - I-, I+,I- >= 0.
        """
        m, N = Phi.shape

        # LASSO formulation: minimize (1/2m)||y - Φ·I||_2^2 + λ||I||_1
        # Using iteratively reweighted least squares with L1 penalty
        from sklearn.linear_model import Lasso

        lasso = Lasso(alpha=eps, fit_intercept=False, max_iter=10000)
        lasso.fit(Phi, y)
        return lasso.coef_

    def _recover_cvxpy(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """L1 recovery via CVXPY (basis pursuit denoising)."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "CVXPY not installed. Install with: pip install cvxpy"
            )

        N = Phi.shape[1]
        I_var = cp.Variable(N)
        objective = cp.Minimize(cp.norm(I_var, 1))
        constraints = [cp.norm(Phi @ I_var - y, 2) <= eps]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            import warnings

            warnings.warn(f"CVXPY solve status: {problem.status}")

        return I_var.value if I_var.value is not None else np.zeros(N)

    def summary(self) -> str:
        """Return recovery summary."""
        lines = [
            f"Compressed Interaction Sensing Summary",
            f"  Features: {self.n_features}",
            f"  Max order: {self.max_order}",
            f"  Total interaction terms (N): {self.N_}",
            f"  Measurements (m): {len(self.sampled_subsets_)}",
            f"  Compression ratio: {len(self.sampled_subsets_)/self.N_:.2%}",
        ]
        if hasattr(self, "recovered_interactions_"):
            n_nonzero = len(self.recovered_interactions_)
            lines.extend([
                f"  Recovered non-zero terms: {n_nonzero}",
                f"  Recovery error (L2): {self.recovery_error_:.6f}",
            ])
        return "\n".join(lines)
