import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SpectralShiftFactorizer:
    """Solves (A - sigma*I) x = b, reusing the symbolic factorization across shifts."""

    def __init__(self, A, backend="auto"):
        self.A = A.tocsc()
        self.I = sp.identity(A.shape[0], format="csc")
        self.backend = backend
        self._factors = {}          # cholmod: mode -> Factor (symbolic analysis cached)
        self._solve = None

        if backend == "auto":
            try:
                import sksparse.cholmod  # noqa
                self.backend = "cholmod"
            except ImportError:
                try:
                    import pypardiso  # noqa
                    self.backend = "pardiso"
                except ImportError:
                    self.backend = "splu"
        print(f"[SpectralShiftFactorizer] backend = {self.backend}")

    def set_shift(self, spectral_shift):
        S = (self.A - spectral_shift * self.I).tocsc()

        if self.backend == "cholmod":
            from sksparse.cholmod import analyze
            # SPD when sigma < 0 -> fast supernodal LL^T.
            # Indefinite when sigma > 0 -> simplicial LDL^T (no pivoting; fine near-shift
            # for most meshes, use PARDISO if you see instability).
            mode = "supernodal" if spectral_shift < 0 else "simplicial"
            if mode not in self._factors:
                self._factors[mode] = analyze(S, mode=mode)   # symbolic: done once per mode
            f = self._factors[mode]
            f.cholesky_inplace(S)                              # numeric only
            self._solve = f.solve_A

        elif self.backend == "pardiso":
            import pypardiso
            ps = pypardiso.PyPardisoSolver(mtype=-2)           # real symmetric indefinite
            ps.factorize(sp.triu(S, format="csr"))             # upper triangle only
            self._ps, self._S = ps, sp.triu(S, format="csr")
            self._solve = lambda b: self._ps.solve(self._S, b)

        else:  # splu fallback
            lu = spla.splu(S)
            self._solve = lu.solve

    def as_linear_operator(self):
        n = self.A.shape[0]
        return spla.LinearOperator((n, n), matvec=lambda b: self._solve(np.ascontiguousarray(b)),
                                   dtype=np.float64)


class SpectralShiftEigenSolver:
    def __init__(self, matrix, mass, backend="auto"):
        m_diagonal = mass.diagonal()
        self.minv_sqrt = 1.0 / np.sqrt(m_diagonal)
        D = sp.diags(self.minv_sqrt)
        self.symmetrized_operator = (D @ matrix @ D).tocsc()
        self.symmetrized_operator = (self.symmetrized_operator + self.symmetrized_operator.T) * 0.5
        self.factorizer = SpectralShiftFactorizer(self.symmetrized_operator, backend)
        self._v0 = None

    def solve(self, sigma, k=7, tol=1e-6, ncv=None, warm_start=True):
        if sigma == 0.0:
            sigma = 1e-6
        self.factorizer.set_shift(sigma)
        ncv = ncv or max(2 * k + 1, 16)
        vals, vecs = spla.eigsh(
            self.symmetrized_operator, k=k, sigma=sigma, which="LM",
            OPinv=self.factorizer.as_linear_operator(),
            tol=tol, ncv=ncv,
            v0=self._v0 if warm_start else None,
        )
        if warm_start:
            self._v0 = vecs.sum(axis=1)                            # start vector for next call
        idx = np.argsort(vals)
        vals, vecs = vals[idx], vecs[:, idx]
        return vals, vecs * self.minv_sqrt[:, None]                # back to M-orthonormal modes