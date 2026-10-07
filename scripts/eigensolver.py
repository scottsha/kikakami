import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import igl


class ShiftInvertSolver:
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
        print(f"[ShiftInvertSolver] backend = {self.backend}")

    def set_shift(self, sigma):
        S = (self.A - sigma * self.I).tocsc()

        if self.backend == "cholmod":
            from sksparse.cholmod import analyze
            # SPD when sigma < 0 -> fast supernodal LL^T.
            # Indefinite when sigma > 0 -> simplicial LDL^T (no pivoting; fine near-shift
            # for most meshes, use PARDISO if you see instability).
            mode = "supernodal" if sigma < 0 else "simplicial"
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


class MeshEigenSolver:
    def __init__(self, V, F, backend="auto"):
        L = igl.cotmatrix(V, F)                                    # negative semidefinite
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)      # diagonal
        A = -L
        m = M.diagonal()
        self.minv_sqrt = 1.0 / np.sqrt(m)
        D = sp.diags(self.minv_sqrt)
        self.A_std = (D @ A @ D).tocsc()                           # M^-1/2 A M^-1/2
        self.A_std = (self.A_std + self.A_std.T) * 0.5             # clean up round-off asymmetry
        self.solver = ShiftInvertSolver(self.A_std, backend)
        self._v0 = None

    def solve(self, sigma, k=7, tol=1e-6, ncv=None, warm_start=True):
        if sigma == 0.0:
            sigma = -1e-6                                          # avoid exact singularity
        self.solver.set_shift(sigma)
        n = self.A_std.shape[0]
        ncv = ncv or max(2 * k + 1, 16)
        vals, vecs = spla.eigsh(
            self.A_std, k=k, sigma=sigma, which="LM",
            OPinv=self.solver.as_linear_operator(),
            tol=tol, ncv=ncv,
            v0=self._v0 if warm_start else None,
        )
        if warm_start:
            self._v0 = vecs.sum(axis=1)                            # start vector for next call
        idx = np.argsort(vals)
        vals, vecs = vals[idx], vecs[:, idx]
        return vals, vecs * self.minv_sqrt[:, None]                # back to M-orthonormal modes


if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("rook.obj")
    print(f"mesh: {V.shape[0]} verts, {F.shape[0]} faces")

    ms = MeshEigenSolver(V, F)

    for sigma in [-1e-6, 5.0, 50.0, 200.0, 1000.0]:
        t = time.perf_counter()
        w, phi = ms.solve(sigma, k=7)
        dt = time.perf_counter() - t
        print(f"sigma={sigma:9.3g}  {dt*1e3:7.1f} ms  eigenvalues: {np.round(w, 3)}")