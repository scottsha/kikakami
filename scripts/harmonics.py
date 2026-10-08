import numpy as np
import igl
import pyvista as pv
import vtk
import scipy as sp
from dataclasses import dataclass
import os
import scipy.sparse.linalg as spla
from scipy.stats import qmc

from eigensolver import SpectralShiftFactorizer, SpectralShiftEigenSolver


def compute_moment_frame(mesh: pv.PolyData) -> np.ndarray:
    """
    Compute principal component analysis based frame.
    Resolve the signs of directions using signed second moments.
    """
    centered = mesh.points - mesh.points.mean(axis=0)
    covar = (centered.transpose() @ centered) / mesh.n_points
    eigenvalues, c_eigenvectors = np.linalg.eig(covar)
    eigenvectors = np.real(c_eigenvectors)
    idx = np.argsort(np.abs(eigenvalues), descending=True)
    sorted_eigenvalues = np.real(eigenvalues[idx])
    sorted_eigenvectors = eigenvectors[:, idx]
    for dim in range(3):
        coord = centered @ sorted_eigenvectors[:, dim].transpose()
        signed_second_moment = (coord * np.abs(coord)).mean()
        if abs(signed_second_moment) > 0.1 * sorted_eigenvalues[dim]:
            # The signed second moment is large enough to be usefully determinative
            if signed_second_moment < 0:
                sorted_eigenvectors[:, dim] *= -1
        else:
            # The shape is too symmetric, choose so that the largest entry is positive
            arr = sorted_eigenvectors[:, dim]
            max_magnitude_idx = np.argmax(np.abs(arr))
            if arr[max_magnitude_idx] < 0:
                sorted_eigenvectors[:, dim] *= -1
    if np.linalg.det(sorted_eigenvectors) < 0:
        sorted_eigenvectors[:, 2] *= -1
    return sorted_eigenvectors


def build_sign_test_function(mesh: pv.PolyData) -> np.ndarray:
    frame = compute_moment_frame(mesh)
    centered = mesh.points - mesh.points.mean(axis=0)
    cannon_coords = centered @ frame
    test_function = 1 / (1 + np.exp(1 - cannon_coords @ np.array([np.pi, np.sqrt(2), 1])))
    return test_function

def canonicalize_mode_signs(mesh: pv.PolyData, modes: np.ndarray) -> np.ndarray:
    test_function = build_sign_test_function(mesh)
    triangle_vertices = mesh.faces.reshape((-1, 4))[:, 1:]
    mass = igl.massmatrix(mesh.points, triangle_vertices)
    signs = np.sign(test_function.transpose() @ mass @ modes)
    signed_modes = modes * signs
    return signed_modes


@dataclass
class Eigenstructure:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class LinearOperator:
    support_indices: list[int]
    matrix: sp.sparse.spmatrix
    support_mass: sp.sparse.spmatrix


def build_dirichlet_operator(mesh: pv.PolyData) -> LinearOperator:
    triangle_vertices = mesh.faces.reshape((-1, 4))[:, 1:]
    loops = igl.boundary_loop_all(triangle_vertices)
    boundary_vertices = set([vv for loop in loops for vv in loop])
    interior_verties = [foo for foo in range(0, mesh.n_points) if foo not in boundary_vertices]
    full_cot = igl.cotmatrix(mesh.points, triangle_vertices)
    full_mass = igl.massmatrix(mesh.points, triangle_vertices, igl.MASSMATRIX_TYPE_VORONOI)
    linear_operator = LinearOperator(
        support_indices=interior_verties,
        matrix=-full_cot.tocsr()[interior_verties, :].tocsc()[:, interior_verties],
        support_mass=full_mass.tocsr()[interior_verties, :].tocsc()[:, interior_verties])
    return linear_operator


def compute_boundary_dirichlet_modes(mesh: pv.PolyData, num_modes: int, sigma: float = 0.0) -> Eigenstructure:
    operator = build_dirichlet_operator(mesh)
    print(f"Computing {num_modes} modes on {mesh.n_points} vertices.")
    eigenvalues, waves_supported = spla.eigs(
        operator.matrix,
        k=num_modes,
        M=operator.support_mass,
        which="SM",
        sigma=sigma)
    waves_prenormalized = np.zeros(shape=(mesh.n_points, num_modes))
    waves_prenormalized[operator.support_indices, :] = waves_supported.real
    linf_norms = np.linalg.norm(waves_prenormalized, ord=np.inf, axis=0, keepdims=True)
    waves_presigned = waves_prenormalized / linf_norms
    waves = canonicalize_mode_signs(mesh, waves_presigned)
    eigenstructure = Eigenstructure(eigenvalues=np.real(eigenvalues), eigenvectors=waves)
    return eigenstructure


def get_cached_boundary_dirichlet_modes(mesh: pv.PolyData, num_modes: int) -> Eigenstructure:
    cache_file = "linear_operator_spectrum.npz"
    cached_evals = None
    cached_evecs = None

    if os.path.exists(cache_file):
        try:
            with np.load(cache_file) as data:
                if data["number_of_cells"] == mesh.n_cells and data["number_of_vertices"] == mesh.n_points:
                    cached_evals = data["spectrum"]
                    cached_evecs = data["eigenvectors"]
        except Exception as e:
            print(f"Cache load failed: {e}")

    if cached_evals is not None and cached_evecs is not None:
        num_cached = len(cached_evals)
        if num_cached >= num_modes:
            print("Loaded eigenstructure from cache.")
            return Eigenstructure(
                eigenvalues=cached_evals[:num_modes],
                eigenvectors=cached_evecs[:, :num_modes]
            )
        else:
            k_needed = num_modes - num_cached
            print(f"Cache partially satisfied. Have {num_cached}, need {num_modes}. Computing {k_needed} more modes...")

            top_cached = float(np.max(cached_evals))
            new_struct = compute_boundary_dirichlet_modes(mesh, num_modes=k_needed, sigma=top_cached)

            all_evals = np.concatenate([cached_evals, new_struct.eigenvalues])
            all_evecs = np.hstack([cached_evecs, new_struct.eigenvectors])

            sort_idx = np.argsort(all_evals)
            all_evals = all_evals[sort_idx]
            all_evecs = all_evecs[:, sort_idx]

            unique_mask = np.ones(len(all_evals), dtype=bool)
            for i in range(1, len(all_evals)):
                for j in range(i - 1, -1, -1):
                    if np.abs(all_evals[i] - all_evals[j]) > 1e-5:
                        break
                    if unique_mask[j]:
                        mode_i = all_evecs[:, i]
                        mode_j = all_evecs[:, j]
                        dot_prod = np.abs(np.dot(mode_i, mode_j))
                        norm_sq_j = np.dot(mode_j, mode_j)
                        if np.abs(dot_prod - norm_sq_j) < 1e-4 * norm_sq_j:
                            unique_mask[i] = False
                            break

            final_evals = all_evals[unique_mask]
            final_evecs = all_evecs[:, unique_mask]

            np.savez(cache_file,
                     number_of_cells=mesh.n_cells,
                     number_of_vertices=mesh.n_points,
                     spectrum=final_evals,
                     eigenvectors=final_evecs)

            return Eigenstructure(
                eigenvalues=final_evals[:num_modes],
                eigenvectors=final_evecs[:, :num_modes]
            )

    print("Computing eigenstructure from scratch...")
    struct = compute_boundary_dirichlet_modes(mesh, num_modes=num_modes)
    np.savez(cache_file,
             number_of_cells=mesh.n_cells,
             number_of_vertices=mesh.n_points,
             spectrum=struct.eigenvalues,
             eigenvectors=struct.eigenvectors)
    return struct


def interpolated_mode(eig: Eigenstructure, relative_eigenmode: float) -> np.ndarray:
    est_eigenmode = (1 - relative_eigenmode) * eig.eigenvalues[0] + relative_eigenmode * eig.eigenvalues[-1]
    decay_rate = 0.5 * np.square(eig.eigenvalues.size / (eig.eigenvalues[-1] - eig.eigenvalues[0]))
    weights = np.exp(-decay_rate * np.square(eig.eigenvalues - est_eigenmode))
    weights /= np.sum(weights)
    print(weights)
    mode = eig.eigenvectors @ weights
    return mode


def weyl_estimate(area: float, boundary_length: float, nth_mode: int) -> float:
    # Weyl says the lambda_N \approx \frac{4\pi N}{A} + \frac{2\sqrt{\pi} L}{A^{3/2}} \sqrt{N}
    return 4 * np.pi / area * nth_mode + 2 * np.sqrt(np.pi) * np.pow(area, -1.5) * boundary_length * np.sqrt(nth_mode)


def weyl_gap_estimate(area: float, boundary_length: float, nth_mode: float) -> float:
    # Derivative of the Weyl estimate says dlambda_N \approx \frac{4\pi}{A} + \frac{2\sqrt{\pi} L}{A^{3/2}} \sqrt{N}
    return 4 * np.pi / area + np.sqrt(np.pi) * np.pow(area, -1.5) * boundary_length / np.sqrt(nth_mode)


def extrude_triangles_to_wedges(mesh: pv.PolyData, offset_field_name: str, min_thickness: float = None, smoothing_iterations=0) -> pv.PolyData:
    """Extrude each triangle of an all-triangle mesh into a vtkWedge.

    Each vertex at x0 is paired with an extruded vertex at
    x0 + height * point_normal; the wedge connects the original
    triangle to its extruded copy.
    """
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
    normals = mesh.point_data["Normals"]

    points = mesh.points
    n_points = points.shape[0]
    offset = mesh.point_data[offset_field_name]
    extruded_points = points + offset[:, None] * normals
    all_points = np.vstack([points, extruded_points])

    tris = mesh.regular_faces
    n_cells = tris.shape[0]
    wedge_conn = np.hstack([tris, tris + n_points])

    cells = np.hstack([np.full((n_cells, 1), 6, dtype=np.int64), wedge_conn]).ravel()
    cell_types = np.full(n_cells, vtk.VTK_WEDGE, dtype=np.uint8)

    ugrid = pv.UnstructuredGrid(cells, cell_types, all_points)
    ugrid.point_data[offset_field_name] = np.vstack([offset[:, None], offset[:, None]])
    if min_thickness is not None:
        thresholded = ugrid.clip_scalar(
            value=min_thickness,
            scalars=offset_field_name,
            invert=False, )
        surface = thresholded.extract_surface(algorithm='geometry')
    else:
        surface = ugrid.extract_surface(algorithm='geometry')
    return surface


def rescale_range(xx: np.ndarray, range_low: float, range_hi: float) -> np.ndarray:
    low_xx = xx.min()
    high_xx = xx.max()
    slope = (range_hi - range_low) / (high_xx - low_xx)
    rescaled = slope * (xx - low_xx) + range_low
    return rescaled


def rescale_range_per_column(xx: np.ndarray, range_low: float, range_hi: float) -> np.ndarray:
    low_xx = xx.min(axis=0, keepdims=True)
    high_xx = xx.max(axis=0, keepdims=True)
    diff = high_xx - low_xx
    # Avoid division by zero if a column is constant
    diff = np.where(diff == 0, 1.0, diff)
    slope = (range_hi - range_low) / diff
    rescaled = slope * (xx - low_xx) + range_low
    return rescaled


class WaveGenerator:
    def __init__(self, mesh: pv.Polydata):
        self.num_vertices = mesh.n_points
        self.area = mesh.area
        boundary_edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        boundary_edges = boundary_edges.compute_cell_sizes()
        self.boundary_measure = boundary_edges["Length"].sum()
        self.operator = build_dirichlet_operator(mesh)
        self.solver = SpectralShiftEigenSolver(self.operator.matrix, self.operator.support_mass)
        self.test_function = build_sign_test_function(mesh)[self.operator.support_indices]

    def supported_to_full_field(self, supported_field: np.ndarray) -> np.ndarray:
        full_shape = list(supported_field.shape)
        full_shape[0] = self.num_vertices
        field = np.zeros(shape=full_shape)
        field[self.operator.support_indices, :] = supported_field
        return field

    def set_mode_signs(self, modes: np.ndarray) -> np.ndarray:
        signs = np.sign(self.test_function.transpose() @ self.operator.support_mass @ modes)
        signed_modes = modes * signs
        return signed_modes

    def generate_ground_mode(self) -> np.ndarray:
        ground_eig, ground_mode = self.solver.solve(1e-6, k=1)
        if ground_mode.sum() < 0:
            ground_mode *= -1
        ground_state_rescaled = rescale_range(self.supported_to_full_field(ground_mode),
                                              0.0,
                                              1.0)
        return ground_state_rescaled

    def generate_pure_mode(self, mode_rank: float) -> np.ndarray:
        spectral_shift = weyl_estimate(self.area, self.boundary_measure, mode_rank)
        eigenvalues_complex, mode = self.solver.solve(spectral_shift, k=1)
        mode = mode / np.abs(mode).max()
        mode = self.set_mode_signs(mode)
        return self.supported_to_full_field(mode)

    def generate_wave(self, mode_rank: float, weight_sharpness:float = 1.0) -> np.ndarray:
        spectral_target = weyl_estimate(self.area, self.boundary_measure, mode_rank)
        eigenvalues, modes_raw = self.solver.solve(spectral_target, k=7)
        mode_scaled = modes_raw / np.abs(modes_raw).max(axis=0, keepdims=True)
        modes = self.set_mode_signs(mode_scaled)
        weyl_gap = weyl_gap_estimate(self.area, self.boundary_measure, mode_rank)
        off_target = (eigenvalues - spectral_target)
        weights_prenormalized = np.exp(-2 * weight_sharpness * np.square(off_target / weyl_gap))
        weights = weights_prenormalized / np.sum(weights_prenormalized)
        wave_prenormalized = modes @ weights[:, np.newaxis]
        full_wave = self.supported_to_full_field(wave_prenormalized)
        return full_wave


def test_generator():
    mesh = pv.read("rook.obj")
    gen = WaveGenerator(mesh)
    ground_state = gen.generate_ground_mode()
    mesh.point_data["ground_state"] = ground_state
    mesh.point_data["s0"] = gen.generate_pure_mode(0.0)
    mesh.point_data["s1"] = gen.generate_pure_mode(1.0)
    mesh.point_data["s2"] = gen.generate_pure_mode(2.0)
    mesh.point_data["s3"] = gen.generate_pure_mode(3.0)
    for ii in range(1, 10):
        wave1 = gen.generate_wave(0.2*ii)
        mesh.point_data["wave{}".format(ii)] = wave1
    mesh.save("generated.vtp")


def wave_to_rook(
    mesh: pv.PolyData,
    ground_state: np.ndarray,
    wave: np.ndarray,
    smoothing_iterations=20,
    big_width=5.0,
    min_thickness = 0.4,
    collar_guarantee_size = 0.35,
    survival_level = 0.3
) -> pv.PolyData:
    base_tone = np.sqrt(ground_state)
    waist_coat = big_width * (1 - 0.5 * base_tone)
    hunger_pattern = rescale_range(survival_level - np.square(wave), -1, 1)
    height_precollar = waist_coat * hunger_pattern
    cap_guarantee = np.clip(
        2 * min_thickness * (collar_guarantee_size - ground_state) + min_thickness,
        0.0,
        np.inf)
    height = np.maximum(height_precollar, cap_guarantee)
    mesh.point_data["height"] = height
    extruded = extrude_triangles_to_wedges(
        mesh,
        offset_field_name="height",
        min_thickness=min_thickness,
        smoothing_iterations=20
    )
    biggest = extruded.extract_largest()
    smoothed = biggest.smooth(n_iter=smoothing_iterations)
    normaled = smoothed.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, split_vertices=False)
    return normaled


def test_cyl_0():
    mesh = pv.read("rook.obj")
    waver = WaveGenerator(mesh)
    ground_state = waver.generate_ground_mode()
    ww = waver.generate_wave(11) + 0.3 * waver.generate_wave(21)
    ww = rescale_range(ww, -1, 1)
    rook = wave_to_rook(mesh=mesh, ground_state=ground_state, wave=ww)
    rook.save("waved.vtp")


def suggested_sampler():
    mesh = pv.read("rook.obj")
    print("Building generator")
    mode_rank_limit = mesh.n_points / 28
    lower_bounds = [1.0,  15.0, -.3]
    upper_bounds = [0.3 * mode_rank_limit, mode_rank_limit, .3]
    sampler = qmc.Sobol(d=3)
    num_samples = 100
    samples_prescaled = sampler.random(num_samples)
    samples = qmc.scale(samples_prescaled, lower_bounds, upper_bounds)
    waver = WaveGenerator(mesh)
    ground_state=waver.generate_ground_mode()
    for foo, sample in enumerate(samples):
        print("Sample {}".format(foo))
        wave = waver.generate_wave(sample[0]) + sample[2] * waver.generate_wave(sample[1])
        rook = wave_to_rook(mesh=mesh, ground_state=ground_state, wave=wave)
        rook.save(f"generated/rook_{foo}.vtp")

if __name__ == "__main__":
    suggested_sampler()
