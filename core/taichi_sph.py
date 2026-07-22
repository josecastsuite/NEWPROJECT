"""Taichi-based Position Based Fluid (PBF) flow simulator for JoseCast.

This module provides a real-time 3-D free-surface particle simulation that runs
on the GPU (CUDA/Vulkan) or CPU via Taichi.  It is intentionally separate from
the Darcy engineering solver: Darcy still supplies the authoritative node
velocities and fill time, while this module produces a visually convincing
"water from a faucet" animation and, if desired, per-section velocity/flow-rate
samples for comparison.

The core is a 3-D extension of the Taichi PBF example (Macklin & Müller 2013)
with static boundary particles sampled from the voxel grid and a continuous
inlet emitter at the top of the sprue/pouring basin.
"""

from typing import List, Optional, Tuple

import numpy as np
import taichi as ti
from scipy import ndimage

from core.types import (
    BODY_CASTING_METAL_TYPES,
    AnalysisResult,
    Body,
    BodyType,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
_GATING_TYPES = [
    BodyType.SPRUE,
    BodyType.SPRUE_THROAT,
    BodyType.POURING_BASIN,
    BodyType.RUNNER,
    BodyType.DISTRIBUTOR,
    BodyType.INGATE,
    BodyType.CURUFLUK,
]


@ti.data_oriented
class TaichiFlowSimulator:
    """Real-time particle flow simulation for casting.

    Parameters
    ----------
    result : AnalysisResult
        Completed analysis that carries the voxel grid, origin and dx.
    particle_radius_mm : float
        Radius of one SPH particle in millimetres.
    inlet_velocity_m_s : float
        Inlet (sprue/pouring basin) velocity in m/s.  If None, it is derived
        from ``result.flow_result.Q_m3_s / inlet_area_m2``.
    max_fluid_particles : int
        Hard cap on the number of fluid particles.
    max_boundary_particles : int
        Hard cap on boundary (mold/core) particles.
    pbf_num_iters : int
        Number of PBF solver iterations per substep.
    gravity_m_s2 : float
        Gravity magnitude; direction is taken from the analysis gravity vector.
    """

    def __init__(
        self,
        result: AnalysisResult,
        particle_radius_mm: float = 1.5,
        inlet_velocity_m_s: Optional[float] = None,
        max_fluid_particles: int = 60000,
        max_boundary_particles: int = 120000,
        pbf_num_iters: int = 3,
        gravity_m_s2: float = 9.81,
    ):
        # Initialize Taichi once per process; prefer CUDA/Vulkan, fall back to CPU.
        if ti.lang.impl.get_runtime().prog is None:
            initialized = False
            for arch in (ti.cuda, ti.vulkan, ti.cpu):
                try:
                    ti.init(arch=arch)
                    initialized = True
                    break
                except Exception:
                    continue
            if not initialized:
                ti.init(arch=ti.cpu)

        self.result = result
        self.particle_radius_mm = float(particle_radius_mm)
        self.spacing_mm = 2.0 * self.particle_radius_mm
        self.h = 4.0 * self.particle_radius_mm  # support / neighbor radius
        self.neighbor_radius = self.h * 1.05
        self.cell_size = self.neighbor_radius
        self.cell_recpr = 1.0 / self.cell_size
        self.pbf_num_iters = max(1, int(pbf_num_iters))
        self.gravity_m_s2 = float(gravity_m_s2)

        # Time step for one PBF substep (seconds).  CFL-based.
        # dt < 0.4 * h / v_max.  With h in mm and v_max ~ 5 m/s = 5000 mm/s:
        # dt < 0.4*6 / 5000 ~ 5e-4.  We use a conservative default and scale
        # the number of substeps per user-frame.
        self.pbf_dt_s = 4.0e-4

        # Voxel grid data.
        self.grid = np.asarray(result.grid, dtype=np.int16)
        self.origin = np.asarray(result.origin_mm, dtype=np.float64)
        self.dx = float(result.dx_mm)
        self.shape = self.grid.shape  # (nx, ny, nz)

        bbox_min = self.origin
        bbox_max = self.origin + np.array(self.shape) * self.dx
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.domain_size = bbox_max - bbox_min

        # Inlet parameters.
        fr = result.flow_result
        if fr is not None:
            self.Q_m3_s = float(fr.Q_m3_s) if fr.Q_m3_s > 0 else 0.0
            self.inlet_area_m2 = float(fr.inlet_area_m2) if fr.inlet_area_m2 > 0 else 0.0
            if inlet_velocity_m_s is None and self.inlet_area_m2 > 0:
                inlet_velocity_m_s = self.Q_m3_s / self.inlet_area_m2
        else:
            self.Q_m3_s = 0.0
            self.inlet_area_m2 = 0.0
        self.inlet_velocity_m_s = float(inlet_velocity_m_s or 0.0)
        self.inlet_velocity_mm_s = self.inlet_velocity_m_s * 1000.0

        # Gravity direction from result, default -Y.
        if result.casting_params is not None:
            g_vec = np.asarray(result.casting_params.gravity_vector, dtype=np.float64)
        else:
            g_vec = np.asarray(getattr(result, "gravity", [0.0, -1.0, 0.0]), dtype=np.float64)
        g_norm = np.linalg.norm(g_vec)
        if g_norm < 1e-9:
            g_vec = np.array([0.0, -1.0, 0.0])
            g_norm = 1.0
        self.gravity_dir = g_vec / g_norm
        self.gravity_vec_mm_s2 = self.gravity_dir * self.gravity_m_s2 * 1000.0

        # Masks and source positions (numpy).
        self._build_masks()
        self._build_source_info()

        # Particle counts.
        self.max_fluid = max(1, int(max_fluid_particles))
        self.max_boundary = max(1, int(max_boundary_particles))
        self.max_total = self.max_fluid + self.max_boundary

        # Allocate Taichi fields and initialize.
        self._allocate_fields()
        self._init_particles()

        self._time_elapsed_s = 0.0
        self._emit_accumulator = 0.0

    # -----------------------------------------------------------------------
    # Geometry preprocessing
    # -----------------------------------------------------------------------
    def _build_masks(self) -> None:
        """Create metal/gating/boundary masks from the voxel grid."""
        solid = self.grid > 0
        metal = np.isin(self.grid, [t.value for t in BODY_CASTING_METAL_TYPES])
        gating = metal & np.isin(
            self.grid, [t.value for t in _GATING_TYPES]
        )

        # Boundary shell: a few voxels thick around every solid voxel.
        # This shell sits inside the mold and around cores/chills.
        iters = max(1, int(np.ceil(self.h / self.dx)) + 1)
        dilated_solid = ndimage.binary_dilation(solid, iterations=iters)
        boundary = dilated_solid & ~solid

        # Remove the source opening from the boundary shell so metal can pour in.
        source_top = self._top_surface_mask(gating)
        source_open = np.zeros_like(boundary)
        if source_top.any():
            source_open = self._source_opening_mask(source_top, layers=iters)
        boundary = boundary & ~source_open

        # Obstacles (CORE, FILTER, CHILL) are solid but not metal; treat them
        # as boundary particles too so metal flows around them.
        obstacle = solid & ~metal

        self.metal_mask = metal
        self.gating_mask = gating
        self.part_mask = metal & (self.grid == BodyType.PART.value)
        self.riser_mask = metal & (self.grid == BodyType.RISER.value)
        self.boundary_mask = boundary | obstacle
        self.source_top_mask = source_top

    def _top_surface_mask(self, gating: np.ndarray) -> np.ndarray:
        """Return gating cells whose +Y neighbor is empty."""
        ny = self.shape[1]
        # Pad gating with False at the +Y face.
        padded = np.pad(gating, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=False)
        above = padded[:, 1:, :]
        top = gating & ~above
        return top

    def _source_opening_mask(self, source_top: np.ndarray, layers: int) -> np.ndarray:
        """Mark the vertical shaft directly above every source-top voxel as
        an opening in the boundary shell."""
        nx, ny, nz = self.shape
        open_mask = np.zeros_like(source_top)
        src_idx = np.argwhere(source_top)
        for (i, j, k) in src_idx:
            j1 = min(ny - 1, j + layers)
            open_mask[i, j:j1 + 1, k] = True
        # Dilate slightly in X/Z so the stream has room.
        open_mask = ndimage.binary_dilation(open_mask, iterations=1)
        return open_mask

    def _build_source_info(self) -> None:
        """Compute the world-space source emission points."""
        if not self.source_top_mask.any():
            # Fallback: emit from the very top of the metal domain.
            metal_idx = np.argwhere(self.metal_mask)
            if metal_idx.size == 0:
                self.source_positions_mm = np.empty((0, 3), dtype=np.float64)
                return
            jmax = metal_idx[:, 1].max()
            top_idx = metal_idx[metal_idx[:, 1] == jmax]
        else:
            top_idx = np.argwhere(self.source_top_mask)

        # Top face centre, offset by half a particle spacing above the face.
        src = top_idx * self.dx + self.origin + np.array([self.dx / 2.0, self.dx, self.dx / 2.0])
        src[:, 1] += self.spacing_mm * 0.5
        self.source_positions_mm = src.astype(np.float64)

    # -----------------------------------------------------------------------
    # Taichi field allocation
    # -----------------------------------------------------------------------
    def _allocate_fields(self) -> None:
        """Allocate the Taichi fields.  Must be called after Ti init."""
        # Grid for spatial hashing.
        grid_num = np.ceil(self.domain_size / self.cell_size).astype(np.int32)
        self.grid_size = tuple(int(x) for x in grid_num)
        max_per_cell = 50
        max_neighbors = 80

        self.max_particles_per_cell = int(max_per_cell)
        self.max_neighbors = int(max_neighbors)

        # Particle fields.
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.max_total)
        self.old_positions = ti.Vector.field(3, dtype=ti.f32, shape=self.max_total)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.max_total)
        self.position_deltas = ti.Vector.field(3, dtype=ti.f32, shape=self.max_total)
        self.lambdas = ti.field(dtype=ti.f32, shape=self.max_total)
        self.is_fluid = ti.field(dtype=ti.i32, shape=self.max_total)
        self.is_active = ti.field(dtype=ti.i32, shape=self.max_total)

        # Neighbor lookup.
        self.grid_num_particles = ti.field(dtype=ti.i32, shape=self.grid_size)
        self.grid2particles = ti.field(
            dtype=ti.i32,
            shape=self.grid_size + (self.max_particles_per_cell,),
        )
        self.particle_num_neighbors = ti.field(dtype=ti.i32, shape=self.max_total)
        self.particle_neighbors = ti.field(
            dtype=ti.i32,
            shape=(self.max_total, self.max_neighbors),
        )

        # SDF field for optional collision safety.
        self.sdf_field = ti.field(dtype=ti.f32, shape=self.shape)

    # -----------------------------------------------------------------------
    # Particle setup
    # -----------------------------------------------------------------------
    def _sample_mask(self, mask: np.ndarray) -> np.ndarray:
        """Sample a regular lattice inside a boolean mask."""
        if not mask.any():
            return np.empty((0, 3), dtype=np.float32)
        x = np.arange(self.bbox_min[0] + self.spacing_mm / 2.0,
                      self.bbox_max[0], self.spacing_mm)
        y = np.arange(self.bbox_min[1] + self.spacing_mm / 2.0,
                      self.bbox_max[1], self.spacing_mm)
        z = np.arange(self.bbox_min[2] + self.spacing_mm / 2.0,
                      self.bbox_max[2], self.spacing_mm)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        # Map to voxel indices (nearest).
        ijk = np.floor((pts - self.origin) / self.dx).astype(np.int32)
        inside = (
            (ijk >= 0).all(axis=1)
            & (ijk[:, 0] < self.shape[0])
            & (ijk[:, 1] < self.shape[1])
            & (ijk[:, 2] < self.shape[2])
        )
        keep = np.zeros(pts.shape[0], dtype=bool)
        valid = inside
        keep[valid] = mask[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]]
        return pts[keep]

    def _init_particles(self) -> None:
        """Upload initial fluid + boundary particles to Taichi."""
        fluid_pts = self._sample_mask(self.gating_mask)
        boundary_pts = self._sample_mask(self.boundary_mask)

        n_fluid = min(fluid_pts.shape[0], self.max_fluid)
        n_boundary = min(boundary_pts.shape[0], self.max_boundary)

        self.n_fluid = 0
        self.n_boundary = 0

        pos = np.zeros((self.max_total, 3), dtype=np.float32)
        vel = np.zeros((self.max_total, 3), dtype=np.float32)
        is_fluid = np.zeros(self.max_total, dtype=np.int32)
        is_active = np.zeros(self.max_total, dtype=np.int32)

        if n_fluid > 0:
            pos[:n_fluid] = fluid_pts[:n_fluid]
            is_fluid[:n_fluid] = 1
            is_active[:n_fluid] = 1
            self.n_fluid = n_fluid
        if n_boundary > 0:
            start = self.max_fluid
            pos[start:start + n_boundary] = boundary_pts[:n_boundary]
            is_fluid[start:start + n_boundary] = 0
            is_active[start:start + n_boundary] = 1
            self.n_boundary = n_boundary

        # Source starts filling after the pre-filled gating column is ready.
        self._next_emit_idx = self.n_fluid

        self.positions.from_numpy(pos)
        self.old_positions.from_numpy(pos)
        self.velocities.from_numpy(vel)
        self.is_fluid.from_numpy(is_fluid)
        self.is_active.from_numpy(is_active)

        # SDF: inside metal is negative, outside positive.
        dist_in = ndimage.distance_transform_edt(self.metal_mask)
        dist_out = ndimage.distance_transform_edt(~self.metal_mask)
        sdf = dist_out.astype(np.float32) - dist_in.astype(np.float32)
        self.sdf_field.from_numpy(sdf)

        # Taichi constants for kernels.
        self._kernel_h = float(self.h)
        self._kernel_neighbor_radius = float(self.neighbor_radius)
        self._kernel_cell_recpr = float(self.cell_recpr)
        self._kernel_gravity = tuple(self.gravity_vec_mm_s2.astype(np.float32))
        self._kernel_origin = tuple(self.origin.astype(np.float32))
        self._kernel_dx = float(self.dx)
        self._kernel_bbox_min = tuple(self.bbox_min.astype(np.float32))
        self._kernel_bbox_max = tuple(self.bbox_max.astype(np.float32))
        self._kernel_eps = 1e-5

        self.dt_field = ti.field(dtype=ti.f32, shape=())
        self.dt_field[None] = float(self.pbf_dt_s)

    # -----------------------------------------------------------------------
    # Emission
    # -----------------------------------------------------------------------
    def emit(self, dt_s: float) -> None:
        """Add new fluid particles from the inlet."""
        if self.source_positions_mm.shape[0] == 0 or self.n_fluid >= self.max_fluid:
            return
        particle_volume_m3 = (4.0 / 3.0) * np.pi * (self.particle_radius_mm / 1000.0) ** 3
        rate = self.Q_m3_s / max(particle_volume_m3, 1e-18)
        self._emit_accumulator += rate * dt_s
        n_emit = int(self._emit_accumulator)
        if n_emit <= 0:
            return
        self._emit_accumulator -= n_emit

        n_emit = min(n_emit, self.max_fluid - self.n_fluid)
        if n_emit <= 0:
            return

        src = self.source_positions_mm
        idx = np.random.randint(0, src.shape[0], size=n_emit)
        spawn = src[idx].copy()
        # Small in-plane jitter and downward velocity.
        jitter = (np.random.rand(n_emit, 3) - 0.5) * self.spacing_mm * 0.5
        jitter[:, 1] = 0.0
        spawn += jitter
        spawn[:, 1] += (np.random.rand(n_emit)) * self.spacing_mm

        vel = self.gravity_dir * self.inlet_velocity_mm_s
        vel = np.tile(vel, (n_emit, 1)).astype(np.float32)

        start = self._next_emit_idx
        end = start + n_emit
        # Wrap around if needed (we never delete here, so end should stay within max_fluid).
        if end > self.max_fluid:
            n_emit = self.max_fluid - start
            if n_emit <= 0:
                return
            end = self.max_fluid
            spawn = spawn[:n_emit]
            vel = vel[:n_emit]

        pos_np = self.positions.to_numpy()
        vel_np = self.velocities.to_numpy()
        active_np = self.is_active.to_numpy()
        fluid_np = self.is_fluid.to_numpy()
        pos_np[start:end] = spawn.astype(np.float32)
        vel_np[start:end] = vel
        active_np[start:end] = 1
        fluid_np[start:end] = 1
        self.positions.from_numpy(pos_np)
        self.velocities.from_numpy(vel_np)
        self.is_active.from_numpy(active_np)
        self.is_fluid.from_numpy(fluid_np)

        self.n_fluid = max(self.n_fluid, end)
        self._next_emit_idx = end

    # -----------------------------------------------------------------------
    # Simulation kernels
    # -----------------------------------------------------------------------
    def step(self, dt_s: float) -> None:
        """Advance the simulation by ``dt_s`` seconds."""
        if self.n_fluid == 0:
            return

        self.emit(dt_s)
        substeps = max(1, int(np.ceil(dt_s / self.pbf_dt_s)))
        sub_dt = dt_s / substeps
        self.dt_field[None] = float(sub_dt)

        for _ in range(substeps):
            self._prologue()
            for _ in range(self.pbf_num_iters):
                self._substep()
            self._epilogue()

        self._time_elapsed_s += dt_s

    @ti.kernel
    def _prologue(self):
        g = ti.Vector(self._kernel_gravity)
        dt = self.dt_field[None]
        eps = self._kernel_eps
        bmin = ti.Vector(self._kernel_bbox_min)
        bmax = ti.Vector(self._kernel_bbox_max)
        pr = self.particle_radius_mm

        # Save positions and integrate active fluid particles.
        for i in range(self.max_total):
            if self.is_fluid[i] == 0 or self.is_active[i] == 0:
                continue
            self.old_positions[i] = self.positions[i]
            vel = self.velocities[i] + g * dt
            pos = self.positions[i] + vel * dt
            # Simple bbox reflection.
            for d in ti.static(range(3)):
                if pos[d] < bmin[d] + pr:
                    pos[d] = bmin[d] + pr + eps * ti.random()
                    vel[d] *= -0.3
                elif pos[d] > bmax[d] - pr:
                    pos[d] = bmax[d] - pr - eps * ti.random()
                    vel[d] *= -0.3
            self.positions[i] = pos
            self.velocities[i] = vel

        # Clear neighbor structures.
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # Update grid and find neighbors (include boundary particles).
        for i in range(self.max_total):
            if self.is_active[i] == 0:
                continue
            cell = self._get_cell(self.positions[i])
            if self._is_in_grid(cell):
                offs = ti.atomic_add(self.grid_num_particles[cell], 1)
                if offs < self.max_particles_per_cell:
                    self.grid2particles[cell, offs] = i

        for i in range(self.max_total):
            if self.is_active[i] == 0:
                continue
            pos_i = self.positions[i]
            cell = self._get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                cell_to_check = cell + offs
                if self._is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        if nb_i >= self.max_neighbors:
                            break
                        p_j = self.grid2particles[cell_to_check, j]
                        if p_j == i:
                            continue
                        if (pos_i - self.positions[p_j]).norm() < self._kernel_neighbor_radius:
                            self.particle_neighbors[i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[i] = nb_i

    @ti.func
    def _poly6_value(self, s: ti.f32) -> ti.f32:
        h = self._kernel_h
        result = 0.0
        if 0.0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = (315.0 / (64.0 * 3.141592653589793)) * x * x * x
        return result

    @ti.func
    def _spiky_gradient(self, r: ti.template()) -> ti.template():
        h = self._kernel_h
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0.0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = (-45.0 / 3.141592653589793) * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def _compute_scorr(self, pos_ji: ti.template()) -> ti.f32:
        coeff = 0.3  # corr_deltaQ_coeff
        kappa = 0.001  # corrK
        num = self._poly6_value(pos_ji.norm())
        den = self._poly6_value(coeff * self._kernel_h)
        ret = 0.0
        if den > 0.0:
            x = num / den
            x = x * x
            x = x * x
            ret = -kappa * x
        return ret

    @ti.kernel
    def _substep(self):
        rho0 = 1.0
        mass = 1.0
        h = self._kernel_h
        eps = 100.0  # lambda_epsilon

        # Compute lambdas for active fluid particles only.
        for i in range(self.max_total):
            if self.is_fluid[i] == 0 or self.is_active[i] == 0:
                continue
            pos_i = self.positions[i]
            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_grad_sqr = 0.0
            density_constraint = 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                grad_j = self._spiky_gradient(pos_ji)
                grad_i += grad_j
                sum_grad_sqr += grad_j.dot(grad_j)
                density_constraint += self._poly6_value(pos_ji.norm())

            density_constraint = (mass * density_constraint / rho0) - 1.0
            sum_grad_sqr += grad_i.dot(grad_i)
            self.lambdas[i] = (-density_constraint) / (sum_grad_sqr + eps)

        # Compute and apply position deltas for active fluid only.
        for i in range(self.max_total):
            if self.is_fluid[i] == 0 or self.is_active[i] == 0:
                continue
            pos_i = self.positions[i]
            lambda_i = self.lambdas[i]
            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i - self.positions[p_j]
                scorr = self._compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr) * self._spiky_gradient(pos_ji)
            pos_delta_i /= rho0
            self.positions[i] += pos_delta_i

    @ti.kernel
    def _epilogue(self):
        eps = self._kernel_eps
        pr = self.particle_radius_mm
        bmin = ti.Vector(self._kernel_bbox_min)
        bmax = ti.Vector(self._kernel_bbox_max)

        for i in range(self.max_total):
            if self.is_fluid[i] == 0 or self.is_active[i] == 0:
                continue
            pos = self.positions[i]
            # Bbox confinement.
            for d in ti.static(range(3)):
                if pos[d] < bmin[d] + pr:
                    pos[d] = bmin[d] + pr + eps * ti.random()
                elif pos[d] > bmax[d] - pr:
                    pos[d] = bmax[d] - pr - eps * ti.random()
            self.positions[i] = pos
            self.velocities[i] = (self.positions[i] - self.old_positions[i]) / self.dt_field[None]

    @ti.func
    def _get_cell(self, pos: ti.template()) -> ti.Vector:
        return (pos * self._kernel_cell_recpr).cast(ti.i32)

    @ti.func
    def _is_in_grid(self, c: ti.template()) -> ti.i32:
        return int(
            (c[0] >= 0)
            and (c[0] < self.grid_size[0])
            and (c[1] >= 0)
            and (c[1] < self.grid_size[1])
            and (c[2] >= 0)
            and (c[2] < self.grid_size[2])
        )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    def get_fluid_particles(self) -> np.ndarray:
        """Return active fluid particle positions (mm)."""
        pos = self.positions.to_numpy()
        active = self.is_active.to_numpy().astype(bool)
        fluid = self.is_fluid.to_numpy().astype(bool)
        mask = active & fluid
        return pos[mask]

    def get_fluid_particles_with_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for active fluid particles (mm, mm/s)."""
        pos = self.positions.to_numpy()
        vel = self.velocities.to_numpy()
        active = self.is_active.to_numpy().astype(bool)
        fluid = self.is_fluid.to_numpy().astype(bool)
        mask = active & fluid
        return pos[mask], vel[mask]

    def get_all_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, is_fluid) for active particles."""
        pos = self.positions.to_numpy()
        active = self.is_active.to_numpy().astype(bool)
        return pos[active], self.is_fluid.to_numpy()[active]

    def sample_section_flux(
        self,
        plane_origin_mm: np.ndarray,
        plane_normal: np.ndarray,
        radius_mm: float = 5.0,
    ) -> Tuple[float, float]:
        """Estimate velocity magnitude (m/s) and flow rate (m³/s) across a plane."""
        pos = self.get_fluid_particles()
        if pos.shape[0] == 0:
            return 0.0, 0.0
        vel = self.velocities.to_numpy()
        active = self.is_active.to_numpy().astype(bool)
        fluid = self.is_fluid.to_numpy().astype(bool)
        p = pos[active & fluid]
        v = vel[active & fluid]
        n = np.asarray(plane_normal, dtype=np.float64)
        n /= np.linalg.norm(n) + 1e-12
        d = p - plane_origin_mm
        dist = np.linalg.norm(d - np.dot(d, n)[:, None] * n, axis=1)
        near = dist < radius_mm
        if not near.any():
            return 0.0, 0.0
        vn = np.dot(v[near], n)
        # Average magnitude weighted by normal component sign.
        v_mag = float(np.mean(np.abs(vn))) / 1000.0  # mm/s -> m/s
        area_m2 = np.pi * (radius_mm / 1000.0) ** 2
        return v_mag, v_mag * area_m2
